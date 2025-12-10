"""
TokenAlign Loss Functions: Max-Sum token similarity and hybrid objective
Based on the second innovation point in the TokenAlign paper.

This module implements:
1. Max-Sum token similarity computation
2. Online hard negative mining
3. Hardest-triplet loss (L_h)
4. Intra-modal consistency loss (L_a)
5. Hybrid objective: L_total = L_h + λ * L_a
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def max_sum_token_similarity(text_features, audio_features, attention_weights=None):
    """
    Compute Max-Sum token similarity based on alignment information.
    
    For each text token, find the maximum match along the audio timeline,
    then sum these maximum scores to obtain the global similarity.
    
    Args:
        text_features: (batch, N, dim) - text token features (after TokenAlign enhancement)
        audio_features: (batch, M, dim) - audio segment features
        attention_weights: (batch, N, M) - optional attention weights from TokenAlign
    
    Returns:
        similarity_scores: (batch, batch) - similarity matrix S(F_a, F_t)
    """
    batch_size = text_features.shape[0]
    device = text_features.device
    
    # If attention weights are provided, use them for alignment
    # Otherwise, compute dot product similarity
    if attention_weights is not None:
        # Use attention-weighted similarity
        # For each text token, find max over audio segments
        # attention_weights: (batch, N, M)
        # text_features: (batch, N, dim), audio_features: (batch, M, dim)
        
        # Compute token-level similarities: (batch, N, M)
        token_similarities = torch.matmul(text_features, audio_features.transpose(1, 2))  # (batch, N, M)
        
        # Weight by attention: element-wise multiplication
        weighted_similarities = token_similarities * attention_weights  # (batch, N, M)
        
        # Max-Sum: max over audio dimension, then sum over text dimension
        max_similarities = weighted_similarities.max(dim=2)[0]  # (batch, N) - max over audio
        global_similarities = max_similarities.sum(dim=1)  # (batch,) - sum over text tokens
        
        # Expand to batch x batch matrix for pairwise comparison
        # For each (i, j) pair, compute similarity between audio_i and text_j
        similarity_matrix = torch.zeros(batch_size, batch_size, device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                # Compute similarity between audio_i and text_j
                token_sim = torch.matmul(text_features[j:j+1], audio_features[i:i+1].transpose(1, 2))  # (1, N, M)
                if attention_weights is not None:
                    weighted_sim = token_sim * attention_weights[j:j+1]  # (1, N, M)
                    max_sim = weighted_sim.max(dim=2)[0]  # (1, N)
                    similarity_matrix[i, j] = max_sim.sum(dim=1)  # scalar
                else:
                    max_sim = token_sim.max(dim=2)[0]  # (1, N)
                    similarity_matrix[i, j] = max_sim.sum(dim=1)  # scalar
    else:
        # Simple case: compute similarity without attention weights
        # For each pair (audio_i, text_j), compute Max-Sum similarity
        similarity_matrix = torch.zeros(batch_size, batch_size, device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                # Compute token-level similarity: (N, M)
                token_sim = torch.matmul(text_features[j], audio_features[i].T)  # (N, M)
                # Max over audio dimension, then sum over text dimension
                max_sim = token_sim.max(dim=1)[0]  # (N,)
                similarity_matrix[i, j] = max_sim.sum()  # scalar
    
    return similarity_matrix


def max_sum_token_similarity_efficient(text_features, audio_features, attention_weights=None, tau=1.0):
    """
    Compute Local Alignment (Token-Level) similarity using Max-Sum strategy.
    
    According to paper: S_Local(F_a, F_t) = τ * Σ_{j=1}^{N} max_{i∈[1,M]} S_ij(a, t)
    where S_ij is cosine similarity between audio segment i and text token j.
    
    Args:
        text_features: (batch, N, dim) - text token features F_t
        audio_features: (batch, M, dim) - audio segment features F_a
        attention_weights: (batch, N, M) - optional attention weights from TokenAlign
        tau: Scaling factor τ (default: 1.0)
    
    Returns:
        similarity_matrix: (batch, batch) - similarity matrix S(F_a, F_t)
                          where S[i, j] = S_Local(F_a_i, F_t_j)
    """
    batch_size, text_len, dim = text_features.shape
    _, audio_len, _ = audio_features.shape
    device = text_features.device
    
    # Normalize features for cosine similarity
    text_norm = F.normalize(text_features, p=2, dim=-1)
    audio_norm = F.normalize(audio_features, p=2, dim=-1)
    
    # Compute similarity matrix for all batch pairs
    similarity_matrix = torch.zeros(batch_size, batch_size, device=device)
    
    for i in range(batch_size):
        for j in range(batch_size):
            # Compute cosine similarity matrix: S_ij = (a_i^T * t_j) / (||a_i|| * ||t_j||)
            # text_norm[j]: (text_len, dim)
            # audio_norm[i]: (audio_len, dim)
            similarity_mat = torch.matmul(audio_norm[i], text_norm[j].T)  # (audio_len, text_len)
            
            # Apply attention weights if provided (from TokenAlign)
            if attention_weights is not None:
                # attention_weights[j]: (text_len, audio_len)
                # We need to transpose to match dimensions
                attn_mat = attention_weights[j].T  # (audio_len, text_len)
                similarity_mat = similarity_mat * attn_mat
            
            # Max-Sum: for each text token j, find max over audio segments i
            # Then sum over all text tokens
            max_similarities = similarity_mat.max(dim=0)[0]  # (text_len,) - max over audio for each text token
            local_sim = max_similarities.sum() * tau  # scalar - sum over all text tokens with scaling
            
            similarity_matrix[i, j] = local_sim
    
    return similarity_matrix


def online_hard_negative_mining(similarity_matrix, positive_mask):
    """
    Identify hardest negatives for each positive pair.
    
    Args:
        similarity_matrix: (batch, batch) - similarity scores S(F_a, F_t)
        positive_mask: (batch, batch) - boolean mask indicating positive pairs
    
    Returns:
        hardest_audio_negatives: (batch,) - indices of hardest audio negatives
        hardest_text_negatives: (batch,) - indices of hardest text negatives
    """
    batch_size = similarity_matrix.shape[0]
    device = similarity_matrix.device
    
    hardest_audio_negatives = torch.zeros(batch_size, dtype=torch.long, device=device)
    hardest_text_negatives = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for i in range(batch_size):
        # For positive pair (audio_i, text_i), find hardest negatives
        # Hardest audio negative: argmax_{j≠i} S(F_{a,j}, F_{t,i})
        text_similarities = similarity_matrix[:, i]  # (batch,) - similarities to text_i
        # Mask out positive pairs (j == i)
        negative_mask = ~positive_mask[:, i]
        if negative_mask.any():
            text_similarities_masked = text_similarities.clone()
            text_similarities_masked[~negative_mask] = float('-inf')
            hardest_audio_negatives[i] = text_similarities_masked.argmax()
        else:
            # Fallback: use any non-positive
            text_similarities_masked = text_similarities.clone()
            text_similarities_masked[i] = float('-inf')
            hardest_audio_negatives[i] = text_similarities_masked.argmax()
        
        # Hardest text negative: argmax_{k≠i} S(F_{a,i}, F_{t,k})
        audio_similarities = similarity_matrix[i, :]  # (batch,) - similarities from audio_i
        # Mask out positive pairs (k == i)
        negative_mask = ~positive_mask[i, :]
        if negative_mask.any():
            audio_similarities_masked = audio_similarities.clone()
            audio_similarities_masked[~negative_mask] = float('-inf')
            hardest_text_negatives[i] = audio_similarities_masked.argmax()
        else:
            # Fallback: use any non-positive
            audio_similarities_masked = audio_similarities.clone()
            audio_similarities_masked[i] = float('-inf')
            hardest_text_negatives[i] = audio_similarities_masked.argmax()
    
    return hardest_audio_negatives, hardest_text_negatives


class TokenAlignHybridLoss(nn.Module):
    """
    Hybrid loss function for TokenAlign:
    - Hardest-triplet loss (L_h)
    - Intra-modal consistency loss (L_a)
    - Total: L_total = L_h + λ * L_a
    """
    
    def __init__(self, margin=0.2, lambda_consistency=0.1, sigma_margin=0.0):
        """
        Args:
            margin (float): Margin δ for triplet loss (default: 0.2)
            lambda_consistency (float): Weight λ for consistency loss (default: 0.1)
            sigma_margin (float): Relaxation hyperparameter σ_margin for consistency loss (default: 0.0)
        """
        super(TokenAlignHybridLoss, self).__init__()
        self.margin = margin
        self.lambda_consistency = lambda_consistency
        self.sigma_margin = sigma_margin
    
    def forward(self, similarity_matrix, positive_mask, 
                hardest_audio_negatives, hardest_text_negatives):
        """
        Compute hybrid loss.
        
        Args:
            similarity_matrix: (batch, batch) - similarity scores S(F_a, F_t)
            positive_mask: (batch, batch) - boolean mask for positive pairs
            hardest_audio_negatives: (batch,) - indices of hardest audio negatives
            hardest_text_negatives: (batch,) - indices of hardest text negatives
        
        Returns:
            total_loss: scalar - total loss L_total
            triplet_loss: scalar - triplet loss L_h
            consistency_loss: scalar - consistency loss L_a
        """
        batch_size = similarity_matrix.shape[0]
        device = similarity_matrix.device
        
        # Extract positive similarities: S(F_a, F_t) for positive pairs
        positive_similarities = similarity_matrix[positive_mask]  # (num_positives,)
        
        # Hardest-triplet loss L_h
        triplet_losses = []
        for i in range(batch_size):
            if positive_mask[i, i]:  # Check if (audio_i, text_i) is a positive pair
                # Positive similarity
                pos_sim = similarity_matrix[i, i]
                
                # Hardest audio negative: S(F_{a,hardest_audio_neg}, F_{t,i})
                hardest_audio_idx = hardest_audio_negatives[i]
                audio_neg_sim = similarity_matrix[hardest_audio_idx, i]
                
                # Hardest text negative: S(F_{a,i}, F_{t,hardest_text_neg})
                hardest_text_idx = hardest_text_negatives[i]
                text_neg_sim = similarity_matrix[i, hardest_text_idx]
                
                # Triplet loss terms
                # max(0, δ - S(F_a, F_t) + S(F_a, F_t^-))
                loss_text_neg = F.relu(self.margin - pos_sim + text_neg_sim)
                # max(0, δ - S(F_a, F_t) + S(F_a^-, F_t))
                loss_audio_neg = F.relu(self.margin - pos_sim + audio_neg_sim)
                
                triplet_losses.append(loss_text_neg + loss_audio_neg)
        
        if len(triplet_losses) > 0:
            triplet_loss = torch.stack(triplet_losses).mean()
        else:
            triplet_loss = torch.tensor(0.0, device=device)
        
        # Intra-modal consistency loss L_a
        consistency_losses = []
        for i in range(batch_size):
            if positive_mask[i, i]:
                hardest_audio_idx = hardest_audio_negatives[i]
                hardest_text_idx = hardest_text_negatives[i]
                
                # Audio-audio similarity: S(F_a, F_a^-)
                # We need to compute similarity between audio_i and audio_hardest_neg
                # For this, we need the audio features, but we only have the cross-modal similarity matrix
                # We'll approximate by using the similarity structure
                # S(F_a, F_a^-) ≈ similarity when both are matched to the same text
                # Actually, we need intra-modal similarities, which we don't have directly
                # Let's compute them from the cross-modal similarities
                # We'll use the fact that if two audios are similar, they should have similar similarities to texts
                
                # Approximate: S(F_a_i, F_a_hardest) by comparing their similarity patterns to all texts
                audio_i_pattern = similarity_matrix[i, :]  # (batch,)
                audio_hardest_pattern = similarity_matrix[hardest_audio_idx, :]  # (batch,)
                audio_audio_sim = F.cosine_similarity(
                    audio_i_pattern.unsqueeze(0), 
                    audio_hardest_pattern.unsqueeze(0)
                )[0]
                
                # Text-text similarity: S(F_t_i, F_t_hardest)
                text_i_pattern = similarity_matrix[:, i]  # (batch,)
                text_hardest_pattern = similarity_matrix[:, hardest_text_idx]  # (batch,)
                text_text_sim = F.cosine_similarity(
                    text_i_pattern.unsqueeze(0),
                    text_hardest_pattern.unsqueeze(0)
                )[0]
                
                # Consistency loss: |S(F_a, F_a^-) - S(F_t, F_t^-)| - σ_margin
                consistency_loss = F.relu(
                    torch.abs(audio_audio_sim - text_text_sim) - self.sigma_margin
                )
                consistency_losses.append(consistency_loss)
        
        if len(consistency_losses) > 0:
            consistency_loss = torch.stack(consistency_losses).mean()
        else:
            consistency_loss = torch.tensor(0.0, device=device)
        
        # Total loss: L_total = L_h + λ * L_a
        total_loss = triplet_loss + self.lambda_consistency * consistency_loss
        
        return total_loss, triplet_loss, consistency_loss


class TokenAlignHybridLossWithIntraModal(nn.Module):
    """
    Improved version that uses actual intra-modal similarities when available.
    """
    
    def __init__(self, margin=0.2, lambda_consistency=0.1, sigma_margin=0.0):
        super(TokenAlignHybridLossWithIntraModal, self).__init__()
        self.margin = margin
        self.lambda_consistency = lambda_consistency
        self.sigma_margin = sigma_margin
    
    def forward(self, similarity_matrix, positive_mask,
                hardest_audio_negatives, hardest_text_negatives,
                audio_audio_similarities=None, text_text_similarities=None):
        """
        Compute hybrid loss with optional intra-modal similarities.
        
        Args:
            similarity_matrix: (batch, batch) - cross-modal similarity S(F_a, F_t)
            positive_mask: (batch, batch) - boolean mask for positive pairs
            hardest_audio_negatives: (batch,) - indices of hardest audio negatives
            hardest_text_negatives: (batch,) - indices of hardest text negatives
            audio_audio_similarities: (batch, batch) - optional intra-modal audio similarities
            text_text_similarities: (batch, batch) - optional intra-modal text similarities
        """
        batch_size = similarity_matrix.shape[0]
        device = similarity_matrix.device
        
        # Hardest-triplet loss L_h (same as before)
        triplet_losses = []
        for i in range(batch_size):
            if positive_mask[i, i]:
                pos_sim = similarity_matrix[i, i]
                
                hardest_audio_idx = hardest_audio_negatives[i]
                audio_neg_sim = similarity_matrix[hardest_audio_idx, i]
                
                hardest_text_idx = hardest_text_negatives[i]
                text_neg_sim = similarity_matrix[i, hardest_text_idx]
                
                loss_text_neg = F.relu(self.margin - pos_sim + text_neg_sim)
                loss_audio_neg = F.relu(self.margin - pos_sim + audio_neg_sim)
                
                triplet_losses.append(loss_text_neg + loss_audio_neg)
        
        if len(triplet_losses) > 0:
            triplet_loss = torch.stack(triplet_losses).mean()
        else:
            triplet_loss = torch.tensor(0.0, device=device)
        
        # Intra-modal consistency loss L_a
        if audio_audio_similarities is not None and text_text_similarities is not None:
            consistency_losses = []
            for i in range(batch_size):
                if positive_mask[i, i]:
                    hardest_audio_idx = hardest_audio_negatives[i]
                    hardest_text_idx = hardest_text_negatives[i]
                    
                    # Use actual intra-modal similarities
                    audio_audio_sim = audio_audio_similarities[i, hardest_audio_idx]
                    text_text_sim = text_text_similarities[i, hardest_text_idx]
                    
                    consistency_loss = F.relu(
                        torch.abs(audio_audio_sim - text_text_sim) - self.sigma_margin
                    )
                    consistency_losses.append(consistency_loss)
            
            if len(consistency_losses) > 0:
                consistency_loss = torch.stack(consistency_losses).mean()
            else:
                consistency_loss = torch.tensor(0.0, device=device)
        else:
            # Fallback to approximation
            consistency_loss = torch.tensor(0.0, device=device)
        
        total_loss = triplet_loss + self.lambda_consistency * consistency_loss
        
        return total_loss, triplet_loss, consistency_loss

