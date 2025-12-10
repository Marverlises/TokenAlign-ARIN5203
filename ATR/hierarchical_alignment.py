"""
Hierarchical Alignment Module: Three-level alignment framework
Based on the updated TokenAlign paper.

This module implements:
1. Local Alignment (Token-Level): Max-Sum similarity
2. Regional Alignment (Event-Level): Audio event discovery + Optimal Transport
3. Global Alignment (Clip-Level): Global semantic consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from scipy.sparse.csgraph import laplacian
    from scipy.linalg import eigh
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Normalized Cuts will use fallback method.")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Clustering will use fallback method.")


def normalized_cuts_clustering(similarity_matrix, n_clusters):
    """
    Apply Normalized Cuts (NCuts) spectral clustering to partition the time dimension.
    
    Args:
        similarity_matrix: (T, T) - temporal self-similarity matrix
        n_clusters: Number of clusters/segments to partition into
    
    Returns:
        labels: (T,) - cluster labels for each time step
    """
    # Convert to numpy if needed
    if isinstance(similarity_matrix, torch.Tensor):
        S = similarity_matrix.detach().cpu().numpy()
    else:
        S = similarity_matrix
    
    # Ensure symmetric and non-negative
    S = (S + S.T) / 2
    S = np.maximum(S, 0)
    
    # Compute normalized Laplacian if scipy is available
    if SCIPY_AVAILABLE:
        try:
            L = laplacian(S, normed=True)
            # Compute first k eigenvectors (excluding the first trivial one)
            eigenvalues, eigenvectors = eigh(L, subset_by_index=[1, n_clusters])
            features = eigenvectors
        except:
            # Fallback: use similarity matrix directly
            features = S
    else:
        # Fallback: use similarity matrix directly
        features = S
    
    # K-means on features
    if SKLEARN_AVAILABLE:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(features)
    else:
        # Fallback: uniform partitioning
        labels = np.linspace(0, n_clusters - 1, len(S), dtype=int)
    
    return labels


def sinkhorn_knopp(cost_matrix, reg=0.1, max_iter=100, tol=1e-9):
    """
    Solve Optimal Transport problem using Sinkhorn-Knopp algorithm.
    
    Args:
        cost_matrix: (K, K) - cost matrix C_ij = 1 - sim(e_i, p_j)
        reg: Regularization parameter (default: 0.1)
        max_iter: Maximum iterations (default: 100)
        tol: Convergence tolerance (default: 1e-9)
    
    Returns:
        transport_plan: (K, K) - optimal transport plan Gamma*
    """
    device = cost_matrix.device
    K = cost_matrix.shape[0]
    
    # Initialize: uniform marginals
    u = torch.ones(K, device=device) / K
    v = torch.ones(K, device=device) / K
    
    # Sinkhorn iterations
    for _ in range(max_iter):
        u_prev = u.clone()
        
        # Update u
        K_matrix = torch.exp(-cost_matrix / reg)
        u = 1.0 / (K_matrix @ v + 1e-10)
        u = u / (u.sum() + 1e-10)
        
        # Update v
        v = 1.0 / (K_matrix.T @ u + 1e-10)
        v = v / (v.sum() + 1e-10)
        
        # Check convergence
        if torch.norm(u - u_prev) < tol:
            break
    
    # Compute transport plan
    transport_plan = torch.diag(u) @ K_matrix @ torch.diag(v)
    
    return transport_plan


class RegionalAlignment(nn.Module):
    """
    Regional Alignment module: Event-level alignment using Optimal Transport.
    
    This module:
    1. Discovers audio events using CutLER-inspired approach
    2. Extracts text phrases using attention-based mechanism
    3. Aligns events and phrases using Optimal Transport
    """
    
    def __init__(self, dim=1024, n_events=4, sinkhorn_reg=0.1):
        """
        Args:
            dim: Feature dimension
            n_events: Number of events/phrases to discover (K)
            sinkhorn_reg: Regularization parameter for Sinkhorn algorithm
        """
        super(RegionalAlignment, self).__init__()
        self.dim = dim
        self.n_events = n_events
        self.sinkhorn_reg = sinkhorn_reg
        
        # Learnable phrase queries for text (BLIP-2 style)
        self.phrase_queries = nn.Parameter(torch.randn(n_events, dim))
        nn.init.xavier_uniform_(self.phrase_queries)
    
    def discover_audio_events(self, audio_features):
        """
        Discover audio events using CutLER-inspired approach.
        
        Args:
            audio_features: (batch, T, dim) - audio patch tokens E_A
        
        Returns:
            event_tokens: (batch, K, dim) - discovered event tokens V_A
        """
        batch_size, T, dim = audio_features.shape
        
        # Compute temporal self-similarity matrix
        # Normalize features
        audio_norm = F.normalize(audio_features, p=2, dim=-1)
        # Self-similarity: (batch, T, T)
        similarity_matrix = torch.matmul(audio_norm, audio_norm.transpose(1, 2))
        
        # Apply Normalized Cuts clustering for each sample in batch
        event_tokens_list = []
        for b in range(batch_size):
            S_b = similarity_matrix[b].detach().cpu().numpy()
            
            # Normalized Cuts clustering
            try:
                labels = normalized_cuts_clustering(S_b, self.n_events)
            except:
                # Fallback: uniform partitioning
                labels = np.linspace(0, self.n_events - 1, T, dtype=int)
            
            # Aggregate features within each segment
            event_tokens_b = torch.zeros(self.n_events, dim, device=audio_features.device)
            for k in range(self.n_events):
                mask = labels == k
                if mask.sum() > 0:
                    event_tokens_b[k] = audio_features[b, mask].mean(dim=0)
                else:
                    # If no samples in cluster, use mean of all features
                    event_tokens_b[k] = audio_features[b].mean(dim=0)
            
            event_tokens_list.append(event_tokens_b)
        
        event_tokens = torch.stack(event_tokens_list)  # (batch, K, dim)
        return event_tokens
    
    def extract_text_phrases(self, text_features):
        """
        Extract text phrases using attention-based mechanism (BLIP-2 style).
        
        Args:
            text_features: (batch, N, dim) - text encoder output E_T
        
        Returns:
            phrase_tokens: (batch, K, dim) - extracted phrase tokens V_T
        """
        batch_size, N, dim = text_features.shape
        
        # Normalize text features
        text_norm = F.normalize(text_features, p=2, dim=-1)
        
        # Cross-attention: phrase queries attend to text features
        # Q: phrase_queries (K, dim)
        # K, V: text_features (batch, N, dim)
        phrase_queries_norm = F.normalize(self.phrase_queries, p=2, dim=-1)  # (K, dim)
        
        # Compute attention scores: (batch, K, N)
        attention_scores = torch.matmul(
            phrase_queries_norm.unsqueeze(0),  # (1, K, dim)
            text_norm.transpose(1, 2)  # (batch, dim, N)
        ) / (dim ** 0.5)  # (batch, K, N)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, K, N)
        
        # Weighted aggregation: (batch, K, dim)
        phrase_tokens = torch.matmul(attention_weights, text_features)  # (batch, K, dim)
        
        return phrase_tokens
    
    def compute_regional_similarity(self, event_tokens, phrase_tokens):
        """
        Compute regional alignment similarity using Optimal Transport.
        
        Args:
            event_tokens: (batch, K, dim) - audio event tokens V_A
            phrase_tokens: (batch, K, dim) - text phrase tokens V_T
        
        Returns:
            regional_similarities: (batch,) - regional alignment scores
        """
        batch_size = event_tokens.shape[0]
        device = event_tokens.device
        
        # Normalize tokens
        event_tokens_norm = F.normalize(event_tokens, p=2, dim=-1)
        phrase_tokens_norm = F.normalize(phrase_tokens, p=2, dim=-1)
        
        regional_similarities = []
        
        for b in range(batch_size):
            events_b = event_tokens_norm[b]  # (K, dim)
            phrases_b = phrase_tokens_norm[b]  # (K, dim)
            
            # Compute cost matrix: C_ij = 1 - sim(e_i, p_j)
            # Similarity: (K, K)
            similarity = torch.matmul(events_b, phrases_b.T)  # (K, K)
            cost_matrix = 1.0 - similarity  # (K, K)
            
            # Solve Optimal Transport using Sinkhorn-Knopp
            transport_plan = sinkhorn_knopp(cost_matrix, reg=self.sinkhorn_reg)  # (K, K)
            
            # Regional alignment similarity: -sum(Γ*_ij * C_ij)
            regional_sim = -(transport_plan * cost_matrix).sum()
            regional_similarities.append(regional_sim)
        
        return torch.stack(regional_similarities)  # (batch,)
    
    def forward(self, audio_features, text_features):
        """
        Forward pass: discover events, extract phrases, and compute regional similarity.
        
        Args:
            audio_features: (batch, T, dim) - audio patch tokens
            text_features: (batch, N, dim) - text encoder output
        
        Returns:
            regional_similarities: (batch,) - regional alignment scores
            event_tokens: (batch, K, dim) - discovered audio events
            phrase_tokens: (batch, K, dim) - extracted text phrases
        """
        # Discover audio events
        event_tokens = self.discover_audio_events(audio_features)
        
        # Extract text phrases
        phrase_tokens = self.extract_text_phrases(text_features)
        
        # Compute regional similarity
        regional_similarities = self.compute_regional_similarity(event_tokens, phrase_tokens)
        
        return regional_similarities, event_tokens, phrase_tokens


class GlobalAlignment(nn.Module):
    """
    Global Alignment module: Clip-level semantic consistency.
    
    Computes global similarity between entire audio clip and full caption.
    """
    
    def __init__(self, aggregation='mean'):
        """
        Args:
            aggregation: How to aggregate features ('mean' or 'cls')
        """
        super(GlobalAlignment, self).__init__()
        self.aggregation = aggregation
    
    def forward(self, audio_features, text_features):
        """
        Compute global alignment similarity.
        
        Args:
            audio_features: (batch, T, dim) - audio features
            text_features: (batch, N, dim) - text features
        
        Returns:
            global_similarities: (batch,) - global alignment scores
        """
        # Aggregate audio features
        if self.aggregation == 'mean':
            v_a = audio_features.mean(dim=1)  # (batch, dim)
        elif self.aggregation == 'cls':
            v_a = audio_features[:, 0, :]  # Use first token as CLS
        else:
            v_a = audio_features.mean(dim=1)
        
        # Aggregate text features
        if self.aggregation == 'mean':
            v_t = text_features.mean(dim=1)  # (batch, dim)
        elif self.aggregation == 'cls':
            v_t = text_features[:, 0, :]  # Use first token as CLS
        else:
            v_t = text_features.mean(dim=1)
        
        # Normalize
        v_a = F.normalize(v_a, p=2, dim=-1)
        v_t = F.normalize(v_t, p=2, dim=-1)
        
        # Cosine similarity: S_global(A, T) = cos(v_a, v_t)
        global_similarities = (v_a * v_t).sum(dim=1)  # (batch,)
        
        return global_similarities


class HierarchicalAlignment(nn.Module):
    """
    Unified Hierarchical Alignment framework combining three levels:
    1. Local Alignment (Token-Level)
    2. Regional Alignment (Event-Level)
    3. Global Alignment (Clip-Level)
    """
    
    def __init__(self, dim=1024, n_events=4, sinkhorn_reg=0.1, 
                 local_weight=1.0, regional_weight=0.5, global_weight=0.3,
                 local_tau=1.0):
        """
        Args:
            dim: Feature dimension
            n_events: Number of events for regional alignment
            sinkhorn_reg: Regularization for Sinkhorn algorithm
            local_weight: Weight for local alignment
            regional_weight: Weight for regional alignment
            global_weight: Weight for global alignment
            local_tau: Scaling factor for local alignment
        """
        super(HierarchicalAlignment, self).__init__()
        self.local_weight = local_weight
        self.regional_weight = regional_weight
        self.global_weight = global_weight
        self.local_tau = local_tau
        
        # Regional alignment module
        self.regional_align = RegionalAlignment(
            dim=dim,
            n_events=n_events,
            sinkhorn_reg=sinkhorn_reg
        )
        
        # Global alignment module
        self.global_align = GlobalAlignment(aggregation='mean')
    
    def compute_local_similarity(self, audio_features, text_features):
        """
        Compute Local Alignment (Token-Level) similarity using Max-Sum strategy.
        
        Args:
            audio_features: (batch, M, dim) - audio segments F_a
            text_features: (batch, N, dim) - text tokens F_t
        
        Returns:
            local_similarities: (batch,) - local alignment scores
        """
        batch_size, M, dim = audio_features.shape
        _, N, _ = text_features.shape
        device = audio_features.device
        
        # Normalize features
        audio_norm = F.normalize(audio_features, p=2, dim=-1)
        text_norm = F.normalize(text_features, p=2, dim=-1)
        
        local_similarities = []
        
        for b in range(batch_size):
            # Compute cosine similarity matrix: (M, N)
            similarity_matrix = torch.matmul(audio_norm[b], text_norm[b].T)  # (M, N)
            
            # Max-Sum: for each text token, find max over audio, then sum
            max_similarities = similarity_matrix.max(dim=0)[0]  # (N,) - max over audio for each text token
            local_sim = max_similarities.sum() * self.local_tau  # scalar
            
            local_similarities.append(local_sim)
        
        return torch.stack(local_similarities)  # (batch,)
    
    def forward(self, audio_features, text_features):
        """
        Compute hierarchical alignment similarities.
        
        Args:
            audio_features: (batch, M, dim) - audio features
            text_features: (batch, N, dim) - text features
        
        Returns:
            total_similarities: (batch,) - combined hierarchical similarity scores
            local_similarities: (batch,) - local alignment scores
            regional_similarities: (batch,) - regional alignment scores
            global_similarities: (batch,) - global alignment scores
        """
        # Local Alignment (Token-Level)
        local_similarities = self.compute_local_similarity(audio_features, text_features)
        
        # Regional Alignment (Event-Level)
        regional_similarities, event_tokens, phrase_tokens = self.regional_align(
            audio_features, text_features
        )
        
        # Global Alignment (Clip-Level)
        global_similarities = self.global_align(audio_features, text_features)
        
        # Combine three levels
        total_similarities = (
            self.local_weight * local_similarities +
            self.regional_weight * regional_similarities +
            self.global_weight * global_similarities
        )
        
        return {
            'total': total_similarities,
            'local': local_similarities,
            'regional': regional_similarities,
            'global': global_similarities,
            'event_tokens': event_tokens,
            'phrase_tokens': phrase_tokens
        }

