import copy
import math
import string
from typing import Any
import os
import ast
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from lightning import pytorch as pl
from transformers import RobertaTokenizer, RobertaModel
from prettytable import PrettyTable

from ATR.passt import CutInputIntoSegmentsWrapper, PaSSTSNoOverlapWrapper
from ATR.losses import AlignmentContrastiveLoss, ContrastiveLoss, l2norm
from ATR.token_align import TokenAlign
from ATR.token_align_loss import (
    max_sum_token_similarity_efficient,
    online_hard_negative_mining,
    TokenAlignHybridLoss,
    TokenAlignHybridLossWithIntraModal
)
from ATR.hierarchical_alignment import HierarchicalAlignment


class AudioRetrievalModel(pl.LightningModule):

    def __init__(
            self,
            **kwargs
    ):

        super().__init__()
        self.save_hyperparameters(kwargs)

        # audio encoder
        self.audio_embedding_model = CutInputIntoSegmentsWrapper(
            PaSSTSNoOverlapWrapper(
                s_patchout_t=kwargs['s_patchout_t'],
                s_patchout_f=kwargs['s_patchout_f']
            ),
            max_input_length=10*32000,
            segment_length=10*32000,
            hop_size=10*32000
        )
        self.audio_projection = torch.nn.Linear(768, 1024)

        # text encoder
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.text_embedding_model = RobertaModel.from_pretrained(
            'roberta-base' if kwargs['roberta_base'] else 'roberta-large',
            add_pooling_layer=False,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            output_hidden_states=False
        )
        self.text_projection = torch.nn.Linear(768 if kwargs['roberta_base'] else 1024, 1024)

        # temperature parameter
        initial_tau = torch.zeros((1,)) + kwargs['initial_tau']
        self.tau = torch.nn.Parameter(initial_tau, requires_grad=kwargs['tau_trainable'])

        # TokenAlign parameters (First Innovation Point)
        self.enable_token_align = kwargs.get('enable_token_align', False)
        self.token_align_sigma_initial = kwargs.get('token_align_sigma_initial', 1.0)
        self.token_align_sigma_trainable = kwargs.get('token_align_sigma_trainable', True)
        
        # Initialize TokenAlign module if enabled
        if self.enable_token_align:
            self.token_align = TokenAlign(
                dim=1024,
                initial_sigma=self.token_align_sigma_initial,
                sigma_trainable=self.token_align_sigma_trainable
            )
        
        # TokenAlign Hybrid Loss parameters (Second Innovation Point)
        self.enable_token_align_loss = kwargs.get('enable_token_align_loss', False)
        self.token_align_margin = kwargs.get('token_align_margin', 0.2)
        self.token_align_lambda = kwargs.get('token_align_lambda', 0.1)
        self.token_align_sigma_margin = kwargs.get('token_align_sigma_margin', 0.0)
        
        # Initialize TokenAlign hybrid loss if enabled
        if self.enable_token_align_loss:
            self.token_align_loss = TokenAlignHybridLoss(
                margin=self.token_align_margin,
                lambda_consistency=self.token_align_lambda,
                sigma_margin=self.token_align_sigma_margin
            )
        
        # Hierarchical Alignment parameters
        self.enable_hierarchical_alignment = kwargs.get('enable_hierarchical_alignment', False)
        self.n_events = kwargs.get('n_events', 4)
        self.sinkhorn_reg = kwargs.get('sinkhorn_reg', 0.1)
        self.local_weight = kwargs.get('local_weight', 1.0)
        self.regional_weight = kwargs.get('regional_weight', 0.5)
        self.global_weight = kwargs.get('global_weight', 0.3)
        self.local_tau = kwargs.get('local_tau', 1.0)
        
        # Initialize Hierarchical Alignment if enabled
        if self.enable_hierarchical_alignment:
            self.hierarchical_align = HierarchicalAlignment(
                dim=1024,
                n_events=self.n_events,
                sinkhorn_reg=self.sinkhorn_reg,
                local_weight=self.local_weight,
                regional_weight=self.regional_weight,
                global_weight=self.global_weight,
                local_tau=self.local_tau
            )
        
        # Intra-Modal Alignment parameters
        self.enable_intra_modal_alignment = kwargs.get('enable_intra_modal_alignment', False)
        self.enable_matching_loss = kwargs.get('enable_matching_loss', False)
        self.enable_alignment_loss = kwargs.get('enable_alignment_loss', False)
        self.alignment_loss_weight = kwargs.get('alignment_loss_weight', 0.4)
        self.matching_loss_weight = kwargs.get('matching_loss_weight', 1.0)
        
        # Initialize loss functions if Intra-Modal Alignment is enabled
        if self.enable_intra_modal_alignment:
            delta = kwargs.get('delta', 0.2)
            measure = kwargs.get('measure', 'cosine')
            max_violation = kwargs.get('max_violation', True)
            aggregation = kwargs.get('aggregation', 'sum-max-sentences')
            sigma = kwargs.get('sigma', 0.0)
            
            if self.enable_alignment_loss:
                self.alignment_criterion = AlignmentContrastiveLoss(
                    margin=delta,
                    measure=measure,
                    max_violation=max_violation,
                    aggregation=aggregation
                )
            
            if self.enable_matching_loss:
                self.matching_criterion = ContrastiveLoss(
                    margin=delta,
                    measure=measure,
                    max_violation=max_violation,
                    sigma=sigma
                )

        self.validation_outputs = []

        self.kwargs = kwargs

        self.compile_model()

    def compile_model(self):
        """Apply torch.compile() if GPU is recent"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()  # Get current GPU device
            properties = torch.cuda.get_device_properties(device)
            if properties.major >= 7 and self.kwargs['compile'] == True:
                print("Compiling Models")
                self.text_embedding_model = torch.compile(self.text_embedding_model)
                self.audio_embedding_model.model.model = torch.compile(self.audio_embedding_model.model.model)

    def forward(self, batch) -> Any:
        """
        Forward pass. If TokenAlign is enabled, uses token-level features and cross-attention.
        Otherwise, uses global pooled features.
        """
        if self.enable_token_align:
            return self.forward_with_token_align(batch)
        else:
            # Original behavior: global features
            text_embeddings = self.forward_text(batch)
            audio_embeddings = self.forward_audio(batch)
            return audio_embeddings, text_embeddings
    
    def forward_with_token_align(self, batch, return_tokens=False):
        """
        Forward pass with TokenAlign: token-level alignment with temporal bias.
        
        Args:
            batch: Input batch
            return_tokens: If True, return token-level features and attention weights
        
        Returns:
            If return_tokens=False:
                audio_embeddings: (batch, dim) - aggregated audio features
                text_embeddings: (batch, dim) - aggregated text features
            If return_tokens=True:
                audio_embeddings: (batch, dim) - aggregated audio features
                text_embeddings: (batch, dim) - aggregated text features
                audio_token_features: (batch, M, dim) - audio token features
                enhanced_text_features: (batch, N, dim) - enhanced text token features
                attention_weights: (batch, N, M) - attention weights
        """
        # Get token-level features for both modalities
        audio_token_features, audio_lengths = self.forward_audio_tokens(batch)
        text_token_features, text_lengths = self.forward_text_tokens(batch)
        
        # Apply TokenAlign cross-attention (only if enabled)
        # Text tokens query audio segments with temporal bias
        if self.enable_token_align:
            enhanced_text_features, attention_weights = self.token_align(
                text_token_features, 
                audio_token_features,
                text_lengths=text_lengths,
                audio_lengths=audio_lengths
            )
        else:
            # If TokenAlign is not enabled, use original features
            enhanced_text_features = text_token_features
            attention_weights = None
        
        # Aggregate enhanced text features (can use mean pooling or other aggregation)
        # For now, we'll use mean pooling over the sequence dimension
        # In the paper, they use Max-Sum for similarity, but for global features we use mean
        text_embeddings = enhanced_text_features.mean(dim=1)  # (batch, dim)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # For audio, we can use the original aggregated features or mean pool
        # Since TokenAlign enhances text based on audio, we keep audio as is
        audio_embeddings = audio_token_features.mean(dim=1)  # (batch, dim)
        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
        
        if return_tokens:
            return audio_embeddings, text_embeddings, audio_token_features, enhanced_text_features, attention_weights
        else:
            return audio_embeddings, text_embeddings
    
    def forward_audio_with_local(self, batch):
        """
        Forward audio and return both global and local features
        Returns:
            global_features: (batch, dim) - global audio features
            local_features: (batch, seq_len, dim) - local audio features
            audio_lengths: list of actual sequence lengths
        """
        audio_embeddings = self.audio_embedding_model(batch['audio'].mean(1))  # (batch, seq_len, 768)
        
        # Get local features (before projection)
        local_features = audio_embeddings  # (batch, seq_len, 768)
        
        # Calculate actual lengths based on duration
        audio_lengths = []
        for i, duration in enumerate(batch['duration']):
            if duration <= 10:
                audio_lengths.append(1)
            elif duration <= 20:
                audio_lengths.append(min(2, audio_embeddings.shape[1]))
            else:
                audio_lengths.append(audio_embeddings.shape[1])
        
        # Aggregate for global features
        aggregated = []
        for i, duration in enumerate(batch['duration']):
            if duration <= 10:
                aggregated.append(audio_embeddings[i, 0])
            elif duration <= 20:
                aggregated.append(audio_embeddings[i, :2].mean(-2))
            else:
                aggregated.append(audio_embeddings[i].mean(-2))
        
        global_features = torch.stack(aggregated)
        global_features = self.audio_projection(global_features)  # (batch, 1024)
        global_features = torch.nn.functional.normalize(global_features, p=2, dim=-1)
        
        # Project local features
        batch_size, seq_len, dim = local_features.shape
        local_features = local_features.reshape(-1, dim)  # (batch*seq_len, 768)
        local_features = self.audio_projection(local_features)  # (batch*seq_len, 1024)
        local_features = local_features.reshape(batch_size, seq_len, -1)  # (batch, seq_len, 1024)
        local_features = torch.nn.functional.normalize(local_features, p=2, dim=2)
        
        return global_features, local_features, audio_lengths
    
    def forward_text_with_local(self, batch):
        """
        Forward text and return both global and local features
        Returns:
            global_features: (batch, dim) - global text features
            local_features: (batch, seq_len, dim) - local text features
            text_lengths: list of actual sequence lengths
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        captions = []
        for i, b in enumerate([c[0] for c in batch['captions']]):
            if not (type(b) == str):
                print(b)
                b = b[0]
            captions.append(b.lower().translate(str.maketrans('', '', string.punctuation)))

        tokenized = self.tokenizer(
            captions,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            max_length=32,
            truncation=True
        )

        token_embeddings = self.text_embedding_model(
            input_ids=tokenized['input_ids'].to(device),
            attention_mask=tokenized['attention_mask'].to(device)
        )[0]  # (batch, seq_len, dim)
        
        # Get local features (before projection)
        local_features = token_embeddings  # (batch, seq_len, 768 or 1024)
        
        # Calculate actual lengths
        text_lengths = tokenized['attention_mask'].sum(dim=1).cpu().tolist()
        
        # Global features (CLS token)
        global_features = token_embeddings[:, 0, :]  # (batch, dim)
        global_features = self.text_projection(global_features)  # (batch, 1024)
        global_features = torch.nn.functional.normalize(global_features, p=2, dim=-1)
        
        # Project local features
        batch_size, seq_len, dim = local_features.shape
        local_features = local_features.reshape(-1, dim)  # (batch*seq_len, dim)
        local_features = self.text_projection(local_features)  # (batch*seq_len, 1024)
        local_features = local_features.reshape(batch_size, seq_len, -1)  # (batch, seq_len, 1024)
        local_features = torch.nn.functional.normalize(local_features, p=2, dim=2)
        
        return global_features, local_features, text_lengths

    def forward_audio(self, batch):

        audio_embeddings = self.audio_embedding_model(batch['audio'].mean(1)) # forward

        # mask embeddings from padded empty audio parts
        aggregated = []
        for i, duration in enumerate(batch['duration']):
            if duration <= 10:
                aggregated.append(audio_embeddings[i, 0])
            elif duration <= 20:
                aggregated.append(audio_embeddings[i, :2].mean(-2))
            else:
                aggregated.append(audio_embeddings[i].mean(-2))

        audio_embeddings = torch.stack(aggregated)
        audio_embeddings = self.audio_projection(audio_embeddings) # project to same dimension
        audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=-1) # normalize
        return audio_embeddings
    
    def forward_audio_tokens(self, batch):
        """
        Forward audio and return all segment-level features (tokens) for TokenAlign.
        
        Returns:
            audio_token_features: (batch, M, dim) - all audio segment features
            audio_lengths: list of actual audio sequence lengths
        """
        audio_embeddings = self.audio_embedding_model(batch['audio'].mean(1))  # (batch, M, 768)
        
        # Calculate actual lengths based on duration
        audio_lengths = []
        for i, duration in enumerate(batch['duration']):
            if duration <= 10:
                audio_lengths.append(1)
            elif duration <= 20:
                audio_lengths.append(min(2, audio_embeddings.shape[1]))
            else:
                audio_lengths.append(audio_embeddings.shape[1])
        
        # Project all segments to 1024-dim
        batch_size, num_segments, dim = audio_embeddings.shape
        audio_embeddings = audio_embeddings.reshape(-1, dim)  # (batch*M, 768)
        audio_embeddings = self.audio_projection(audio_embeddings)  # (batch*M, 1024)
        audio_embeddings = audio_embeddings.reshape(batch_size, num_segments, -1)  # (batch, M, 1024)
        
        # L2 normalize each segment
        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
        
        return audio_embeddings, audio_lengths

    def forward_text(self, batch):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        captions = []
        for i, b in enumerate([c[0] for c in batch['captions']]):
            if not (type(b) == str):
                print(b)
                b = b[0]
            captions.append(b.lower().translate(str.maketrans('', '', string.punctuation)))

        tokenized = self.tokenizer(
            captions,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            max_length=32,
            truncation=True
        )

        token_embeddings = self.text_embedding_model(
            input_ids=tokenized['input_ids'].to(device),
            attention_mask=tokenized['attention_mask'].to(device)
        )[0]
        # select first token of sequence
        sentence_features = token_embeddings[:, 0, :]
        # project
        sentence_features = self.text_projection(sentence_features)
        # normalize
        sentence_features = torch.nn.functional.normalize(sentence_features, p=2, dim=-1)

        return sentence_features
    
    def forward_text_tokens(self, batch):
        """
        Forward text and return all token-level features for TokenAlign.
        
        Returns:
            text_token_features: (batch, N, dim) - all text token features
            text_lengths: list of actual text sequence lengths
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        captions = []
        for i, b in enumerate([c[0] for c in batch['captions']]):
            if not (type(b) == str):
                print(b)
                b = b[0]
            captions.append(b.lower().translate(str.maketrans('', '', string.punctuation)))

        tokenized = self.tokenizer(
            captions,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            max_length=32,
            truncation=True
        )

        token_embeddings = self.text_embedding_model(
            input_ids=tokenized['input_ids'].to(device),
            attention_mask=tokenized['attention_mask'].to(device)
        )[0]  # (batch, N, 1024 or 768)
        
        # Get actual text lengths (excluding padding)
        text_lengths = tokenized['attention_mask'].sum(dim=1).cpu().tolist()
        
        # Project all tokens to 1024-dim
        batch_size, seq_len, dim = token_embeddings.shape
        token_embeddings = token_embeddings.reshape(-1, dim)  # (batch*N, dim)
        token_embeddings = self.text_projection(token_embeddings)  # (batch*N, 1024)
        token_embeddings = token_embeddings.reshape(batch_size, seq_len, -1)  # (batch, N, 1024)
        
        # L2 normalize each token
        token_embeddings = F.normalize(token_embeddings, p=2, dim=-1)
        
        return token_embeddings, text_lengths

    def training_step(self, batch, batch_idx):

        self.lr_scheduler_step(batch_idx)

        # Get embeddings based on whether Hierarchical Alignment, TokenAlign, or Intra-Modal Alignment is enabled
        if self.enable_hierarchical_alignment:
            # Hierarchical Alignment: three-level alignment framework
            audio_token_features, audio_lengths = self.forward_audio_tokens(batch)
            text_token_features, text_lengths = self.forward_text_tokens(batch)
            
            # Apply TokenAlign if enabled (for temporal bias)
            if self.enable_token_align:
                enhanced_text_features, attention_weights = self.token_align(
                    text_token_features,
                    audio_token_features,
                    text_lengths=text_lengths,
                    audio_lengths=audio_lengths
                )
            else:
                enhanced_text_features = text_token_features
                attention_weights = None
            
            # Compute hierarchical alignment similarities
            hierarchical_results = self.hierarchical_align(
                audio_token_features,
                enhanced_text_features
            )
            
            # Use total similarity for loss computation
            total_similarities = hierarchical_results['total']  # (batch,)
            
            # Create similarity matrix for batch pairs
            batch_size = total_similarities.shape[0]
            paths = np.array([hash(batch['dataset'][i] + batch['subset'][i] + p) for i, p in enumerate(batch['fname'])])
            positive_mask = torch.tensor(paths[None, :] == paths[:, None], device=total_similarities.device)
            
            # Build similarity matrix from hierarchical similarities
            # For each pair (i, j), compute hierarchical similarity
            similarity_matrix = torch.zeros(batch_size, batch_size, device=total_similarities.device)
            for i in range(batch_size):
                for j in range(batch_size):
                    # Compute hierarchical similarity for pair (audio_i, text_j)
                    if i == j:
                        # Use precomputed total similarity for positive pairs
                        similarity_matrix[i, j] = total_similarities[i]
                    else:
                        # Compute on-the-fly for negative pairs
                        # This is simplified - in practice, we'd compute full hierarchical alignment
                        # For efficiency, we use local alignment only for negatives
                        local_sim = max_sum_token_similarity_efficient(
                            enhanced_text_features[j:j+1],
                            audio_token_features[i:i+1],
                            attention_weights[j:j+1] if attention_weights is not None else None,
                            tau=self.local_tau
                        )[0, 0]
                        similarity_matrix[i, j] = local_sim
            
            # Online hard negative mining
            hardest_audio_negatives, hardest_text_negatives = online_hard_negative_mining(
                similarity_matrix, positive_mask
            )
            
            # Compute hybrid loss
            if self.enable_token_align_loss:
                total_loss, triplet_loss, consistency_loss = self.token_align_loss(
                    similarity_matrix,
                    positive_mask,
                    hardest_audio_negatives,
                    hardest_text_negatives
                )
            else:
                # Fallback to standard contrastive loss
                C = similarity_matrix / torch.abs(self.tau)
                C_audio = torch.log_softmax(C, dim=0)
                C_text = torch.log_softmax(C, dim=1)
                main_loss = -0.5 * (C_audio[torch.where(positive_mask)].mean() + C_text[torch.where(positive_mask)].mean())
                total_loss = main_loss
                triplet_loss = main_loss
                consistency_loss = torch.tensor(0.0, device=similarity_matrix.device)
            
            # Log hierarchical alignment components
            self.log("train/hierarchical_local", hierarchical_results['local'].mean(), batch_size=batch_size, sync_dist=True)
            self.log("train/hierarchical_regional", hierarchical_results['regional'].mean(), batch_size=batch_size, sync_dist=True)
            self.log("train/hierarchical_global", hierarchical_results['global'].mean(), batch_size=batch_size, sync_dist=True)
            self.log("train/token_align_triplet_loss", triplet_loss, batch_size=batch_size, sync_dist=True)
            self.log("train/token_align_consistency_loss", consistency_loss, batch_size=batch_size, sync_dist=True)
            
            main_loss = triplet_loss
            audio_embeddings = audio_token_features.mean(dim=1)
            text_embeddings = enhanced_text_features.mean(dim=1)
            
        elif self.enable_token_align and self.enable_token_align_loss:
            # TokenAlign with hybrid loss: use Max-Sum similarity and hybrid objective
            audio_embeddings, text_embeddings, audio_token_features, enhanced_text_features, attention_weights = \
                self.forward_with_token_align(batch, return_tokens=True)
            
            # Compute Max-Sum token similarity matrix
            similarity_matrix = max_sum_token_similarity_efficient(
                enhanced_text_features, 
                audio_token_features, 
                attention_weights,
                tau=self.local_tau if hasattr(self, 'local_tau') else 1.0
            )
            
            # Create positive mask
            paths = np.array([hash(batch['dataset'][i] + batch['subset'][i] + p) for i, p in enumerate(batch['fname'])])
            positive_mask = torch.tensor(paths[None, :] == paths[:, None], device=similarity_matrix.device)
            
            # Online hard negative mining
            hardest_audio_negatives, hardest_text_negatives = online_hard_negative_mining(
                similarity_matrix, positive_mask
            )
            
            # Compute hybrid loss
            total_loss, triplet_loss, consistency_loss = self.token_align_loss(
                similarity_matrix,
                positive_mask,
                hardest_audio_negatives,
                hardest_text_negatives
            )
            
            main_loss = triplet_loss  # For logging
            
            # Log losses
            self.log("train/token_align_triplet_loss", triplet_loss, batch_size=len(audio_embeddings), sync_dist=True)
            self.log("train/token_align_consistency_loss", consistency_loss, batch_size=len(audio_embeddings), sync_dist=True)
            
        elif self.enable_token_align:
            # TokenAlign without hybrid loss: use standard contrastive loss
            audio_embeddings, text_embeddings = self.forward(batch)
            
            # compute pairwise similarities
            C = torch.matmul(audio_embeddings, text_embeddings.T)

            # scale cosine similarities with temperature < 1
            C = C / torch.abs(self.tau)

            # compute P(a|t) and P(t|a)
            C_audio = torch.log_softmax(C, dim=0)
            C_text = torch.log_softmax(C, dim=1)

            # prediction target
            paths = np.array([hash(batch['dataset'][i] + batch['subset'][i] + p) for i, p in enumerate(batch['fname'])])
            I = torch.tensor(paths[None, :] == paths[:, None])

            # Main contrastive loss
            main_loss = -0.5 * (C_audio[torch.where(I)].mean() + C_text[torch.where(I)].mean())
            total_loss = main_loss
            
        elif self.enable_intra_modal_alignment:
            # Get global and local features
            audio_global, audio_local, audio_lengths = self.forward_audio_with_local(batch)
            text_global, text_local, text_lengths = self.forward_text_with_local(batch)
            
            # Use global features for main contrastive loss
            audio_embeddings = audio_global
            text_embeddings = text_global
            
            # compute pairwise similarities
            C = torch.matmul(audio_embeddings, text_embeddings.T)
            C = C / torch.abs(self.tau)
            C_audio = torch.log_softmax(C, dim=0)
            C_text = torch.log_softmax(C, dim=1)
            
            # prediction target
            paths = np.array([hash(batch['dataset'][i] + batch['subset'][i] + p) for i, p in enumerate(batch['fname'])])
            I = torch.tensor(paths[None, :] == paths[:, None])
            
            # Main contrastive loss
            main_loss = -0.5 * (C_audio[torch.where(I)].mean() + C_text[torch.where(I)].mean())
            total_loss = main_loss
        else:
            # Original behavior: only global features
            audio_embeddings, text_embeddings = self.forward(batch)

            # compute pairwise similarities
            C = torch.matmul(audio_embeddings, text_embeddings.T)

            # scale cosine similarities with temperature < 1
            # (otherwise $-1 <= C_{ij} <= 1$)
            C = C / torch.abs(self.tau)

            # compute P(a|t) and P(t|a)
            C_audio = torch.log_softmax(C, dim=0)
            C_text = torch.log_softmax(C, dim=1)

            # prediction target
            paths = np.array([hash(batch['dataset'][i] + batch['subset'][i] + p) for i, p in enumerate(batch['fname'])])
            I = torch.tensor(paths[None, :] == paths[:, None])

            # Main contrastive loss
            main_loss = -0.5 * (C_audio[torch.where(I)].mean() + C_text[torch.where(I)].mean())
            total_loss = main_loss
        
        # Add Intra-Modal Alignment losses if enabled
        # Note: This only applies when enable_intra_modal_alignment is True and we're not using hierarchical alignment
        if self.enable_intra_modal_alignment and not self.enable_hierarchical_alignment:
            # Handle duplicate paths: average features for duplicate paths
            # paths should be defined from the intra_modal_alignment branch above
            unique_paths_dict = {}
            for i, p in enumerate(paths):
                if p not in unique_paths_dict:
                    unique_paths_dict[p] = []
                unique_paths_dict[p].append(i)
            
            # Get unique paths and their indices
            unique_paths_list = list(unique_paths_dict.keys())
            num_unique = len(unique_paths_list)
            
            # Initialize tensors for averaged features
            # audio_global and text_global are defined in the enable_intra_modal_alignment branch
            device = audio_global.device
            audio_global_unique = torch.zeros(num_unique, audio_global.size(1), device=device)
            text_global_unique = torch.zeros(num_unique, text_global.size(1), device=device)
            audio_local_unique = torch.zeros(num_unique, audio_local.size(1), audio_local.size(2), device=device)
            text_local_unique = torch.zeros(num_unique, text_local.size(1), text_local.size(2), device=device)
            audio_lengths_unique = []
            text_lengths_unique = []
            mask = torch.zeros(len(paths), dtype=torch.bool, device=device)
            
            # Average features for each unique path
            for unique_idx, path in enumerate(unique_paths_list):
                indices = unique_paths_dict[path]
                replacement_idx = indices[0]
                mask[replacement_idx] = True
                
                # Average global features
                audio_global_unique[unique_idx] = audio_global[indices].mean(0)
                text_global_unique[unique_idx] = text_global[indices].mean(0)
                
                # Average local features
                audio_local_unique[unique_idx] = audio_local[indices].mean(0)
                text_local_unique[unique_idx] = text_local[indices].mean(0)
                
                # Use average length (or max, depending on your preference)
                audio_lengths_unique.append(max([audio_lengths[i] for i in indices]))
                text_lengths_unique.append(max([text_lengths[i] for i in indices]))
            
            # Matching loss (global-to-global)
            if self.enable_matching_loss:
                matching_loss = self.matching_criterion(audio_global_unique, text_global_unique)
                total_loss = total_loss + matching_loss * self.matching_loss_weight
                self.log("train/matching_loss", matching_loss, batch_size=len(audio_embeddings), sync_dist=True)
            
            # Alignment loss (local-to-local)
            if self.enable_alignment_loss:
                alignment_loss = self.alignment_criterion(
                    audio_local_unique, 
                    text_local_unique, 
                    audio_lengths_unique, 
                    text_lengths_unique
                )
                total_loss = total_loss + alignment_loss * self.alignment_loss_weight
                self.log("train/alignment_loss", alignment_loss, batch_size=len(audio_embeddings), sync_dist=True)

        self.log("train/loss", total_loss, batch_size=len(audio_embeddings), sync_dist=True, prog_bar=True)
        self.log("train/main_loss", main_loss, batch_size=len(audio_embeddings), sync_dist=True)
        self.log('train/tau', torch.abs(self.tau), sync_dist=True)
        
        # Log TokenAlign sigma if enabled
        if self.enable_token_align:
            self.log('train/token_align_sigma', torch.abs(self.token_align.sigma), sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):

        audio_embeddings, text_embeddings = self.forward(batch)

        args = {
            'audio_embeddings': copy.deepcopy(audio_embeddings.detach()),
            'text_embeddings': copy.deepcopy(text_embeddings.detach()),
            'caption': [c[0] for c in batch['captions']],
            'path': batch['fname']
        }

        self.validation_outputs.append(args)

    def on_validation_epoch_end(self, prefix='val'):
        outputs = self.validation_outputs

        # concatenate metadata
        paths = np.array([p for b in outputs for p in b['path']])
        captions = np.array([p for b in outputs for p in b['caption']])

        # audios in clotho can have five captions
        # this snippet discards every occurrence of a duplicate audio
        #
        target = [] # prediction targets for later
        select = [] # indices of the first occurrence for later
        first_occurrence = {} # temporary cache to keep track of first occurrences
        for i, p in enumerate(paths): # iterate over all paths
            index = first_occurrence.get(p)
            if index is None:  # First time seeing this path
                index = len(first_occurrence)
                first_occurrence[p] = index
                select.append(i) # these audios will be selected
            target.append(index) # all paths need a target - choose the correct one
        paths = paths[select]

        # concatenate embeddings
        audio_embeddings = torch.cat([o['audio_embeddings'] for o in outputs])[select]# only select unique audios
        text_embeddings = torch.cat([o['text_embeddings'] for o in outputs])

        # concatenate global ranking
        C_text_to_audio = torch.matmul(text_embeddings, audio_embeddings.T)
        C_audio_to_text = C_text_to_audio.T  # Transpose for audio-to-text retrieval

        # ========== Text-to-Audio Retrieval Metrics ==========
        # get top 10 for text-to-audio
        top_ten_t2a = C_text_to_audio.topk(10, dim=1)[1].detach().cpu().numpy()
        target = np.array(target)

        # recall metrics for text-to-audio
        r_1_t2a = (top_ten_t2a[:, :1] == target[:, None]).sum(axis=1).mean()
        r_5_t2a = (top_ten_t2a[:, :5] == target[:, None]).sum(axis=1).mean()
        r_10_t2a = (top_ten_t2a == target[:, None]).sum(axis=1).mean()

        # mAP@10 for text-to-audio
        AP_t2a = 1 / ((top_ten_t2a == target[:, None]).argmax(axis=1) + 1)
        AP_t2a[~(top_ten_t2a == target[:, None]).any(axis=1)] = 0
        mAP_t2a = AP_t2a.mean()

        # log text-to-audio retrieval performance
        prefix = 'text-to-audio/' + prefix
        self.log(f'{prefix}/R@1', r_1_t2a)
        self.log(f'{prefix}/R@5', r_5_t2a)
        self.log(f'{prefix}/R@10', r_10_t2a)
        self.log(f'{prefix}/mAP@10', mAP_t2a)

        # ========== Audio-to-Text Retrieval Metrics ==========
        # For audio-to-text, we need to find which captions correspond to each audio
        # Create a mapping: audio_idx -> list of caption indices that match this audio
        audio_to_caption_indices = {}
        for caption_idx, audio_idx in enumerate(target):
            if audio_idx not in audio_to_caption_indices:
                audio_to_caption_indices[audio_idx] = []
            audio_to_caption_indices[audio_idx].append(caption_idx)

        # get top 10 for audio-to-text
        top_ten_a2t = C_audio_to_text.topk(10, dim=1)[1].detach().cpu().numpy()
        
        # Calculate recall metrics for audio-to-text
        # For each audio, check if any of its ground truth captions are in top-k
        r_1_a2t_list = []
        r_5_a2t_list = []
        r_10_a2t_list = []
        AP_a2t_list = []
        
        for audio_idx in range(len(paths)):
            true_caption_indices = set(audio_to_caption_indices.get(audio_idx, []))
            retrieved_indices = top_ten_a2t[audio_idx]
            
            # R@1
            r_1_a2t_list.append(1 if retrieved_indices[0] in true_caption_indices else 0)
            
            # R@5
            r_5_a2t_list.append(1 if any(idx in true_caption_indices for idx in retrieved_indices[:5]) else 0)
            
            # R@10
            r_10_a2t_list.append(1 if any(idx in true_caption_indices for idx in retrieved_indices[:10]) else 0)
            
            # mAP@10
            # Find the rank of the first relevant caption
            relevant_ranks = [rank + 1 for rank, idx in enumerate(retrieved_indices) if idx in true_caption_indices]
            if relevant_ranks:
                AP_a2t_list.append(1.0 / relevant_ranks[0])
            else:
                AP_a2t_list.append(0.0)
        
        r_1_a2t = np.mean(r_1_a2t_list)
        r_5_a2t = np.mean(r_5_a2t_list)
        r_10_a2t = np.mean(r_10_a2t_list)
        mAP_a2t = np.mean(AP_a2t_list)

        # log audio-to-text retrieval performance
        prefix = 'audio-to-text/' + prefix
        self.log(f'{prefix}_a2t/R@1', r_1_a2t)
        self.log(f'{prefix}_a2t/R@5', r_5_a2t)
        self.log(f'{prefix}_a2t/R@10', r_10_a2t)
        self.log(f'{prefix}_a2t/mAP@10', mAP_a2t)

        if os.path.exists(f'resources/metadata_eval.csv') and prefix == 'test':

            matched_files = pd.read_csv(f'resources/metadata_eval.csv')
            matched_files["audio_filenames"] = matched_files["audio_filenames"].transform(lambda x: ast.literal_eval(x))

            def get_ranks(c, r):
                ranks = [i.item() for i in torch.argsort(torch.argsort(-c))[r]]
                return ranks

            # Create mapping dictionaries for safe lookup
            captions_list = captions.tolist()
            captions_to_index = {cap: idx for idx, cap in enumerate(captions_list)}
            paths_list = paths.tolist()
            paths_to_index = {path: idx for idx, path in enumerate(paths_list)}

            # index of query in C - use safe lookup with fallback
            def safe_caption_index(x):
                if x in captions_to_index:
                    return captions_to_index[x]
                else:
                    # Try case-insensitive and stripped version
                    x_normalized = x.lower().strip() if isinstance(x, str) else str(x).lower().strip()
                    for cap, idx in captions_to_index.items():
                        if isinstance(cap, str) and cap.lower().strip() == x_normalized:
                            return idx
                    # If still not found, return None (will be filtered out)
                    print(f"Warning: Query '{x}' not found in captions. Skipping this entry.")
                    return None

            matched_files["query_index"] = matched_files["query"].transform(safe_caption_index)
            
            # Filter out rows where query_index is None
            matched_files = matched_files[matched_files["query_index"].notna()].copy()
            
            if len(matched_files) == 0:
                print("Warning: No valid queries found in metadata_eval.csv after matching. Skipping multiple positives mAP calculation.")
            else:
                matched_files["query_index"] = matched_files["query_index"].astype(int)

                # new ground truth - use safe lookup
                def safe_path_indices(x):
                    indices = []
                    for y in x:
                        if y in paths_to_index:
                            indices.append(paths_to_index[y])
                        else:
                            # Try to find similar path (case-insensitive)
                            y_normalized = y.lower().strip() if isinstance(y, str) else str(y).lower().strip()
                            found = False
                            for path, idx in paths_to_index.items():
                                if isinstance(path, str) and path.lower().strip() == y_normalized:
                                    indices.append(idx)
                                    found = True
                                    break
                            if not found:
                                print(f"Warning: Audio filename '{y}' not found in paths. Skipping this filename.")
                    return indices

                matched_files["new_audio_indices"] = matched_files["audio_filenames"].transform(safe_path_indices)
                
                # Filter out rows where new_audio_indices is empty
                matched_files = matched_files[matched_files["new_audio_indices"].apply(lambda x: len(x) > 0)].copy()
                
                if len(matched_files) > 0:
                    matched_files["TP_ranks"] = matched_files.apply(lambda row: get_ranks(C_text_to_audio[row["query_index"]], row["new_audio_indices"]), axis=1)

                    def average_precision_at_k(relevant_ranks, k=10):
                        relevant_ranks = sorted(relevant_ranks)
                        ap = 0.0
                        for i, rank in enumerate(relevant_ranks, start=1):
                            if rank >= k:
                                break
                            ap += i / (rank + 1) # precision at threshold
                        return ap / len(relevant_ranks)  # Normalize by total number of relevant items

                    new_mAP = matched_files["TP_ranks"].apply(lambda ranks: average_precision_at_k(ranks, 10)).mean()
                    self.log(f'{prefix}_multiple_positives/mAP@10', new_mAP)
        # empty cached batches from validation loop
        self.validation_outputs.clear()


    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end(prefix='test')

    def configure_optimizers(self):
        """
        Configure optimizer with different learning rates for encoders and projection layers.
        According to paper: encoders use 5e-5, projection layers and σ use 1e-4.
        Weight decay is set to 1e-4 as specified in the paper.
        """
        # Separate parameter groups for different learning rates
        encoder_params = []
        projection_params = []
        
        # Encoders: PaSST and RoBERTa
        encoder_params.extend(list(self.audio_embedding_model.parameters()))
        encoder_params.extend(list(self.text_embedding_model.parameters()))
        
        # Projection layers
        projection_params.extend(list(self.audio_projection.parameters()))
        projection_params.extend(list(self.text_projection.parameters()))
        
        # TokenAlign sigma (if enabled) - should use projection_lr
        if self.enable_token_align:
            projection_params.append(self.token_align.sigma)
        
        # Hierarchical Alignment parameters (if enabled) - should use projection_lr
        if self.enable_hierarchical_alignment:
            projection_params.extend(list(self.hierarchical_align.parameters()))
        
        # Temperature parameter tau
        projection_params.append(self.tau)
        
        # Get learning rates from kwargs (with fallback to old max_lr for backward compatibility)
        encoder_lr = self.kwargs.get('encoder_lr', self.kwargs.get('max_lr', 5e-5))
        projection_lr = self.kwargs.get('projection_lr', self.kwargs.get('max_lr', 1e-4))
        
        # Weight decay: 1e-4 as specified in paper
        weight_decay = self.kwargs.get('weight_decay', 1e-4)
        
        optimizer = torch.optim.AdamW(
            [
                {'params': encoder_params, 'lr': encoder_lr},
                {'params': projection_params, 'lr': projection_lr}
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
            amsgrad=False
        )

        return optimizer

    def lr_scheduler_step(self, batch_idx):
        """
        Learning rate scheduler with different rates for encoders and projection layers.
        Uses cosine decay scheduler as specified in the paper.
        """
        steps_per_epoch = self.trainer.num_training_batches

        # Get learning rates (with fallback for backward compatibility)
        encoder_max_lr = self.kwargs.get('encoder_lr', self.kwargs.get('max_lr', 5e-5))
        encoder_min_lr = self.kwargs.get('encoder_min_lr', self.kwargs.get('min_lr', 1e-7))
        projection_max_lr = self.kwargs.get('projection_lr', self.kwargs.get('max_lr', 1e-4))
        projection_min_lr = self.kwargs.get('projection_min_lr', self.kwargs.get('min_lr', 1e-7))
        
        current_step = self.current_epoch * steps_per_epoch + batch_idx
        warmup_steps = self.kwargs['warmup_epochs'] * steps_per_epoch
        total_steps = (self.kwargs['warmup_epochs'] + self.kwargs['rampdown_epochs']) * steps_per_epoch
        decay_steps = total_steps - warmup_steps

        if current_step < warmup_steps:
            # Linear warmup
            encoder_lr = encoder_min_lr + (encoder_max_lr - encoder_min_lr) * (current_step / warmup_steps)
            projection_lr = projection_min_lr + (projection_max_lr - projection_min_lr) * (current_step / warmup_steps)
        elif current_step < total_steps:
            # Cosine decay
            decay_progress = (current_step - warmup_steps) / decay_steps
            encoder_lr = encoder_min_lr + (encoder_max_lr - encoder_min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
            projection_lr = projection_min_lr + (projection_max_lr - projection_min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
        else:
            # Constant learning rate
            encoder_lr = encoder_min_lr
            projection_lr = projection_min_lr

        # Update learning rates for different parameter groups
        optimizer = self.optimizers(use_pl_optimizer=False)
        if len(optimizer.param_groups) >= 2:
            optimizer.param_groups[0]['lr'] = encoder_lr
            optimizer.param_groups[1]['lr'] = projection_lr
            self.log('train/encoder_lr', encoder_lr, sync_dist=True)
            self.log('train/projection_lr', projection_lr, sync_dist=True)
        else:
            # Fallback for old configuration (single learning rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = encoder_lr
            self.log('train/lr', encoder_lr, sync_dist=True)
