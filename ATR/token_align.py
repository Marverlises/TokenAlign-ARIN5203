"""
TokenAlign Module: Token-level alignment with learnable relative positional bias
Based on the first innovation point in the TokenAlign paper.

This module implements cross-attention with Gaussian relative positional bias
to enforce temporal consistency between text tokens and audio segments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenAlign(nn.Module):
    """
    Token-level alignment module with learnable relative positional bias.
    
    Implements cross-attention mechanism where:
    - Text tokens (Q) query audio segments (K, V)
    - Gaussian relative positional bias enforces temporal consistency
    - Formula: Attention(Q, K, V) = Softmax((QK^T)/√d - D_rel^2/(2σ^2)) V
    
    Args:
        dim (int): Feature dimension (default: 1024)
        initial_sigma (float): Initial value for learnable bandwidth σ (default: 1.0)
        sigma_trainable (bool): Whether σ is trainable (default: True)
    """
    
    def __init__(self, dim=1024, initial_sigma=1.0, sigma_trainable=True):
        super(TokenAlign, self).__init__()
        self.dim = dim
        self.sqrt_dim = dim ** 0.5
        
        # Learnable bandwidth parameter σ
        # Initialize with a positive value to ensure stable training
        initial_sigma_value = torch.tensor([initial_sigma], dtype=torch.float32)
        self.sigma = nn.Parameter(initial_sigma_value, requires_grad=sigma_trainable)
        
    def compute_relative_distance_matrix(self, text_len, audio_len, device):
        """
        Compute relative distance matrix D_rel between text tokens and audio segments.
        
        D_rel[i, j] = |i/N - j/M| where:
        - i is the normalized position of the i-th text token (i/N)
        - j is the normalized position of the j-th audio segment (j/M)
        - N is the text sequence length
        - M is the audio sequence length
        
        Args:
            text_len (int): Text sequence length N
            audio_len (int): Audio sequence length M
            device: Device to create tensor on
            
        Returns:
            torch.Tensor: Relative distance matrix of shape (text_len, audio_len)
        """
        # Normalized positions for text tokens: [0/N, 1/N, 2/N, ..., (N-1)/N]
        text_positions = torch.arange(text_len, dtype=torch.float32, device=device) / text_len
        
        # Normalized positions for audio segments: [0/M, 1/M, 2/M, ..., (M-1)/M]
        audio_positions = torch.arange(audio_len, dtype=torch.float32, device=device) / audio_len
        
        # Compute pairwise absolute differences
        # text_positions: (text_len, 1), audio_positions: (1, audio_len)
        text_positions = text_positions.unsqueeze(1)  # (text_len, 1)
        audio_positions = audio_positions.unsqueeze(0)  # (1, audio_len)
        
        # D_rel[i, j] = |i/N - j/M|
        D_rel = torch.abs(text_positions - audio_positions)  # (text_len, audio_len)
        
        return D_rel
    
    def forward(self, text_features, audio_features, text_lengths=None, audio_lengths=None):
        """
        Apply TokenAlign cross-attention with Gaussian relative positional bias.
        
        Args:
            text_features (torch.Tensor): Text token features F_t, shape (batch, N, dim)
            audio_features (torch.Tensor): Audio segment features F_a, shape (batch, M, dim)
            text_lengths (list, optional): Actual text sequence lengths (excluding padding)
            audio_lengths (list, optional): Actual audio sequence lengths (excluding padding)
        
        Returns:
            torch.Tensor: Enhanced text features F'_t, shape (batch, N, dim)
        """
        batch_size, text_len, dim = text_features.shape
        _, audio_len, _ = audio_features.shape
        
        device = text_features.device
        
        # Ensure sigma is positive (use abs or clamp)
        sigma = torch.abs(self.sigma) + 1e-8  # Add small epsilon for numerical stability
        
        # Compute attention scores: QK^T / √d
        # Q: text_features (batch, N, dim)
        # K: audio_features (batch, M, dim)
        attention_scores = torch.matmul(text_features, audio_features.transpose(1, 2))  # (batch, N, M)
        attention_scores = attention_scores / self.sqrt_dim
        
        # Compute relative distance matrix D_rel for each sample in batch
        # For efficiency, we compute one matrix and broadcast (assuming same lengths in batch)
        # In practice, lengths may vary, but we use max lengths for the bias matrix
        D_rel = self.compute_relative_distance_matrix(text_len, audio_len, device)  # (N, M)
        
        # Expand to batch dimension
        D_rel = D_rel.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, N, M)
        
        # Apply Gaussian relative positional bias: -D_rel^2 / (2 * σ^2)
        positional_bias = -(D_rel ** 2) / (2 * sigma ** 2)
        
        # Add positional bias to attention scores
        attention_scores = attention_scores + positional_bias
        
        # Apply attention mask if lengths are provided
        if text_lengths is not None or audio_lengths is not None:
            attention_mask = self._create_attention_mask(
                batch_size, text_len, audio_len, text_lengths, audio_lengths, device
            )
            # Mask out padding positions by setting to large negative value
            attention_scores = attention_scores.masked_fill(~attention_mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, N, M)
        
        # Apply attention to values: V = audio_features
        enhanced_text_features = torch.matmul(attention_weights, audio_features)  # (batch, N, dim)
        
        return enhanced_text_features, attention_weights
    
    def _create_attention_mask(self, batch_size, text_len, audio_len, text_lengths, audio_lengths, device):
        """
        Create attention mask to exclude padding positions.
        
        Args:
            batch_size (int): Batch size
            text_len (int): Maximum text sequence length
            audio_len (int): Maximum audio sequence length
            text_lengths (list): Actual text sequence lengths
            audio_lengths (list): Actual audio sequence lengths
            device: Device to create tensor on
            
        Returns:
            torch.Tensor: Attention mask of shape (batch, text_len, audio_len), True for valid positions
        """
        mask = torch.ones(batch_size, text_len, audio_len, dtype=torch.bool, device=device)
        
        if text_lengths is not None:
            for i, length in enumerate(text_lengths):
                if length < text_len:
                    mask[i, length:, :] = False
        
        if audio_lengths is not None:
            for i, length in enumerate(audio_lengths):
                if length < audio_len:
                    mask[i, :, length:] = False
        
        return mask

