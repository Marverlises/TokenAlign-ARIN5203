# Loss Functions for Intra-Modal Alignment
# Adapted from RICA-release/model/losses.py

import torch
from torch import nn as nn
from torch.nn import functional as F


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def dot_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    im = l2norm(im)
    s = l2norm(s)
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class Contrastive(nn.Module):
    def __init__(self, margin=0, measure=False, max_violation=False):
        super(Contrastive, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    def compute_contrastive_loss(self, scores):
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class AlignmentContrastiveLoss(Contrastive):
    """
    Compute contrastive loss for local feature alignment
    Adapted for audio-text retrieval
    """

    def __init__(self, margin=0, measure=False, max_violation=False, aggregation='sum-max-sentences',
                 return_similarity_mat=False):
        super(AlignmentContrastiveLoss, self).__init__(margin, measure, max_violation)
        self.aggregation = aggregation
        self.return_similarity_mat = return_similarity_mat

    def forward(self, audio_seq, text_seq, audio_len, text_len):
        """
        Args:
            audio_seq: (batch, audio_seq_len, dim) - audio local features
            text_seq: (batch, text_seq_len, dim) - text local features
            audio_len: list of actual audio sequence lengths (excluding padding)
            text_len: list of actual text sequence lengths (excluding padding)
        """
        # Remove CLS token from audio (assuming first token is CLS-like)
        # For audio, we typically don't have a CLS token, so we use all tokens
        # But we need to handle the length properly
        audio_seq = audio_seq[:, :, :]  # Use all audio tokens
        
        # For text, remove CLS token (first token) and handle EOS
        # text_seq shape: (batch, seq_len, dim)
        # We need to remove CLS and handle actual text length
        batch_size = text_seq.size(0)
        max_text_len = text_seq.size(1)
        
        # Create a new tensor for text without CLS token
        # Assuming CLS is at position 0, and we use positions 1:actual_len
        text_seq_processed = torch.zeros(
            batch_size, max_text_len - 1, text_seq.size(2), 
            device=text_seq.device
        )
        
        for i in range(batch_size):
            actual_len = text_len[i] - 1  # Exclude CLS token
            if actual_len > 0:
                # Take tokens from position 1 to actual_len (skip CLS)
                # Ensure we don't exceed the processed sequence length
                copy_len = min(actual_len, max_text_len - 1)
                text_seq_processed[i, :copy_len, :] = text_seq[i, 1:copy_len+1, :]
        
        text_seq = text_seq_processed
        
        # Adjust lengths (subtract 1 for CLS token removal)
        audio_len = [l for l in audio_len]  # Audio lengths remain the same
        text_len = [max(0, l - 1) for l in text_len]  # Text lengths minus CLS

        audio_batch = audio_seq.size(0)
        audio_seq_len = audio_seq.size(1)
        text_batch = text_seq.size(0)
        text_seq_len = text_seq.size(1)
        
        # Expand for pairwise comparison: (audio_batch, text_batch, audio_seq_len, dim)
        audio_seq_expanded = audio_seq.unsqueeze(1).expand(-1, text_batch, -1, -1)
        # Expand for pairwise comparison: (audio_batch, text_batch, text_seq_len, dim)
        text_seq_expanded = text_seq.unsqueeze(0).expand(audio_batch, -1, -1, -1)
        
        # Compute alignment matrix: (audio_batch, text_batch, audio_seq_len, text_seq_len)
        alignments = torch.matmul(audio_seq_expanded, text_seq_expanded.permute(0, 1, 3, 2))
        
        # Create masks for padding
        audio_len_mask = torch.zeros(audio_batch, audio_seq_len, device=audio_seq.device).bool()
        for i, l in enumerate(audio_len):
            if l < audio_seq_len:
                audio_len_mask[i, l:] = True
        audio_len_mask = audio_len_mask.unsqueeze(2).unsqueeze(1).expand(-1, text_batch, -1, text_seq_len)

        text_len_mask = torch.zeros(text_batch, text_seq_len, device=audio_seq.device).bool()
        for i, l in enumerate(text_len):
            if l < text_seq_len:
                text_len_mask[i, l:] = True
        text_len_mask = text_len_mask.unsqueeze(1).unsqueeze(0).expand(audio_batch, -1, audio_seq_len, -1)

        alignment_mask = audio_len_mask | text_len_mask
        alignments.masked_fill_(alignment_mask, value=0)
        
        # Aggregate: max over text dimension, then sum over audio dimension
        aggr_similarity = alignments.max(3)[0].sum(2) * 10  # Scale factor

        if self.return_similarity_mat:
            return aggr_similarity
        else:
            loss = self.compute_contrastive_loss(aggr_similarity)
            return loss


class ContrastiveLoss(Contrastive):
    """
    Compute contrastive loss for global feature matching
    """

    def __init__(self, margin=0, measure=False, max_violation=False, sigma=0.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sigma = sigma
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'dot':
            self.sim = dot_sim
        self.mse = torch.nn.MSELoss(reduce=True, size_average=False)

        self.max_violation = max_violation

    def forward(self, audio, text):
        """
        Args:
            audio: (batch, dim) - global audio features
            text: (batch, dim) - global text features
        """
        # compute audio-text score matrix
        scores = self.sim(audio, text)
        con_loss = self.compute_contrastive_loss(scores)
        
        # Intra-modal consistency: audio-audio and text-text similarities should be similar
        audio_scores = self.sim(audio, audio)
        text_scores = self.sim(text, text)

        # Intra-modal loss: encourage similar intra-modal similarity patterns
        intra_loss = ((text_scores - audio_scores).abs().max(0)[0] - self.sigma).clamp(min=0).sum()
        out_loss = con_loss + intra_loss
        return out_loss

