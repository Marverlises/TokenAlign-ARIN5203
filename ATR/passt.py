import torch
import torch.nn as nn
from hear21passt.base import get_model_passt, AugmentMelSTFT

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="hearpasst")
warnings.filterwarnings("ignore", category=UserWarning, module="hearpasst")


class PaSSTSNoOverlapWrapper(torch.nn.Module):

    def __init__(self, s_patchout_t=15, s_patchout_f=2):
        super().__init__()
        """
        Args:
            s_patchout_t (int): Temporal patchout size.
            s_patchout_f (int): Frequency patchout size.
        """
        super().__init__()
        self.model = get_model_passt(
                "passt_s_p16_s16_128_ap468",
                input_tdim=1000,
                fstride=16, # larger stride means less compute
                tstride=16,
                s_patchout_t=s_patchout_t, # more dropout means less compute
                s_patchout_f=s_patchout_f
            )

        self.mel = AugmentMelSTFT(
            n_mels=128,
            sr=32000,
            win_length=800,
            hopsize=320,
            n_fft=1024,
            freqm=0,
            timem=0,
            htk=False,
            fmin=0.0,
            fmax=None,
            norm=1,
            fmin_aug_range=10,
            fmax_aug_range=2000
        )

    def forward(self, x):
        with torch.no_grad():
            mel = self.mel(x)

        tokens = self.model(mel[:, None])[-1] # get embedding, not token
        return tokens


class CutInputIntoSegmentsWrapper(nn.Module):
    def __init__(self, model, max_input_length, segment_length, hop_size):
        """
        Args:
            model (nn.Module): The PyTorch model to wrap.
            max_input_length (int): Maximum length of input the model can handle.
            segment_length (int): Length of each segment if input exceeds max_input_length.
            hop_size (int): Hop size for overlapping segmentation.
        """
        super().__init__()
        self.model = model
        self.max_input_length = max_input_length
        self.segment_length = segment_length
        self.hop_size = hop_size

    def forward(self, x):
        """Processes the input audio through the model, handling segmentation if needed."""
        batch_size, input_length = x.shape

        if input_length <= self.max_input_length:
            return self.model(x).unsqueeze(1)  # Add segment dimension

        # Split into overlapping segments
        segments = []
        indices = list(range(0, input_length - self.segment_length + 1, self.hop_size))
        for i in indices:
            segments.append(x[:, i:i + self.segment_length])

        segments = torch.stack(segments)  # Shape: (num_segments, batch_size, segment_length)
        outputs = self.model(segments.reshape(-1, self.segment_length))  # Process each segment
        outputs = outputs.view(len(indices), batch_size, -1).permute(1, 0, 2)   # Reshape back to (batch, num_segments , embedding_dim)

        # Return segments separately
        return outputs