import aac_datasets.datasets.base
import ffmpeg
import numpy as np
import os
import librosa
import torch
import logging

from torch import Tensor
from ATR.logger_config import get_logger

logger = get_logger()


def custom_loading(dataset: aac_datasets.datasets.base.AACDataset, normalize_audios=False) ->  aac_datasets.datasets.base.AACDataset:
    """
    Uses custom data loading to a dataset:
    - Loads a 30s snippet from a longer audio file efficiently.
    - Pads shorter audios to 30s automatically.
    - Resamples audios to 32kHz.

    Args:
        dataset (aac_datasets.datasets.base.AACDataset): The dataset to which transformations will be applied.
        normalize_audios (bool): Whether to normalize the audios to the range [-1, 1]. Default is False.

    Returns:
         aac_datasets.datasets.base.AACDataset: The transformed dataset with added online columns for audio and metadata.
    """
    dataset.add_online_column("audio", _custom_load_audio_mp3, True)
    dataset.add_online_column("audio_metadata", _custom_load_metadata, True)
    dataset.transform = custom_transform
    return dataset


def custom_transform(sample: dict):
    """
    Custom audio padding logic.
    """
    sample['duration'] = sample['audio'].shape[-1] / sample['sr']
    sample['audio'] = _pad_or_subsample_audio(sample['audio'], 32000 * 30)
    return sample

def _custom_load_audio(self, index: int) -> Tensor:
    """
    WARNING: THIS FUNCTION DOES NOT YIELD THE EXPECTED RESULTS – SOMETHING'S WRONG WITH LOADING .FLAC FILES.
    LET ME KNOW IF YOU FIND THE ISSUE...

    USE _custom_load_audio_mp3 INSTEAD!

    Custom audio loading logic.
    """
    fpath = self.at(index, "fpath")

    # load segment; truncate to 30 if longer than 30s
    audio, sr = _load_random_segment_ffmpeg(fpath, sample_rate=32000)  # type: ignore

    # Sanity check
    if audio.nelement() == 0:
        raise RuntimeError(
            f"Invalid audio number of elements in {fpath}. (expected audio.nelement()={audio.nelement()} > 0)"
        )

    return audio


def _custom_load_audio_mp3(self, index: int) -> Tensor:
    """
    Custom audio loading logic.
    """
    fpath = self.at(index, "fpath")
    base, extension = os.path.splitext(fpath)
    extension = extension[1:]
    # replace WavCaps default folder with WavCaps_mp3
    base =  base.replace("WavCaps", "WavCaps_mp3")
    if extension == "flac": # load mp3 version of WavCaps
        fpath = ".".join([base, 'mp3'])
        extension = 'mp3'
    # load segment; truncate to 30 if longer than 30s

    if extension == "mp3":
        # load audiocaps and wavcaps with ffmpeg
        audio, sr = _load_random_segment_ffmpeg(fpath, sample_rate=32000)  # type: ignore
    elif extension == "wav":
        # load clotho files with librosa, no subsampling required because all clotho files <= 30s
        audio, sr = librosa.load(fpath, sr=32000, mono=True)
        audio = torch.tensor(audio).unsqueeze(0)

    # Sanity check
    if audio.nelement() == 0:
        raise RuntimeError(
            f"Invalid audio number of elements in {fpath}. (expected audio.nelement()={audio.nelement()} > 0)"
        )
    return audio


def _normalize_waveform_tensor(waveform):
    """
    Normalize a waveform (PyTorch tensor) to the range [-1, 1] safely.

    Parameters:
    - waveform (torch.Tensor): Input waveform (integer or float tensor)

    Returns:
    - torch.Tensor: Normalized waveform in the range [-1, 1]
    """
    if not isinstance(waveform, torch.Tensor):
        raise ValueError("Input waveform must be a PyTorch tensor.")

    max_val = torch.max(torch.abs(waveform))

    if max_val == 0:
        return waveform  # Avoid division by zero, return unchanged (silent signal)

    return waveform / max_val


def _custom_load_metadata(self, index: int) -> dict:
    """
    Custom metadata loading logic.
    """
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    # just a placeholder...
    return dotdict(dict(
        duration = -1,
        sample_rate = int(32000),
        channels = int(1),
        num_frames = int(-1)
    ))


def _load_random_segment_ffmpeg(
        file_path: str,
        segment_duration: int = 30,
        sample_rate: int = 32000
) -> tuple[Tensor, int]:
    """
    Efficiently extracts a random 30-second segment from an audio file using ffmpeg without loading the full file.
    Falls back to librosa if ffmpeg/ffprobe is not available.

    :param file_path: Path to the audio file
    :param segment_duration: Segment duration in seconds (default: 30s)
    :param sample_rate: Sample rate for extracted audio (default: 32kHz)
    :return: PyTorch tensor of extracted audio and sample rate
    """
    try:
        # Get the total duration of the audio file using ffprobe
        probe = ffmpeg.probe(file_path)
        if 'format' not in probe or 'duration' not in probe['format']:
            raise ValueError(f"File appears to be corrupted or unreadable: {file_path}")

        duration = float(probe['format']['duration'])

        if duration < segment_duration:
            segment_duration = duration  # Adjust to full length
            start_time = 0  # Start from the beginning
        else:
            # avoid very small numbers to not run into ffmpeg issues
            start_time = max(0, round(torch.rand(1).item() * (duration - segment_duration), 3)) #

        # Use ffmpeg to extract only the required segment
        out, err = (
            ffmpeg.input(file_path, ss=start_time, t=segment_duration)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=sample_rate)
            .run(capture_stdout=True, capture_stderr=True)
        )

        audio = np.frombuffer(out, dtype=np.float32)
        # Convert to PyTorch tensor with shape (1, num_samples)
        return torch.tensor(audio).unsqueeze(0), sample_rate

    except FileNotFoundError as e:
        # FFmpeg/ffprobe not installed, use librosa fallback
        # Only log warning once to avoid spam
        if not hasattr(_load_random_segment_ffmpeg, '_ffmpeg_warning_printed'):
            logger.warning("ffmpeg/ffprobe not found. Falling back to librosa for audio loading.")
            logger.warning("This may be slower for large audio files. Consider installing ffmpeg for better performance.")
            _load_random_segment_ffmpeg._ffmpeg_warning_printed = True
        return _load_random_segment_librosa(file_path, segment_duration, sample_rate)
    
    except ffmpeg.Error as e:
        # FFmpeg error, try librosa as fallback
        logger.warning(f"FFmpeg error when processing file: {file_path}, falling back to librosa")
        if hasattr(e, 'stdout') and e.stdout:
            logger.debug(f"Standard Output: {e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            logger.debug(f"Standard Error: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}")
        return _load_random_segment_librosa(file_path, segment_duration, sample_rate)

    except Exception as e:
        # For any other error, try librosa as fallback
        logger.warning(f"Unexpected error when processing file: {file_path}, falling back to librosa")
        logger.debug(str(e))
        return _load_random_segment_librosa(file_path, segment_duration, sample_rate)


def _load_random_segment_librosa(
        file_path: str,
        segment_duration: int = 30,
        sample_rate: int = 32000
) -> tuple[Tensor, int]:
    """
    Fallback function to load a random segment using librosa when ffmpeg is not available.
    
    :param file_path: Path to the audio file
    :param segment_duration: Segment duration in seconds (default: 30s)
    :param sample_rate: Sample rate for extracted audio (default: 32kHz)
    :return: PyTorch tensor of extracted audio and sample rate
    """
    # Load the full audio file with librosa
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    duration = len(audio) / sr
    
    if duration < segment_duration:
        # Audio is shorter than segment_duration, pad with zeros
        target_length = int(segment_duration * sample_rate)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        start_idx = 0
    else:
        # Randomly select a segment
        max_start = int((duration - segment_duration) * sample_rate)
        start_idx = torch.randint(0, max_start + 1, (1,)).item()
        end_idx = start_idx + int(segment_duration * sample_rate)
        audio = audio[start_idx:end_idx]
    
    # Convert to PyTorch tensor with shape (1, num_samples)
    return torch.tensor(audio).unsqueeze(0), sample_rate



def _pad_or_subsample_audio(audio: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Adjusts the audio tensor to a fixed length by randomly selecting a snippet if too long,
    or padding with zeros if too short.

    Args:
        audio (torch.Tensor): Input audio tensor of shape (channels, audio_length)
        max_length (int): Desired maximum length of the audio snippet

    Returns:
        torch.Tensor: Processed audio tensor of shape (channels, max_length)
    """
    channels, audio_length = audio.shape

    if audio_length > max_length:
        # Randomly select a start index
        start_idx = torch.randint(0, audio_length - max_length + 1, (1,)).item()
        audio = audio[:, start_idx:start_idx + max_length]
    elif audio_length < max_length:
        # Pad with zeros to the right
        pad = torch.zeros((channels, max_length - audio_length), dtype=audio.dtype, device=audio.device)
        audio = torch.cat([audio, pad], dim=1)

    return audio
