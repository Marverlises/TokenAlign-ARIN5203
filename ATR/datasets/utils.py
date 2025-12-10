import concurrent.futures
import json
import os
from typing import List, Optional
import pandas as pd
import ffmpeg
import torch
from tqdm import tqdm
import logging

from ATR.logger_config import get_logger

logger = get_logger()


def get_broken_wavcaps_files(
        datasets: List[torch.utils.data.Dataset],
        force_refresh: bool = False,
        broken_files_file: str = "broken_flacs.json"
) -> List[str]:
    """
    Checks for broken audio files in multiple datasets with a progress bar.

    Args:
        datasets (list): List of datasets, each containing a raw_data dictionary with 'fpath' key.
        force_refresh (bool): If True, re-checks all files instead of loading cached results.
        broken_files_file (str): Name of file to store the list of broken files.
    Returns:
        list: A merged list of broken audio files.
    """
    # Check if cached results exist
    if not force_refresh and os.path.exists(broken_files_file):
        with open(broken_files_file, "r") as f:
            broken_flacs = json.load(f)
        logger.info(f"Loaded {len(broken_flacs)} broken files from cache.")
        return broken_flacs

    # Collect all file paths from multiple datasets
    all_file_paths = []
    for dataset in datasets:
        all_file_paths.extend(dataset.raw_data['fpath'])

    logger.info(f"Checking {len(all_file_paths)} audio files...")

    broken_flacs = []

    # Use multiprocessing for parallel execution with tqdm progress bar
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(check_audio_file, all_file_paths)
        for result in tqdm(results, total=len(all_file_paths), desc="Checking files", unit="file"):
            if result is not None:
                broken_flacs.append(result)

    # Store the list of broken files on disk
    with open(broken_files_file, "w") as f:
        json.dump(broken_flacs, f, indent=4)

    logger.info(f"Found {len(broken_flacs)} broken files. List saved to {broken_files_file}.")
    return broken_flacs


def check_audio_file(fpath: str) -> Optional[str]:
    """Checks if an audio file is valid by probing and decoding a small segment."""
    try:
        # Probe metadata
        probe = ffmpeg.probe(fpath)
        if 'format' not in probe or 'duration' not in probe['format']:
            return fpath  # Return broken file path

        # Try decoding the first second of audio
        (
            ffmpeg
            .input(fpath, t=1)  # Load only the first second
            .output("pipe:", format="null")  # Discard output, just check if it processes
            .run(quiet=True)
        )
    except ffmpeg.Error:
        return fpath  # Return broken file path if decoding fails

    return None  # Valid file


def exclude_broken_files(dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
    """
    Excludes known broken files from the AudioSet subset of WavCaps.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to filter.

    Returns:
        torch.utils.data.Subset: A subset of the dataset excluding the broken files.
    """
    # broken files; see utils.get_broken_wavcaps_files for details...
    BROKEN_FILES =[
        "YVcu0pVF1npM",
        "Y06-g5jz-OGc",
        "Y3sSblRfEG2o",
        "YWudGD6ZHRoY",
        "YFli8wjBFV2M",
        "YmW3S0u8bj58"
    ]

    indices = []
    for i in range(len(dataset)):
        if dataset[i, 'fname'].split(os.sep)[-1].split('.')[0] not in BROKEN_FILES:
            indices.append(i)
        else:
            logger.debug(f"Excluding: {dataset[i, 'fname']}")

    return torch.utils.data.Subset(dataset, indices)


def exclude_forbidden_files(dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
    """
    Excludes known overlap with ClothoV2 evaluation set from WavCaps.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to filter.

    Returns:
        torch.utils.data.Subset: A subset of the dataset excluding the broken files.
    """
    # broken files; see utils.get_broken_wavcaps_files for details...
    forbidden_files = set(pd.read_csv("resources/dcase2025_task6_excluded_freesound_ids.csv")["sound_id"].to_list())

    indices = []
    exclude = 0
    for i in range(len(dataset)):
        if dataset[i, 'fname'].split(os.sep)[-1].split('.')[0] not in forbidden_files:
            indices.append(i)
        else:
            exclude += 1

    logger.info(f"Excluding {exclude} files.")

    return torch.utils.data.Subset(dataset, indices)


def exclude_forbidden_and_long_files(dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
    """
    Excludes known overlap with ClothoV2 evaluation set from WavCaps.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to filter.

    Returns:
        torch.utils.data.Subset: A subset of the dataset excluding the broken files.
    """
    # broken files; see utils.get_broken_wavcaps_files for details...
    forbidden_files = set(pd.read_csv("resources/dcase2025_task6_excluded_freesound_ids.csv")["sound_id"].to_list())

    with open('data/WavCaps/json_files/FreeSound/fsd_final.json', 'r') as f:
        import json
        a = json.load(f)['data']
        long_files = set([d['id'] for d in a if d['duration'] > 120])
    forbidden_files = forbidden_files.union(long_files)

    indices = []
    exclude = 0
    for i in range(len(dataset)):
        if dataset[i, 'fname'].split(os.sep)[-1].split('.')[0] not in forbidden_files:
            indices.append(i)
        else:
            exclude += 1

    logger.info(f"Excluding {exclude} files.")

    return torch.utils.data.Subset(dataset, indices)
