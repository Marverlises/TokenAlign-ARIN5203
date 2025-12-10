import os
import requests
from tqdm import tqdm
import subprocess
import logging

import aac_datasets
from aac_datasets.datasets.functional.clotho import download_clotho_datasets
from aac_datasets.datasets.functional.wavcaps import download_wavcaps_datasets
from ATR.logger_config import get_logger

logger = get_logger()

def download_clotho(data_path: str):
    os.makedirs(data_path, exist_ok=True)
    if os.path.exists(data_path):
        logger.info("Clotho already exists. Skipping download.")
        return
    download_clotho_datasets(
        subsets=["dev", "val", "eval"],
        root=data_path,
        clean_archives=False,
        verbose=5
    )

def download_audiocaps(data_path: str):
    os.makedirs(data_path, exist_ok=True)
    URL = "https://cloud.cp.jku.at/index.php/s/9MiMcrNjJ3Z9FfH/download/AUDIOCAPS.zip"

    zip_file = os.path.join(data_path, 'AUDIOCAPS.zip')
    extract_to_dir = os.path.join(data_path, 'AUDIOCAPS')
    if os.path.exists(extract_to_dir):
        logger.info("AUDIOCAPS already exists. Skipping download and extraction.")
    else:
        download_zip_from_cloud(URL, zip_file)
        extract_zip(zip_file, extract_to_dir)


def download_wavcaps_mp3(data_path: str):
    os.makedirs(data_path, exist_ok=True)
    URLS = [
        "https://cloud.cp.jku.at/index.php/s/BxBp6r6asdsiWK8/download/WavCaps_mp3.7z.001",
        "https://cloud.cp.jku.at/index.php/s/dYMoW7D7nAx2LSE/download/WavCaps_mp3.7z.002",
        "https://cloud.cp.jku.at/index.php/s/6TREKcC4e7nzk2E/download/WavCaps_mp3.7z.003",
        "https://cloud.cp.jku.at/index.php/s/pg47anEan8kPaTD/download/WavCaps_mp3.7z.004",
        "https://cloud.cp.jku.at/index.php/s/4DSTDiC2kWPSgc7/download/WavCaps_mp3.7z.005",
        "https://cloud.cp.jku.at/index.php/s/XZ3s7mkdyn9pZ2K/download/WavCaps_mp3.7z.006",
        "https://cloud.cp.jku.at/index.php/s/fKwb8c5FMS3KYDa/download/WavCaps_mp3.7z.007",
        "https://cloud.cp.jku.at/index.php/s/ZW4LgFfxNJS3PGb/download/WavCaps_mp3.7z.008",
        "https://cloud.cp.jku.at/index.php/s/pG2i2fqJBr96EW9/download/WavCaps_mp3.7z.009",
        "https://cloud.cp.jku.at/index.php/s/NfN977dpoirFr9D/download/WavCaps_mp3.7z.010",
        "https://cloud.cp.jku.at/index.php/s/qWoqYiHzeDa7xSw/download/WavCaps_mp3.7z.011",
        "https://cloud.cp.jku.at/index.php/s/8EPsKozRn9WekG4/download/WavCaps_mp3.7z.012",
        "https://cloud.cp.jku.at/index.php/s/8Mri5yQmLPfKKSt/download/WavCaps_mp3.7z.013",
        "https://cloud.cp.jku.at/index.php/s/qc6FRwTSTXw6FX7/download/WavCaps_mp3.7z.014",
        "https://cloud.cp.jku.at/index.php/s/mTG3XMirG9EzMEb/download/WavCaps_mp3.7z.015"
    ]


    extract_to_dir = data_path
    if os.path.exists(os.path.join(extract_to_dir, "WavCaps_mp3")):
        logger.info("WavCaps_mp3 already exists. Skipping download and extraction.")
    else:
        for i in range(len(URLS)):
            zip_file = os.path.join(data_path, f"WavCaps_mp3.7z.{i + 1:03d}")
            download_zip_from_cloud(URLS[i], zip_file)
        extract_zip(os.path.join(data_path, "WavCaps_mp3.7z.001"), extract_to_dir)


def download_wavcaps(data_path: str, huggingface_cache_path: str):
    os.makedirs(data_path, exist_ok=True)
    # dirty fix to bypass a directory does not exist error...
    f_ = aac_datasets.datasets.functional.wavcaps._is_prepared_wavcaps
    def f(*args, **kwargs): return False
    aac_datasets.datasets.functional.wavcaps._is_prepared_wavcaps = f
    download_wavcaps_datasets(
        subsets=["bbc", "soundbible", "freesound_no_clotho_v2", "freesound_no_clotho", "audioset_no_audiocaps"],
        root=data_path,
        verbose=8,
        hf_cache_dir=huggingface_cache_path
    )
    aac_datasets.datasets.functional.wavcaps._is_prepared_wavcaps = f_


def download_zip_from_cloud(url: str, zip_file: str):

    if os.path.exists(zip_file):
        logger.info(f"{zip_file} already exists. Skipping download. {url}")
        return

    response = requests.get(
        url,
        stream=True
    )

    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))  # Get file size in bytes
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {url}")

        with open(zip_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
    else:
        raise Exception(f"Failed to download {url}.")


def extract_zip(zip_file: str, extract_to_dir: str):
    subprocess.run(["7z", "x", zip_file, f"-o {extract_to_dir}"], check=True)



if __name__ == '__main__':
    data_path = r'~/my_datasets/dcase2025'
    # huggingface_cache_path = r'/share/project/baiyu/my_datasets/dcase2025/huggingface_cache'
    # logger.info("Downloading  Clotho...")
    download_clotho(os.path.join(data_path, "Clotho"))
    # download_audiocaps(os.path.join(data_path, "AudioCaps"))
    # download_wavcaps_mp3(os.path.join(data_path, "WavCaps1"))
    # download_wavcaps(os.path.join(data_path, "WavCaps2"), huggingface_cache_path)
    logger.info("All datasets downloaded.")
