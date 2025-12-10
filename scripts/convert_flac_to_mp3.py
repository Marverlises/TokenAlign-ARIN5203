"""
Use this script to convert WavCaps from flac to mp3. This saves a lot of disk space, i.e, from 1.5TB to around 150GB.
"""

import os
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def convert_flac_to_mp3(source_file, target_file):
    """Convert a FLAC file to MP3 using ffmpeg."""
    if target_file.exists():
        return  # Skip conversion if target file already exists
    cmd = f"ffmpeg -hide_banner -nostats -loglevel error -n -i '{source_file}' -codec:a mp3 -ar 32000 -ac 1 '{target_file}'"
    subprocess.run(cmd, shell=True, check=True)


def process_file(source_path, target_root, source_root, progress):
    """Copy or convert files based on their type."""
    relative_path = source_path.relative_to(source_root)
    target_path = target_root / relative_path

    excluded_extensions = {".zip"} | {f".z{i}" for i in range(1, 200)} | {f".z{i:02d}" for i in range(1, 200)}

    if source_path.suffix.lower() == ".flac":
        target_path = target_path.with_suffix(".mp3")
        convert_flac_to_mp3(source_path, target_path)
    elif source_path.suffix.lower() not in excluded_extensions:
        if not target_path.exists():  # Skip if file already exists
            shutil.copy2(source_path, target_path)

    progress.update(1)


def copy_and_convert_folder(source_folder, target_folder):
    """Recursively copy a folder, converting FLAC files to MP3 in parallel."""
    source_folder = Path(source_folder).resolve()
    target_folder = Path(target_folder).resolve()

    for dirpath, _, filenames in os.walk(source_folder):
        rel_path = Path(dirpath).relative_to(source_folder)
        target_path = target_folder / rel_path
        if not target_path.exists():  # Skip if folder already exists
            target_path.mkdir(parents=True, exist_ok=True)

    files_to_process = []
    for dirpath, _, filenames in os.walk(source_folder):
        for filename in filenames:
            source_path = Path(dirpath) / filename
            target_path = target_folder / source_path.relative_to(source_folder)
            if source_path.suffix.lower() not in {".zip"} | {f".z{i}" for i in range(1, 100)} | {f".z{i:02d}" for i in
                                                                                                 range(1, 100)}:
                if not target_path.exists():  # Skip if file already exists
                    files_to_process.append(source_path)

    with tqdm(total=len(files_to_process), desc="Processing Files") as progress:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_file, file, target_folder, source_folder, progress) for file in
                       files_to_process]
            for future in futures:
                future.result()  # Ensure exceptions are raised


if __name__ == "__main__":
    source_dir = "data/WavCaps"  # Change this to your source folder
    target_dir = "data/WavCaps_mp3"  # Change this to your target folder
    copy_and_convert_folder(source_dir, target_dir)
