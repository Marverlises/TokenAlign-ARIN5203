import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=FutureWarning, module="hear21passt.models.preprocess")

import librosa
from collections import defaultdict
import os
from typing import Union, List, Mapping
import torch
import argparse
import pandas as pd
from tqdm import tqdm

from ATR.datasets.audio_loading import _pad_or_subsample_audio
from ATR.retrieval_module import AudioRetrievalModel


def predict(
        model: AudioRetrievalModel,
        audio_file_paths: List[str],
        queries: List[str]
) -> pd.DataFrame:
    """
    Tests the trained AudioRetrievalModel on a given test dataset.

    Args:
        model (d25_t6.retrieval_module.AudioRetrievalModel): The trained model to be evaluated.

    Returns:
        df: A pandas DataFrame containing the similarity scores between the queries (rows) and the audio files (columns).
    """

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # embed audios
    audio_embeddings = {}
    with torch.no_grad():
        for f in tqdm(audio_file_paths, desc="Embedding audio files..."):
            audio, sr = librosa.load(f, sr=32000)
            duration = [len(audio) / sr]
            audio = torch.tensor(audio).unsqueeze(0)
            audio = _pad_or_subsample_audio(audio, max_length=32000*30)
            audio = audio.cuda() if torch.cuda.is_available() else audio
            audio_embeddings[f] = model.forward_audio({
                'audio': audio.unsqueeze(0),
                'duration': duration
            })[0].detach().cpu()

    # embed queries
    query_embeddings = {}
    with torch.no_grad():
        for q in tqdm(queries, desc="Embedding queries..."):
            query_embeddings[q] = model.forward_text({
                'captions': [[q]]
            })[0].detach().cpu()

    # compute similarities
    similarities = defaultdict(list)
    for path, audio_embedding in audio_embeddings.items():
        key = os.path.basename(path)
        similarities[key] = [
            (query_embedding * audio_embedding).mean().item() for query_embedding in query_embeddings.values()
        ]
    similarities['index'] = list(query_embeddings.keys())

    # to pandas array
    df = pd.DataFrame.from_dict(similarities).set_index('index')

    return df

def get_args() -> dict:
    """
    Parses command-line arguments for configuring the training and testing process.

    Returns:
        dict: A dictionary containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Argument parser for prediction configuration.")

    # Parameter initialization & resume training
    parser.add_argument('--load_ckpt_path', type=str, default=None, required=True, help='Path to checkpoint.')

    parser.add_argument('--retrieval_audio_path', type=str, default='../data/predict/retrieval_audio', help='Path to items to be retrieved.')
    parser.add_argument('--retrieval_captions', type=str, default='../data/predict/retrieval_captions.csv', help='Path to CSV containing the queries.')

    parser.add_argument('--predictions_path', type=str, default='../resources', help='Path where predictions CSV will be stored.')

    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    """
    Entry point for creating predictions with a trained model.
    - Loads the checkpoint.
    - Runs inference and stores the result as a csv.
    """
    args = get_args()
    os.makedirs(args["predictions_path"], exist_ok=True)
    assert os.path.exists(args["retrieval_audio_path"]), "retrieval_audio_path must exist."
    assert os.path.exists(args["retrieval_captions"]), "retrieval_captions must exist."
    assert os.path.exists(args["predictions_path"]), "predictions_path must exist."
    assert os.path.exists(args["load_ckpt_path"]), "load_ckpt_path must exist."

    # initialize the model
    if args['load_ckpt_path']:
        model = AudioRetrievalModel.load_from_checkpoint(args['load_ckpt_path'])
    else:
        raise AttributeError("load_ckpt_path must be specified.")

    # get audio paths
    audio_files = [os.path.join(args["retrieval_audio_path"], f) for f in os.listdir(args["retrieval_audio_path"])]
    # get queries
    queries = pd.read_csv(args["retrieval_captions"])["caption"].tolist()
    # predict
    results = predict(model, audio_files, queries)
    # save CSV
    id = args["load_ckpt_path"].split(os.sep)[-2]
    results.to_csv(os.path.join(args["predictions_path"], f"{id}.csv"))
