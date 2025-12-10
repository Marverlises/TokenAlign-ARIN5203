import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=FutureWarning, module="hear21passt.models.preprocess")

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['MASTER_PORT'] = '25777'
from typing import Union, List, Mapping
import torch
import wandb
import argparse
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from lightning.pytorch.strategies import DDPStrategy
import logging

from ATR.logger_config import setup_logger

from aac_datasets import Clotho, WavCaps, AudioCaps
from torch.utils.data import DataLoader
from ATR.datasets.download_datasets import download_clotho, download_audiocaps, download_wavcaps_mp3
from ATR.datasets.audio_loading import custom_loading
from ATR.datasets.utils import exclude_broken_files, exclude_forbidden_files, exclude_forbidden_and_long_files
from ATR.datasets.batch_collate import CustomCollate

from ATR.retrieval_module import AudioRetrievalModel


def train(
        model: AudioRetrievalModel,
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        logger: Union[None, WandbLogger],
        args: dict
):
    """
    Trains the AudioRetrievalModel using provided datasets, logger, and configuration arguments.

    Args:
        model (d25_t6.retrieval_module.AudioRetrievalModel): The model to be trained.
        train_ds (torch.utils.data.Dataset): The training dataset.
        val_ds (torch.utils.data.Dataset): The validation dataset.
        logger (Union[None, WandbLogger]): The logger for tracking training metrics.
        args (dict): A dictionary of configuration arguments for training.

    Returns:
        d25_t6.retrieval_module.AudioRetrievalModel: The trained model.
    """
    # get a unique experiment name for name of checkpoint
    if wandb.run is not None:
        experiment_name = wandb.run.name or wandb.run.id  # Use name if available, else use ID
    else:
        experiment_name = "experiment_" + wandb.util.generate_id()  # Random unique ID fallback

    # create path for the model checkpoints
    checkpoint_dir = os.path.join(args["checkpoints_path"], experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="{epoch}",
        save_top_k=1,
        monitor="text-to-audio/val/mAP@10",
        mode="max",
        save_last=True
    )

    # trainer
    # Configure strategy for DDP to handle unused parameters (e.g., tau when tau_trainable=False)
    # This is needed when tau_trainable=False, as tau is used but doesn't require gradients
    strategy = DDPStrategy(find_unused_parameters=True) if args['devices'] != 'cpu' else None

    trainer = pl.Trainer(
        devices=args['devices'],
        strategy=strategy,
        logger=logger if wandb.run else None,
        callbacks=[checkpoint_callback],
        max_epochs=args['max_epochs'],
        precision="16-mixed",
        accumulate_grad_batches=args.get('accumulate_grad_batches', 1),
        num_sanity_val_steps=0,
        fast_dev_run=False
    )

    ### train on training set; monitor performance on val
    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_ds, batch_size=args['batch_size'], num_workers=args['n_workers'], shuffle=True, drop_last=True,
            persistent_workers=True, collate_fn=CustomCollate()
        ),
        val_dataloaders=DataLoader(
            val_ds, batch_size=args['batch_size_eval'], num_workers=args['n_workers'], shuffle=False, drop_last=False,
            persistent_workers=True, collate_fn=CustomCollate()
        ),
        ckpt_path=args['resume_ckpt_path'] # should be none unless training is resumed
    )

    return model

def test(
        model: AudioRetrievalModel,
        test_ds: torch.utils.data.Dataset,
        logger: Union[None, WandbLogger],
        args: dict
) -> List[Mapping[str, float]]:
    """
    Tests the trained AudioRetrievalModel on a given test dataset.

    Args:
        model (d25_t6.retrieval_module.AudioRetrievalModel): The trained model to be evaluated.
        test_ds (torch.utils.data.Dataset): The test dataset.
        logger (Union[None, WandbLogger]): The logger for tracking test metrics.
        args (dict): A dictionary of configuration arguments for testing.

    Returns:
        dict: The result of the model evaluation on the test dataset.
    """
    # Configure strategy for DDP to handle unused parameters (e.g., tau when tau_trainable=False)
    # This is needed when tau_trainable=False, as tau is used but doesn't require gradients
    strategy = DDPStrategy(find_unused_parameters=True) if args['devices'] != 'cpu' else None

    trainer = pl.Trainer(
        devices=args['devices'],
        strategy=strategy,
        logger=logger if wandb.run else None,
        callbacks=None,
        max_epochs=args['max_epochs'],
        precision="16-mixed",
        num_sanity_val_steps=0,
        fast_dev_run=False
    )

    ### test on the eval set
    result = trainer.test(
        model,
        DataLoader(
            test_ds, batch_size=args['batch_size_eval'], num_workers=args['n_workers'], shuffle=False, drop_last=False,
            persistent_workers=True, collate_fn=CustomCollate()
        )
    )

    return result


def get_args() -> dict:
    """
    Parses command-line arguments for configuring the training and testing process.

    Returns:
        dict: A dictionary containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Argument parser for training configuration.")

    parser.add_argument('--devices', type=str, default='auto', help='Device selection (e.g., auto, cpu, cuda, etc.)')
    parser.add_argument('--n_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--compile', default=True, action=argparse.BooleanOptionalAction, help='Compile the model if GPU version >= 7.')
    parser.add_argument('--logging', default=True, action=argparse.BooleanOptionalAction, help='Log metrics in wandb or not.')
    parser.add_argument('--exp_name', type=str, default='', help='Directory to save logs.')
    # Parameter initialization & resume training
    parser.add_argument('--resume_ckpt_path', type=str, default=None, help='Path to checkpoint to resume training from.')
    parser.add_argument('--load_ckpt_path', type=str, default="", help='Path to checkpoint used as a weight initialization for training.')

    # Training parameters
    parser.add_argument('--seed', type=int, default=13, help='Random seed of experiment')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--batch_size_eval', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Number of batches to accumulate gradients before optimizer step')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Number of warmup epochs')
    parser.add_argument('--rampdown_epochs', type=int, default=15, help='Number of ramp-down epochs')
    parser.add_argument('--max_lr', type=float, default=2e-5, help='Maximum learning rate (deprecated, use encoder_lr/projection_lr instead)')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate (deprecated, use encoder_min_lr/projection_min_lr instead)')
    parser.add_argument('--encoder_lr', type=float, default=5e-5, help='Initial learning rate for encoders (PaSST and RoBERTa) (default: 5e-5)')
    parser.add_argument('--projection_lr', type=float, default=1e-4, help='Initial learning rate for projection layers and learnable σ (default: 1e-4)')
    parser.add_argument('--encoder_min_lr', type=float, default=1e-7, help='Minimum learning rate for encoders (default: 1e-7)')
    parser.add_argument('--projection_min_lr', type=float, default=1e-7, help='Minimum learning rate for projection layers (default: 1e-7)')
    parser.add_argument('--initial_tau', type=float, default=0.05, help='Initial tau value')
    parser.add_argument('--tau_trainable', default=False, action=argparse.BooleanOptionalAction, help='Temperature parameter is trainable or not.')

    # PaSST parameters
    parser.add_argument('--s_patchout_t', type=int, default=15, help='Temporal patchout size')
    parser.add_argument('--s_patchout_f', type=int, default=2, help='Frequency patchout size')

    # RoBERTa parameters
    parser.add_argument('--roberta_base', default=False, action=argparse.BooleanOptionalAction,  help='Use Roberta base or large.')

    # TokenAlign parameters (First Innovation Point)
    parser.add_argument('--enable_token_align', default=False, action=argparse.BooleanOptionalAction,
                       help='Enable TokenAlign: token-level alignment with learnable relative positional bias.')
    parser.add_argument('--token_align_sigma_initial', type=float, default=1.0,
                       help='Initial value for learnable bandwidth σ in TokenAlign (default: 1.0).')
    parser.add_argument('--token_align_sigma_trainable', default=True, action=argparse.BooleanOptionalAction,
                       help='Whether σ is trainable in TokenAlign (default: True).')
    
    # TokenAlign Hybrid Loss parameters (Second Innovation Point)
    parser.add_argument('--enable_token_align_loss', default=False, action=argparse.BooleanOptionalAction,
                       help='Enable TokenAlign hybrid loss: Max-Sum similarity + hardest-triplet + consistency loss.')
    parser.add_argument('--token_align_margin', type=float, default=0.2,
                       help='Margin δ for triplet loss in TokenAlign (default: 0.2).')
    parser.add_argument('--token_align_lambda', type=float, default=0.1,
                       help='Weight λ for consistency loss in TokenAlign (default: 0.1).')
    parser.add_argument('--token_align_sigma_margin', type=float, default=0.0,
                       help='Relaxation hyperparameter σ_margin for consistency loss (default: 0.0).')
    
    # Hierarchical Alignment parameters
    parser.add_argument('--enable_hierarchical_alignment', default=False, action=argparse.BooleanOptionalAction,
                       help='Enable Hierarchical Alignment: three-level alignment (Local, Regional, Global).')
    parser.add_argument('--n_events', type=int, default=4,
                       help='Number of events/phrases for regional alignment (default: 4).')
    parser.add_argument('--sinkhorn_reg', type=float, default=0.1,
                       help='Regularization parameter for Sinkhorn algorithm (default: 0.1).')
    parser.add_argument('--local_weight', type=float, default=1.0,
                       help='Weight for local alignment (default: 1.0).')
    parser.add_argument('--regional_weight', type=float, default=0.5,
                       help='Weight for regional alignment (default: 0.5).')
    parser.add_argument('--global_weight', type=float, default=0.3,
                       help='Weight for global alignment (default: 0.3).')
    parser.add_argument('--local_tau', type=float, default=1.0,
                       help='Scaling factor τ for local alignment (default: 1.0).')
    
    # Intra-Modal Alignment parameters
    parser.add_argument('--enable_intra_modal_alignment', default=False, action=argparse.BooleanOptionalAction,
                       help='Enable Intra-Modal Alignment (global-local alignment).')
    parser.add_argument('--enable_matching_loss', default=False, action=argparse.BooleanOptionalAction,
                       help='Enable matching loss (global-to-global contrastive loss).')
    parser.add_argument('--enable_alignment_loss', default=False, action=argparse.BooleanOptionalAction,
                       help='Enable alignment loss (local-to-local contrastive loss).')
    parser.add_argument('--alignment_loss_weight', type=float, default=0.4,
                       help='Weight for alignment loss (default: 0.4).')
    parser.add_argument('--matching_loss_weight', type=float, default=1.0,
                       help='Weight for matching loss (default: 1.0).')
    parser.add_argument('--delta', type=float, default=0.2,
                       help='Margin for contrastive losses (default: 0.2).')
    parser.add_argument('--measure', type=str, default='cosine',
                       choices=['cosine', 'dot', 'order'],
                       help='Similarity measure for contrastive losses (default: cosine).')
    parser.add_argument('--max_violation', default=True, action=argparse.BooleanOptionalAction,
                       help='Use max violation in contrastive losses.')
    parser.add_argument('--aggregation', type=str, default='sum-max-sentences',
                       help='Aggregation method for alignment loss (default: sum-max-sentences).')
    parser.add_argument('--sigma', type=float, default=0.0,
                       help='Sigma parameter for intra-modal consistency (default: 0.0).')

    # use additional data sets...
    parser.add_argument('--wavcaps', default=False, action=argparse.BooleanOptionalAction, help='Include WavCaps in the training or not.')
    parser.add_argument('--audiocaps', default=False, action=argparse.BooleanOptionalAction, help='Include AudioCaps in the training or not.')
    parser.add_argument('--clotho', default=False, action=argparse.BooleanOptionalAction, help='Include ClothoV2.1 eval, test in the training or not.')

    # Paths
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset; dataset will be downloaded into this folder.')
    parser.add_argument('--checkpoints_path', type=str, default='checkpoints', help='Path to save checkpoints to.')
    # Separate set dataset path
    parser.add_argument('--audiocaps_path', type=str, default='/share/project/baiyu/my_datasets/dcase2025/AudioCaps', help='Path to AudioCaps dataset if separate from data_path.')
    parser.add_argument('--wavcaps_path', type=str, default='/share/project/baiyu/my_datasets/dcase2025/WavCaps1', help='Path to WavCaps dataset if separate from data_path.')
    parser.add_argument('--clotho_path', type=str, default='/share/project/baiyu/my_datasets/dcase2025/Clotho', help='Path to Clotho dataset if separate from data_path.')


    # run training / test
    parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction, help='Run training or not.')
    parser.add_argument('--test', default=True, action=argparse.BooleanOptionalAction, help='Run testing or not.')

    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    """
    Entry point for training and testing the model.
    - Downloads datasets if necessary.
    - Initializes logging and model.
    - Runs training and/or testing based on arguments.
    """
    args = get_args()
    
    # 设置logger，在DDP环境中只在rank 0输出
    rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
    logger = setup_logger(rank=rank)
    
    # log used datasets
    logger.info("Using datasets:")
    if args['clotho']:
        logger.info("- ClothoV2.1")
    if args['audiocaps']:
        logger.info("- AudioCaps")
    if args['wavcaps']:
        logger.info("- WavCaps")
    os.makedirs(args["data_path"], exist_ok=True)
    # download data sets; will be ignored if exists
    # ClothoV2.1
    download_clotho(args["clotho_path"])
    # AudioCAps
    if args['audiocaps']:
        download_audiocaps(args["audiocaps_path"])
    # WavCaps
    if args['wavcaps']:
        download_wavcaps_mp3(args["wavcaps_path"])
        # download_wavcaps(args["data_path"], args["huggingface_cache_path"])

    # set a seed to make experiments reproducible
    if args['seed'] > 0:
        seed_everything(args['seed'], workers=True)
    else:
        logger.info("Not seeding experiment.")

    # initialize wandb, i.e., the logging framework
    if args['logging']:
        wandb.init(project="d25_t6", name=args['exp_name'])
        wandb_logger = WandbLogger()
    else:
        wandb_logger = None

    # initialize the model
    if args['load_ckpt_path']:
        logger.info(f"Loading model from checkpoint: {args['load_ckpt_path']}")
        if wandb_logger is not None:
            wandb_logger.log_text(f"Loading model from checkpoint: {args['load_ckpt_path']}")
        model = AudioRetrievalModel.load_from_checkpoint(args['load_ckpt_path'])
    else:
        model = AudioRetrievalModel(**args)

    # train
    if args['train']:
        # get training ad validation data sets; add the resampling transformation
        if args['clotho']:
            train_ds = custom_loading(Clotho(subset="dev", root=args["clotho_path"], flat_captions=True, download=False))

        if args['audiocaps']:
            train_ds = custom_loading(
                AudioCaps(subset="train", root=args["audiocaps_path"], download=False, download_audio=False, audio_format='mp3')
            )
            # train_ds = torch.utils.data.ConcatDataset([train_ds, ac])

        if args['wavcaps']:
            # load the subsets
            wc_f = exclude_forbidden_files(custom_loading(WavCaps(subset="freesound", root=args["wavcaps_path"])))
            wc_b = custom_loading(WavCaps(subset="bbc", root=args["wavcaps_path"]))
            wc_s = custom_loading(WavCaps(subset="soundbible", root=args["wavcaps_path"]))
            wc_a = exclude_broken_files(custom_loading(WavCaps(subset="audioset_no_audiocaps" if not args["clotho"] else "audioset", root=args["wavcaps_path"])))
            train_ds = torch.utils.data.ConcatDataset([train_ds, wc_f, wc_b, wc_s, wc_a])

        val_ds = custom_loading(Clotho(subset="val", root=args["clotho_path"], flat_captions=True))

        model = train(model, train_ds, val_ds, wandb_logger, args)

    # test
    if args['test']:
        if args['clotho']:
            # test on ClothoV2.1 eval set
            test_ds = custom_loading(Clotho(subset="eval", root=args["clotho_path"], flat_captions=True))
            results = test(model, test_ds, wandb_logger, args)
            logger.info("Test on ClothoV2.1 evaluation set:")
            logger.info(results)
        if args['audiocaps']:
            # test on AudioCaps eval set
            test_ds = custom_loading(
                AudioCaps(subset="test", root=args["audiocaps_path"], download=False, download_audio=False, audio_format='mp3'))
            results = test(model, test_ds, wandb_logger, args)
            logger.info("Test on AudioCaps evaluation set:")
            logger.info(results)

