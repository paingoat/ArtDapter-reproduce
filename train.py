import argparse
import torch

torch.set_float32_matmul_precision('high')

import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, OnExceptionCheckpoint, TQDMProgressBar

from models import load_state_dict
from ldm.util import instantiate_from_config
from utils import freeze, set_warn_with_traceback, rank_zero_call


def get_args():
	parser = argparse.ArgumentParser(description='Art-dapter Training')
	parser.add_argument('--config_filepath',		'-cfg',	type=str,	default='./configs/train_config.yaml')
	parser.add_argument('--gpus',							'-g',		type=str,	default='0')
	parser.add_argument('--warning_traceback',	'-wt',	action='store_true')
	parser.add_argument('--resume_from',				'-r',		type=str,	default=None, help='Path to checkpoint to resume training from')
	parser.add_argument('--wandb_dir',					'-wd',	type=str,	default=None, help='Directory for wandb run files/media cache (default: ./wandb). Use /tmp/wandb to avoid filling SSD.')
	return parser.parse_args()


def main():
	args = get_args()
	gpus = [int(g.strip()) for g in args.gpus.split(',')]
	if args.warning_traceback:
		set_warn_with_traceback()

	config = OmegaConf.load(args.config_filepath)
	assert config.training.precision in ['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', '64', '32', '16', 'bf16']

	model = instantiate_from_config(config.model)
	model.load_state_dict(load_state_dict(config.model.init_path, location='cpu'))
	model.learning_rate = config.training.learning_rate
	model.weight_decay = config.training.weight_decay
	model.sd_locked = config.model.sd_locked

	if model.sd_locked:
		freeze(model.model)

	dataset = instantiate_from_config(config.dataset)
	dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, **dict(config.training.dataloader))

	wandb_logger = WandbLogger(project="ArtDapter", save_dir=args.wandb_dir or '.')
	wandb_logger.watch(model, log="all", log_freq=config.logger.params.log_frequency)
	rank_zero_call(wandb_logger.experiment.config, 'update', OmegaConf.to_container(config, resolve=True))

	# Gradient accumulation to compensate for smaller batch sizes
	accumulate_grad_batches = config.training.get('accumulate_grad_batches', 1)

	training = pl.Trainer(
		accelerator =				'auto',
		devices =						gpus,
		strategy =					config.training.strategy,
		max_steps =					config.training.training_steps,
		precision =					config.training.precision,
		logger =						wandb_logger,
		log_every_n_steps =	config.logger.params.log_frequency,
		accumulate_grad_batches = accumulate_grad_batches,
		callbacks =					[
			instantiate_from_config(config.logger),
			TQDMProgressBar(refresh_rate=50),
			ModelCheckpoint(
				dirpath =							config.training.ckpt_dir,
				filename =						str(wandb_logger.experiment.name) + '-{epoch}-{step}',
				every_n_train_steps =	config.logger.params.log_frequency,
				save_top_k =					config.training.get('save_top_k', -1),
				save_last =						config.training.get('save_last', False),
				monitor =							'train/loss',
				mode =								'min',
			),
			OnExceptionCheckpoint(config.training.ckpt_dir, f'{wandb_logger.experiment.name}_EXCEPTION')
		]
	)
	training.fit(model, dataloader, ckpt_path=args.resume_from)


if __name__ == '__main__':
	main()
