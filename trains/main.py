from config import *
from data.datasets import get_dataloaders
from data.augmentations import parallel_augment, contrast_transforms
from training.simclr_positions_trainer import SimCLRPositionTrainer
from training.simclr_trainer import SimCLRTrainer
from jax import random
import jax.numpy as jnp
import logging

def train_simclr():
    logger.info("Preparing dataloaders...")
    train_loader, val_loader = get_dataloaders(DATASET_PATH, BATCH_SIZE, contrast_transforms)
    logger.info(f"Training dataset batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    logger.info("Initializing SimCLR trainer...")
    exmp_imgs = parallel_augment(random.PRNGKey(0), next(iter(train_loader)))

    lambdas_spatial = jnp.linspace(LAMBDA_SPATIAL, LAMBDA_SPATIAL/20, 150)
    loss_params = {"correlation_mode":"TDANN"}

    trainer = SimCLRPositionTrainer(
        model_name='SimCLR_scalar_TDANN',
        exmp_imgs=exmp_imgs, init_imgs=exmp_imgs,
        hidden_dim=HIDDEN_DIM, lr=LR, temperature=TEMPERATURE,
        weight_decay=WEIGHT_DECAY, loss_params=loss_params, aggr_mode="scalar", lambdas_spatial=lambdas_spatial
    )

    logger.info("Starting training loop...")
    trainer.train_model(train_loader, val_loader, num_epochs=100, save_every=5)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("SimCLR_scalar_TDANN")
    train_simclr()
