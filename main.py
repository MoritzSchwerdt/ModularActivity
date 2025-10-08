from config import *
from data.datasets import get_dataloaders
from data.augmentations import parallel_augment, contrast_transforms
from training.simclr_trainer import SimCLRTrainer
from jax import random
import logging

def train_simclr():
    logger.info("Preparing dataloaders...")
    train_loader, val_loader = get_dataloaders(DATASET_PATH, BATCH_SIZE, contrast_transforms)
    logger.info(f"Training dataset batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    logger.info("Initializing SimCLR trainer...")
    trainer = SimCLRTrainer(
        exmp_imgs=parallel_augment(random.PRNGKey(0), next(iter(train_loader))),
        hidden_dim=HIDDEN_DIM, lr=LR, temperature=TEMPERATURE,
        weight_decay=WEIGHT_DECAY
    )
    logger.info("Starting training loop...")
    trainer.train_model(train_loader, val_loader, num_epochs=NUM_EPOCHS)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("SimCLR")
    train_simclr()
