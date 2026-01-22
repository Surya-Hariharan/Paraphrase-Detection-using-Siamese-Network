"""Training script"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.models.siamese import SiameseModel
from backend.core.training.trainer import Trainer, ParaphraseDataset, load_dataset
from backend.config.settings import settings


def main():
    print("=" * 70)
    print("PARAPHRASE DETECTION - TRAINING")
    print("=" * 70)
    
    # Load dataset
    print(f"\nLoading dataset from {settings.training.DATA_PATH}")
    examples = load_dataset(settings.training.DATA_PATH)
    print(f"Loaded {len(examples):,} examples")
    
    # Create dataset
    train_dataset = ParaphraseDataset(examples)
    
    # Create model
    print("\nInitializing model...")
    model = SiameseModel(
        encoder_name=settings.model.ENCODER_NAME,
        embedding_dim=settings.model.EMBEDDING_DIM,
        projection_dim=settings.model.PROJECTION_DIM,
        freeze_encoder=False  # Fine-tune encoder
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        batch_size=settings.training.BATCH_SIZE,
        learning_rate=settings.training.LEARNING_RATE,
        num_epochs=settings.training.NUM_EPOCHS,
        checkpoint_dir=settings.training.CHECKPOINT_DIR,
        device=settings.model.DEVICE
    )
    
    # Train
    print("\n🚀 Starting training...\n")
    history = trainer.train(
        save_every=settings.training.SAVE_EVERY_N_EPOCHS,
        early_stopping_patience=settings.training.EARLY_STOPPING_PATIENCE,
        min_epochs=settings.training.MIN_EPOCHS
    )
    
    print("\n✅ Training complete!")
    print(f"Best loss: {min(history['train_loss']):.4f}")
    print(f"Best accuracy: {max(history['train_acc']):.2%}")


if __name__ == "__main__":
    main()
