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
    train_examples, val_examples, test_examples = load_dataset(
        settings.training.DATA_PATH,
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    # Create datasets
    train_dataset = ParaphraseDataset(train_examples)
    val_dataset = ParaphraseDataset(val_examples)
    test_dataset = ParaphraseDataset(test_examples)
    
    # Create model
    print("\nInitializing model...")
    model = SiameseModel(
        encoder_name=settings.model.ENCODER_NAME,
        embedding_dim=settings.model.EMBEDDING_DIM,
        projection_dim=settings.model.PROJECTION_DIM,
        freeze_encoder=False  # Fine-tune encoder
    )
    print(f"✓ Model created with {settings.model.ENCODER_NAME} encoder")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=settings.training.BATCH_SIZE,
        learning_rate=settings.training.LEARNING_RATE,
        num_epochs=settings.training.NUM_EPOCHS,
        device=settings.model.DEVICE
    )
    
    # Train
    print("\n🚀 Starting training...\n")
    history = trainer.train(
        checkpoint_dir=settings.training.CHECKPOINT_DIR,
        save_every=settings.training.SAVE_EVERY_N_EPOCHS,
        early_stopping_patience=settings.training.EARLY_STOPPING_PATIENCE,
        min_epochs=settings.training.MIN_EPOCHS
    )
    
    print("\n✅ Training complete!")
    if history['val_loss']:
        print(f"Best validation loss: {min(history['val_loss']):.4f}")
        print(f"Best validation accuracy: {max(history['val_acc']):.2%}")
    if history['test_acc']:
        print(f"Final test accuracy: {history['test_acc'][-1]:.2%}")


if __name__ == "__main__":
    main()
