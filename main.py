import argparse
import os
import torch
import pytorch_lightning as pl
from typing import Optional

from src.models.padim_module import PaDiM
from src.models.patchcore_module import PatchCore
from src.data.dataset import get_train_loader, get_test_images
from src.utils.database import InferenceDatabase
from src.utils.file_manager import ResultsManager
from config.config import Config as C


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train anomaly detection models')
    
    parser.add_argument('--algorithm', '-a', 
                       choices=['padim', 'patchcore'], 
                       default='padim',
                       help='Algorithm to use (default: padim)')
    
    parser.add_argument('--dataset', '-d',
                       type=str,
                       default=None,
                       help='Path to dataset directory')
    
    parser.add_argument('--epochs', '-e',
                       type=int,
                       default=1,
                       help='Number of training epochs (default: 1)')
    
    parser.add_argument('--threshold', '-t',
                       type=float,
                       default=0.5,
                       help='Anomaly detection threshold (default: 0.5)')
    
    parser.add_argument('--output-dir', '-o',
                       type=str,
                       default='results',
                       help='Output directory for results (default: results)')
    
    parser.add_argument('--gpu', '-g',
                       action='store_true',
                       help='Use GPU if available')
    
    return parser.parse_args()


def create_model(algorithm: str) -> pl.LightningModule:
    """Create model based on algorithm choice"""
    if algorithm == 'padim':
        return PaDiM()
    elif algorithm == 'patchcore':
        return PatchCore()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def train_model(model: pl.LightningModule, 
                epochs: int, 
                dataset_path: Optional[str] = None,
                use_gpu: bool = False) -> pl.LightningModule:
    """Train the model"""
    # Clear GPU memory
    if use_gpu:
        torch.cuda.empty_cache()
    
    # Optimize tensor cores for better performance
    torch.set_float32_matmul_precision('medium')
    
    # Configure trainer
    accelerator = "gpu" if use_gpu and torch.cuda.is_available() else "cpu"
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Get data loader
    if dataset_path:
        # Update config with new dataset path
        C.DATASET_PATH = dataset_path
    
    train_loader = get_train_loader()
    
    # Train model
    trainer.fit(model, train_loader)
    
    return model


def save_results(model: pl.LightningModule, 
                algorithm: str,
                threshold: float,
                dataset_path: Optional[str],
                output_dir: str) -> None:
    """Save model and results"""
    # Initialize database and results manager
    db = InferenceDatabase()
    results_manager = ResultsManager(base_results_dir=output_dir)
    
    # Get model ID and create folder
    model_id = results_manager.get_next_model_id()
    output_folder, folder_name = results_manager.create_result_folder(model_id)
    model_path = os.path.join(output_folder, f"{algorithm}_model.pkl")
    
    # Save model
    results_manager.save_model(model, model_path, metadata={
        "model_type": algorithm.upper(),
        "threshold": threshold,
        "dataset_path": dataset_path or "default"
    })
    
    # Save model info to database
    db.save_model_info(
        model_name=f"{algorithm.upper()}_{model_id}",
        model_type=algorithm.upper(),
        model_path=model_path,
        performance_metrics={"threshold": threshold}
    )
    
    # Create model summary
    results_manager.create_model_summary(output_folder, {
        "model_type": algorithm.upper(),
        "model_id": model_id,
        "dataset": dataset_path or "MVTec (default)",
        "training_date": "Today",
        "threshold": threshold
    })
    
    print(f"Training completed successfully!")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Model ID: {model_id}")
    print(f"Results saved to: {output_folder}")
    print(f"Threshold: {threshold}")


def main():
    """Main function"""
    args = parse_args()
    
    print(f"🚀 Starting training with {args.algorithm.upper()}")
    print(f"📁 Dataset: {args.dataset or 'default'}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"🎯 Threshold: {args.threshold}")
    print(f"💾 Output: {args.output_dir}")
    print(f"⚡ GPU: {'enabled' if args.gpu else 'disabled'}")
    print("-" * 50)
    
    # Create model
    model = create_model(args.algorithm)
    
    # Train model
    trained_model = train_model(
        model=model,
        epochs=args.epochs,
        dataset_path=args.dataset,
        use_gpu=args.gpu
    )
    
    # Save results
    save_results(
        model=trained_model,
        algorithm=args.algorithm,
        threshold=args.threshold,
        dataset_path=args.dataset,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()