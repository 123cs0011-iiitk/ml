#!/usr/bin/env python3
"""
Full Dataset Training Script

This script trains all 9 ML models on the full dataset of ~1,000 stocks
with 5 years of historical data, providing real-time progress updates
and dynamic pacing.

Usage:
    python backend/training/train_full_dataset.py
    python backend/training/train_full_dataset.py --model linear_regression
    python backend/training/train_full_dataset.py --force-retrain
    python status.py
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import json
import time

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from training.enhanced_model_trainer import EnhancedModelTrainer
from prediction.config import config

# Setup logging
def setup_logging():
    """Setup comprehensive logging configuration."""
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'full_dataset_training_{timestamp}.log')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def main():
    """Main function to train all models on full dataset."""
    # Change to project root directory for correct relative paths
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    os.chdir(project_root)
    
    parser = argparse.ArgumentParser(description='Train all ML models on full dataset')
    
    parser.add_argument('--model', 
                       choices=['linear_regression', 'random_forest', 'svm', 'knn', 
                               'decision_tree', 'ann', 'cnn', 'arima', 'autoencoder'],
                       help='Train only a specific model')
    
    parser.add_argument('--force-retrain', 
                       action='store_true',
                       help='Force retrain even if model is completed')
    
    
    parser.add_argument('--validate-only', 
                       action='store_true',
                       help='Only run validation on completed models')
    
    parser.add_argument('--cleanup', 
                       action='store_true',
                       help='Clean up temporary files and exit')
    
    # Batch training arguments
    parser.add_argument('--batch-training', 
                       action='store_true',
                       help='Enable batch training mode')
    parser.add_argument('--stock-batch-size', 
                       type=int, 
                       default=100,
                       help='Number of stocks per batch (default: 100)')
    parser.add_argument('--row-batch-size', 
                       type=int, 
                       default=50000,
                       help='Number of rows per mini-batch (default: 50000)')
    parser.add_argument('--subsample-percent', 
                       type=int, 
                       default=50,
                       help='Subsampling percentage for non-incremental models (default: 50)')
    parser.add_argument('--disable-batch-training', 
                       action='store_true',
                       help='Disable batch training (use original single-pass method)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Initialize enhanced trainer
        logger.info("Initializing Enhanced Model Trainer...")
        logger.info("Target: ~1,000 stocks with 5 years of historical data")
        trainer = EnhancedModelTrainer()
        
        # Apply CLI overrides to config
        if args.batch_training:
            config.USE_BATCH_TRAINING = True
            logger.info("Batch training enabled via CLI")
        elif args.disable_batch_training:
            config.USE_BATCH_TRAINING = False
            logger.info("Batch training disabled via CLI")
        
        if args.stock_batch_size:
            config.STOCK_BATCH_SIZE = args.stock_batch_size
            logger.info(f"Stock batch size set to {args.stock_batch_size} via CLI")
        
        if args.row_batch_size:
            config.ROW_BATCH_SIZE = args.row_batch_size
            logger.info(f"Row batch size set to {args.row_batch_size} via CLI")
        
        if args.subsample_percent:
            config.SUBSAMPLE_PERCENT = args.subsample_percent
            logger.info(f"Subsample percentage set to {args.subsample_percent}% via CLI")
        
        
        # Cleanup if requested
        if args.cleanup:
            trainer.cleanup_temp_files()
            logger.info("Cleanup completed")
            return
        
        # Show current status
        summary = trainer.get_training_summary()
        logger.info(f"Training Status: {summary['completed']}/{summary['total_models']} completed")
        
        if args.validate_only:
            run_validation_only(trainer, logger)
            return
        
        # Train models
        if args.model:
            # Train specific model
            logger.info(f"Training {args.model} on full dataset...")
            start_time = time.time()
            
            success = trainer.train_single_model_enhanced(args.model, args.force_retrain)
            
            total_time = time.time() - start_time
            
            if success:
                logger.info(f"[SUCCESS] {args.model} training completed in {total_time:.2f} seconds")
            else:
                logger.error(f"[ERROR] {args.model} training failed after {total_time:.2f} seconds")
                sys.exit(1)
        else:
            # Train all models
            logger.info("Starting training of all models on full dataset...")
            logger.info("This will train on ~1,000 stocks with 5 years of data")
            logger.info("Progress will be logged in real-time")
            
            start_time = time.time()
            results = trainer.train_all_models_enhanced(args.force_retrain)
            total_time = time.time() - start_time
            
            # Print results
            print_training_results(results, logger, total_time)
        
        # Final status
        final_summary = trainer.get_training_summary()
        logger.info(f"Final Status: {final_summary['completed']}/{final_summary['total_models']} completed")
        
        # Cleanup temporary files
        logger.info("Cleaning up temporary files...")
        trainer.cleanup_temp_files()
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def show_training_status(trainer, logger):
    """Show detailed training status."""
    summary = trainer.get_training_summary()
    
    print("\n" + "="*100)
    print("FULL DATASET TRAINING STATUS SUMMARY")
    print("="*100)
    
    print(f"Total Models: {summary['total_models']}")
    print(f"Completed: {summary['completed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pending: {summary['pending']}")
    print(f"In Progress: {summary['in_progress']}")
    
    print("\nModel Details:")
    print("-" * 100)
    print(f"{'Model Name':<20} | {'Status':<12} | {'Stocks':<8} | {'Duration':<10} | {'R² Score':<10} | {'Details'}")
    print("-" * 100)
    
    for model_name, details in summary['models'].items():
        status = details['status']
        model_details = details['details']
        
        stocks_trained = model_details.get('stocks_trained', 0)
        duration = model_details.get('training_duration_seconds', 0)
        val_metrics = model_details.get('validation_metrics', {})
        r2 = val_metrics.get('avg_r2_score', 0)
        
        print(f"{model_name:<20} | {status:<12} | {stocks_trained:<8} | {duration:<10.1f} | {r2:<10.3f} | ", end="")
        
        if status == 'completed':
            dataset_size = model_details.get('dataset_size', 0)
            print(f"Dataset: {dataset_size:,} samples")
        elif status == 'failed':
            error = model_details.get('error_message', 'Unknown error')
            print(f"Error: {error[:50]}...")
        elif status == 'in_progress':
            start_time = model_details.get('start_time', 0)
            if start_time:
                elapsed = time.time() - start_time
                print(f"Running for {elapsed:.0f}s")
            else:
                print("Starting...")
        else:
            print("Not started")
    
    print("="*100)


def run_validation_only(trainer, logger):
    """Run validation on all completed models."""
    logger.info("Running validation on completed models...")
    
    summary = trainer.get_training_summary()
    completed_models = [name for name, details in summary['models'].items() 
                       if details['status'] == 'completed']
    
    if not completed_models:
        logger.warning("No completed models found for validation")
        return
    
    logger.info(f"Validating {len(completed_models)} completed models...")
    
    for model_name in completed_models:
        try:
            # Load the model
            model_path = os.path.join(trainer.models_dir, f"{model_name}_model.pkl")
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                continue
            
            # Get model class and load
            model_class = trainer.get_model_class(model_name)
            if model_class is None:
                logger.warning(f"Could not import {model_name}")
                continue
            
            model = model_class().load(model_path)
            
            # Run validation
            validation_metrics = trainer.validate_model(model, model_name)
            
            # Update status with new validation results
            trainer.status[model_name]['validation_metrics'] = validation_metrics
            trainer._save_status()
            
            logger.info(f"[SUCCESS] Validation completed for {model_name}")
            
        except Exception as e:
            logger.error(f"[ERROR] Error validating {model_name}: {e}")


def print_training_results(results, logger, total_time):
    """Print training results summary."""
    print("\n" + "="*100)
    print("FULL DATASET TRAINING RESULTS")
    print("="*100)
    
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"Successfully trained: {successful}/{total} models")
    print(f"Success rate: {successful/total*100:.1f}%")
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    
    print("\nModel Results:")
    print("-" * 100)
    
    for model_name, success in results.items():
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"{model_name:<20} | {status}")
    
    print("="*100)
    
    # Save results to file
    results_file = os.path.join(os.path.dirname(__file__), '..', 'models', 'full_dataset_training_results.json')
    try:
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': total_time,
            'successful_models': successful,
            'total_models': total,
            'success_rate': successful/total*100,
            'results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Results saved to {results_file}")
    except Exception as e:
        logger.warning(f"Could not save results file: {e}")


if __name__ == "__main__":
    main()
