# Training Module - Standalone Model Trainers

Independent training scripts for each ML model. Each model has its own optimized trainer.

## Structure

```
training/
├── basic_models/
│   ├── linear_regression/    (config.py, trainer.py)
│   ├── decision_tree/         (config.py, trainer.py)
│   ├── random_forest/         (config.py, trainer.py)
│   └── svm/                   (config.py, trainer.py)
├── advanced_models/
│   ├── knn/                   (config.py, trainer.py)
│   ├── arima/                 (config.py, trainer.py)
│   └── autoencoder/           (config.py, trainer.py)
├── common_trainer_utils.py    # Shared utilities
├── display_manager.py          # Progress display
└── train_full_dataset.py       # Runs all trainers
```

## Quick Start

### Train Individual Model
```bash
python backend/training/basic_models/linear_regression/trainer.py
python backend/training/basic_models/random_forest/trainer.py
python backend/training/advanced_models/knn/trainer.py
```

### Train All Models
```bash
python backend/training/train_full_dataset.py
```

### Options
```bash
# Test with limited stocks
--max-stocks 100

# Force retrain
--force-retrain
```

## Models

| Model | Type | Time | Notes |
|-------|------|------|-------|
| Linear Regression | Basic | ~2 min | Fastest |
| Decision Tree | Basic | ~3 min | Interpretable |
| Random Forest | Basic | ~8 min | Ensemble |
| SVM | Basic | ~15 min | Kernel optimization |
| KNN | Advanced | ~5 min | Distance-based |
| ARIMA | Advanced | ~25 min | Time series |
| Autoencoder | Advanced | ~18 min | Neural network |

## Configuration

Edit hyperparameters in each model's `config.py`:
```python
# Example: backend/training/basic_models/random_forest/config.py
class RandomForestConfig:
    N_ESTIMATORS = 100
    MAX_DEPTH = 15
    VERBOSE = True
```

## Output

Models save to: `backend/models/{model_name}/{model_name}_model.pkl`  
Logs save to: `backend/logs/{model_name}_training_{timestamp}.log`

## Algorithm Implementations

Model code is in: `backend/algorithms/optimised/`  
Training code is in: `backend/training/basic_models/` or `advanced_models/`
