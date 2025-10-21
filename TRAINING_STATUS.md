# ML Model Training Status

**Last Updated:** 2025-10-21 06:42:53  
**Total Models:** 9  
**Completed:** 4  
**Failed:** 3  
**Pending:** 2

## Model Comparison Table

| Icon | Model            | Status   | Stocks   | Dataset Size | R² Score      | Performance   | Trained Date        | Error                |
|------|------------------|----------|----------|--------------|---------------|---------------|---------------------|----------------------|
| ⚠️   | ANN              | TRAINED  | 913      | 1,053,635    | -9,757,687.1  | Catastrophic  | 2025-10-21 06:11    | Needs retraining     |
| ❌   | ARIMA            | FAILED   | 0        | 0            | N/A           | N/A           | None                | Training incomplete  |
| ⚠️   | Autoencoder      | TRAINED  | 17/913   | 0            | -136,355.3    | Poor          | 2025-10-21 02:39    | Incomplete training  |
| ❌   | CNN              | FAILED   | 0        | 0            | N/A           | N/A           | None                | Stuck at batch 10/11 |
| ✅   | Decision Tree    | TRAINED  | 913      | 1,053,586    | 0.85          | Good          | 2025-10-21 05:44    | None                 |
| ✅   | KNN              | TRAINED  | 913      | 1,053,586    | -27.9         | Poor          | 2025-10-21 05:41    | None                 |
| ❌   | Linear Regression| FAILED   | 0        | 0            | N/A           | N/A           | None                | Stuck at batch 9/11  |
| ✅   | Random Forest    | TRAINED  | 913      | 1,053,586    | 0.994         | Excellent     | 2025-10-21 03:26    | None                 |
| ✅   | SVM              | TRAINED  | 913      | 1,053,586    | -26.2         | Poor          | 2025-10-21 05:40    | None                 |

## Performance Ranking (by R² Score)

| Rank | Model | R² Score | Performance Level |
|------|-------|----------|-------------------|
| 1 | Random Forest | 0.994 | 🏆 Excellent |
| 2 | Decision Tree | 0.85 | 🥈 Good |
| 3 | SVM | -26.2 | 🥉 Poor |
| 4 | KNN | -27.9 | Poor |
| 5 | Autoencoder | -136,355.3 | Very Poor |
| 6 | ANN | -9,757,687.1 | Catastrophic |
| 7-9 | Linear Regression, CNN, ARIMA | N/A | Failed Training |

## Summary

- **Successfully trained models:** 4/9 (44%)
- **Models with good performance:** 2/9 (Random Forest, Decision Tree)
- **Models needing retraining:** 2 (ANN, Autoencoder)
- **Failed models:** 3 (Linear Regression, CNN, ARIMA)

## Next Steps

1. **Retrain ANN** with better hyperparameters
2. **Retrain Autoencoder** on full dataset
3. **Fix Linear Regression** batch training issue
4. **Fix CNN** batch training issue
5. **Fix ARIMA** training completion issue

## Training Configuration

- **Total Stocks Available:** ~1,000
- **Historical Data:** 5 years
- **Batch Training:** Enabled
- **Stock Batch Size:** 100
- **Row Batch Size:** 50,000
- **Subsample Percentage:** 50%

---

*This file can be easily updated manually or through scripts as training progresses.*
