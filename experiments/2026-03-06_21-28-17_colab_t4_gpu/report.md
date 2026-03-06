# Evaluation Report: 2026-03-06_21-28-17_colab_t4_gpu

## Configuration

| Parameter | Value |
|-----------|-------|
| Waveforms | 50000 |
| Epochs | 30 |
| Batch size | 128 |
| Test split | 0.2 |
| Min spacing (ns) | 2.0 |
| Max signals | 7 |
| Noise std | 0.5 |
| Varied noise | True |
| PIT (Hungarian) | True |

## Timing

| Step | Time (s) |
|------|----------|
| Step 1/7: Generating waveforms | 1.1 |
| Step 2/7: Subtracting baselines | 5.5 |
| Step 3/7: Preparing ML dataset | 0.3 |
| Steps 4-5/7: Training both models (parallel) | 1187.1 |
| Step 6/7: Comparing predictions | 21.2 |
| Step 7/7: Plotting individual waveforms | 47.6 |
| **Total** | **1262.9** |

## Count Model

- **Accuracy**: 95.2%
- **ROC AUC**: 0.9962

- Final train accuracy: 98.0%
- Final val accuracy: 95.2%
- Final train loss: 0.0525
- Final val loss: 0.1682
- Final learning rate: 0.0005000000237487257

### Per-Class Accuracy

| Count | Accuracy |
|-------|----------|
| 0 | 100.0% |
| 1 | 100.0% |
| 2 | 99.8% |
| 3 | 99.4% |
| 4 | 96.7% |
| 5 | 94.0% |
| 6 | 91.6% |

### Top Misclassifications

| True | Predicted | Count |
|------|-----------|-------|
| 6 | 5 | 593 |
| 5 | 4 | 392 |
| 4 | 3 | 216 |
| 5 | 6 | 43 |
| 3 | 2 | 40 |

## Signal Model

### Overall Metrics

| Metric | t0 (ns) | Amplitude |
|--------|---------|-----------|
| MAE | 18.8880 | 1.8487 |
| RMSE | 30.2091 | 2.8208 |
| Pearson r | 0.5660 | 0.7168 |
| Spearman r | 0.6801 | 0.7480 |

- Final train loss: 0.0133
- Final val loss: 0.0177
- Final train MAE: 0.0894
- Final val MAE: 0.0886

### Comparison Metrics (Hungarian-matched)

| Metric | t0 (ns) | Amplitude |
|--------|---------|-----------|
| MAE | 18.8605 | 1.9362 |
| RMSE | 26.3633 | 2.6130 |
| R² | 0.2427 | -0.0818 |
| Pearson r | 0.7194 | 0.6382 |

- **Count accuracy**: 97.3%
- **Matched signal pairs**: 149171

## Error Analysis

### Per-Slot MAE

| Slot | t0 MAE (ns) | Amp MAE | t0 RMSE (ns) | Amp RMSE |
|------|-------------|---------|---------------|----------|
| 0 | 9.5340 | 2.3831 | 13.9014 | 3.1606 |
| 1 | 13.3610 | 1.6211 | 19.1881 | 2.3475 |
| 2 | 21.8982 | 1.7267 | 29.1991 | 2.3207 |
| 3 | 28.0205 | 1.8477 | 34.5938 | 2.3168 |
| 4 | 32.1480 | 1.9618 | 37.5126 | 2.4729 |
| 5 | 39.1123 | 1.8673 | 44.2441 | 2.4034 |
| 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Error by Signal Count

| Count | t0 Bias (ns) | t0 Std (ns) | Amp Bias | Amp Std |
|-------|-------------|-------------|----------|---------|
| 1 | -0.6899 | 1.2761 | -0.2833 | 0.3095 |
| 2 | -10.3748 | 21.1522 | -0.4356 | 1.9302 |
| 3 | -17.0680 | 26.0288 | -1.2251 | 2.3217 |
| 4 | -19.8738 | 23.3696 | -1.1894 | 2.1919 |
| 5 | -15.2983 | 19.8752 | -1.1291 | 2.3045 |
| 6 | -17.4205 | 17.6707 | -1.1605 | 2.8648 |

### Temporal Error Profile

| Time Bin | t0 MAE (ns) |
|----------|-------------|
| 0-10ns | 14.2685 |
| 10-20ns | 12.2045 |
| 20-30ns | 7.8814 |
| 30-40ns | 4.7120 |
| 40-50ns | 5.1573 |
| 50-60ns | 10.4563 |
| 60-70ns | 15.3811 |
| 70-80ns | 21.2123 |
| 80-90ns | 27.3291 |
| 90-100ns | 34.0330 |
| 100-110ns | 42.5556 |
| 110-120ns | 48.5316 |

### Signal Spacing Impact

- Spacing vs t0 MAE (Spearman): 0.1451
- Spacing vs amp MAE (Spearman): -0.2445
- Multi-signal waveforms: 35850

### Error Percentiles

| Percentile | Total Error |
|------------|-------------|
| p50 | 65.62 |
| p90 | 128.46 |
| p95 | 138.64 |
| p99 | 154.05 |

### Residual Statistics

| Metric | t0 | Amplitude |
|--------|----|-----------| 
| Mean | -15.8229 | -1.0546 |
| Std | 21.0870 | 2.3907 |
| Skewness | -0.7976 | -0.4808 |
| Kurtosis | 0.3017 | 0.5557 |

---
*Report generated for experiment: 2026-03-06_21-28-17_colab_t4_gpu*