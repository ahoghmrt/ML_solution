# Evaluation Report: 2026-03-06_02-34-17_full_optimized

## Configuration

| Parameter | Value |
|-----------|-------|
| Waveforms | 50000 |
| Epochs | 40 |
| Batch size | 128 |
| Test split | 0.2 |
| Min spacing (ns) | 0.0001 |
| Max signals | 7 |
| Noise std | 0.5 |

## Timing

| Step | Time (s) |
|------|----------|
| Step 1/7: Generating waveforms | 1.1 |
| Step 2/7: Subtracting baselines | 2.6 |
| Step 3/7: Preparing ML dataset | 0.3 |
| Steps 4-5/7: Training both models (parallel) | 377.9 |
| Step 6/7: Comparing predictions | 10.0 |
| Step 7/7: Plotting individual waveforms | 21.5 |
| **Total** | **413.4** |

## Count Model

- **Accuracy**: 85.7%
- **ROC AUC**: 0.9807

- Final train accuracy: 92.3%
- Final val accuracy: 85.7%
- Final train loss: 0.1880
- Final val loss: 0.3963
- Final learning rate: 3.125000148429535e-05

### Per-Class Accuracy

| Count | Accuracy |
|-------|----------|
| 0 | 100.0% |
| 1 | 99.8% |
| 2 | 99.1% |
| 3 | 95.9% |
| 4 | 87.0% |
| 5 | 79.4% |
| 6 | 83.8% |

### Top Misclassifications

| True | Predicted | Count |
|------|-----------|-------|
| 6 | 5 | 1108 |
| 5 | 6 | 740 |
| 5 | 4 | 684 |
| 4 | 3 | 483 |
| 4 | 5 | 450 |

## Signal Model

### Overall Metrics

| Metric | t0 (ns) | Amplitude |
|--------|---------|-----------|
| MAE | 3.9157 | 0.6889 |
| RMSE | 12.6018 | 1.4879 |
| Pearson r | 0.9377 | 0.9284 |
| Spearman r | 0.8639 | 0.8566 |

- Final train loss: 0.0884
- Final val loss: 0.1482
- Final train MAE: 0.0884
- Final val MAE: 0.1482

### Comparison Metrics (Hungarian-matched)

| Metric | t0 (ns) | Amplitude |
|--------|---------|-----------|
| MAE | 4.2683 | 0.9703 |
| RMSE | 9.9175 | 1.5548 |
| R² | 0.8947 | 0.6421 |
| Pearson r | 0.9463 | 0.8053 |

- **Count accuracy**: 92.2%
- **Matched signal pairs**: 148409

## Error Analysis

### Per-Slot MAE

| Slot | t0 MAE (ns) | Amp MAE | t0 RMSE (ns) | Amp RMSE |
|------|-------------|---------|---------------|----------|
| 0 | 0.8167 | 0.4657 | 1.2294 | 0.9028 |
| 1 | 3.0263 | 0.8272 | 5.9899 | 1.3183 |
| 2 | 5.2296 | 1.2614 | 9.6800 | 1.7943 |
| 3 | 8.4323 | 1.5403 | 16.7864 | 2.1507 |
| 4 | 11.6638 | 1.6523 | 23.8515 | 2.4179 |
| 5 | 18.0623 | 1.6837 | 34.5760 | 2.7783 |
| 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Error by Signal Count

| Count | t0 Bias (ns) | t0 Std (ns) | Amp Bias | Amp Std |
|-------|-------------|-------------|----------|---------|
| 1 | -0.2311 | 1.3565 | -0.0315 | 0.3082 |
| 2 | -0.3555 | 4.0150 | -0.0265 | 0.7565 |
| 3 | -0.3442 | 5.9852 | -0.0280 | 1.1167 |
| 4 | -1.3394 | 10.1411 | -0.0320 | 1.6155 |
| 5 | -0.9796 | 10.9467 | 0.0190 | 1.7445 |
| 6 | 0.1087 | 9.8318 | 0.1602 | 1.7759 |

### Temporal Error Profile

| Time Bin | t0 MAE (ns) |
|----------|-------------|
| 0-10ns | 1.2763 |
| 10-20ns | 1.7317 |
| 20-30ns | 2.3133 |
| 30-40ns | 3.0169 |
| 40-50ns | 3.4700 |
| 50-60ns | 3.7063 |
| 60-70ns | 3.8400 |
| 70-80ns | 4.0667 |
| 80-90ns | 4.5761 |
| 90-100ns | 5.5818 |
| 100-110ns | 7.1684 |
| 110-120ns | 8.8809 |

### Signal Spacing Impact

- Spacing vs t0 MAE (Spearman): -0.5678
- Spacing vs amp MAE (Spearman): -0.5790
- Multi-signal waveforms: 35751

### Error Percentiles

| Percentile | Total Error |
|------------|-------------|
| p50 | 7.42 |
| p90 | 39.07 |
| p95 | 55.52 |
| p99 | 106.91 |

### Residual Statistics

| Metric | t0 | Amplitude |
|--------|----|-----------| 
| Mean | -0.5521 | 0.0351 |
| Std | 9.0623 | 1.5324 |
| Skewness | -3.9769 | 0.0199 |
| Kurtosis | 43.6216 | 4.9349 |

---
*Report generated for experiment: 2026-03-06_02-34-17_full_optimized*