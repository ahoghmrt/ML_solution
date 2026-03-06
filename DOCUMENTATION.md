# ADC Waveform Signal Extraction — ML Pipeline Documentation

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Pipeline Steps](#pipeline-steps)
  - [Step 1: Waveform Generation](#step-1-waveform-generation)
  - [Step 2: Baseline Subtraction](#step-2-baseline-subtraction)
  - [Step 3: Dataset Preparation](#step-3-dataset-preparation)
  - [Step 4: Count Model Training](#step-4-count-model-training)
  - [Step 5: Signal Model Training](#step-5-signal-model-training)
  - [Step 6: Prediction Comparison](#step-6-prediction-comparison)
  - [Step 7: Waveform Inspection Plots](#step-7-waveform-inspection-plots)
  - [Step 8: Error Analysis](#step-8-error-analysis)
  - [Step 9: Evaluation Report](#step-9-evaluation-report)
- [Design Choices & Algorithms](#design-choices--algorithms)
  - [Conv1D Architecture](#conv1d-architecture)
  - [Permutation-Invariant Training (PIT)](#permutation-invariant-training-pit)
  - [Hungarian Matching](#hungarian-matching)
  - [Weighted Huber Loss](#weighted-huber-loss)
  - [Baseline Subtraction: Rolling Quantile](#baseline-subtraction-rolling-quantile)
  - [BatchNormalization](#batchnormalization)
  - [Class Weights](#class-weights)
  - [Varied Noise Augmentation](#varied-noise-augmentation)
- [Configuration Reference](#configuration-reference)
- [Experiment Tracking](#experiment-tracking)
- [TensorBoard Integration](#tensorboard-integration)
- [Performance Optimizations](#performance-optimizations)

---

## Overview

This project uses deep learning to extract overlapping signal parameters from simulated ADC (Analog-to-Digital Converter) waveforms. Given a 120-bin digitized waveform containing 0–6 overlapping bi-exponential pulses buried in noise and baseline offset, the pipeline:

1. **Predicts how many signals** are present (classification, 0–6)
2. **Extracts each signal's timing (t0) and amplitude** (regression)

The full pipeline — from synthetic data generation through model training to comprehensive error analysis — is orchestrated via a single CLI.

---

## Problem Statement

A detector digitizes analog signals at 1 GHz (1 ns/sample) over a 120 ns window. Each waveform may contain 0–6 overlapping pulses, each described by a bi-exponential shape:

```
pulse(t) = A × (1 − exp(−(t − t0) / τ_rise)) × exp(−(t − t0) / τ_fall)    for t ≥ t0
         = 0                                                                  for t < t0
```

With `τ_rise = 2 ns` and `τ_fall = 10 ns`, the pulse has a fast rise and slow decay. Pulses overlap when their t0 values are close, making extraction difficult — especially with additive Gaussian noise (σ = 0.5) and a baseline offset (200 units).

**Goal**: Given only the raw digitized waveform, recover the number of signals and each signal's peak time and peak amplitude.

---

## Repository Structure

```
ML_solution/
├── cli.py                          # Command-line interface (main entry point)
├── config.py                       # Central configuration (all parameters)
├── gen_wave.py                     # Step 1: Synthetic waveform generation
├── baseline_subtract.py            # Step 2: Baseline removal
├── prepare_ml_dataset.py           # Step 3: Dataset preparation
├── train_count_model.py            # Step 4: Count classifier training
├── train_signal_model.py           # Step 5: Signal regressor training (with PIT)
├── compare_signal_predictions.py   # Step 6: Evaluation & comparison plots
├── plot_individual_waveform.py     # Step 7: Per-waveform visualization
├── error_analysis/
│   └── analyze.py                  # Step 8: Deep error diagnostics
├── generate_report.py              # Step 9: Markdown evaluation report
├── experiments/                    # Timestamped experiment snapshots
├── tensorboard_logs/               # TensorBoard training logs
├── waveform_raw/                   # Generated raw waveforms (data.npz)
├── waveform_baseline_removed/      # Baseline-subtracted waveforms (data.npz)
├── ml_training_data/               # Training datasets (.npz)
├── training_plots/                 # Scalers (.pkl), training curves, histories
├── comparison_plots/               # Prediction comparison plots
├── waveform_inspection/            # Individual waveform plots
├── logs/                           # Pipeline execution logs
└── reports/                        # Beamer presentation (.tex, .pdf)
```

---

## Quick Start

### Run the full pipeline end-to-end

```bash
python cli.py run-all --experiment-name my_experiment
```

This executes all 9 steps sequentially (steps 4–5 in parallel), saves results to `experiments/<timestamp>_my_experiment/`, and prints a full evaluation report.

### Run with custom parameters

```bash
python cli.py run-all \
    --num-waveforms 100000 \
    --epochs 50 \
    --batch-size 256 \
    --no-pit \
    --experiment-name large_no_pit
```

### Run individual steps

```bash
python cli.py generate --num-waveforms 50000
python cli.py baseline
python cli.py prepare
python cli.py train-count --epochs 40
python cli.py train-signal --epochs 30 --no-pit
python cli.py compare
python cli.py plot --start 1 --end 100
python cli.py analyze --experiment experiments/2026-03-06_my_experiment
python cli.py report --experiment experiments/2026-03-06_my_experiment
```

---

## CLI Reference

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `run-all` | Full pipeline end-to-end | `--num-waveforms`, `--epochs`, `--batch-size`, `--experiment-name`, `--pit`/`--no-pit`, `--varied-noise`/`--no-varied-noise`, `--min-spacing` |
| `generate` | Generate synthetic waveforms | `--num-waveforms`, `--noise-std`, `--baseline`, `--min-spacing`, `--max-signals`, `--varied-noise` |
| `baseline` | Subtract baseline from waveforms | `--window-size`, `--quantile` |
| `prepare` | Create ML training datasets | `--max-signals` |
| `train-count` | Train signal count classifier | `--epochs`, `--batch-size`, `--test-size` |
| `train-signal` | Train signal parameter regressor | `--epochs`, `--batch-size`, `--test-size`, `--pit`/`--no-pit` |
| `compare` | Generate prediction comparison plots | *(none)* |
| `plot` | Plot individual waveforms | `--start`, `--end` |
| `analyze` | Run error analysis on experiment | `--experiment` (path, required) |
| `report` | Generate markdown evaluation report | `--experiment` (path, required) |

---

## Pipeline Steps

### Step 1: Waveform Generation

**File**: `gen_wave.py`

Generates synthetic ADC waveforms with known ground truth parameters.

**Process**:
1. For each signal count (0–6), generate a batch of waveforms
2. For each waveform, sample random t0 values (uniform in [5 ns, 110 ns]) and amplitudes (uniform in [5, 20])
3. Enforce minimum inter-signal spacing (2.0 ns default) — re-sample if signals are too close
4. Compute bi-exponential pulses and sum them
5. Add Gaussian noise and DC baseline offset (200 units)
6. Record ground truth: peak time and peak amplitude for each signal (via `argmax`)

**Output**: `waveform_raw/data.npz` containing:
- `waveforms` (N × 120): raw digitized waveforms
- `truth` (N × 7 × 2): [peak_t0, peak_amplitude] per signal slot (zero-padded)
- `counts` (N,): true signal count per waveform
- `time` (120,): time axis in nanoseconds

**Implementation detail**: Waveforms are generated in vectorized batches grouped by signal count, using NumPy broadcasting to compute all pulses simultaneously — no Python loops over individual waveforms.

---

### Step 2: Baseline Subtraction

**File**: `baseline_subtract.py`

Removes the DC baseline offset and any slow drift.

**Algorithm**: Rolling quantile (10th percentile) over a 31-sample window.

For each waveform, the local baseline at each time bin is estimated as the 10th percentile of all values within ±15 bins. This baseline is then subtracted.

**Why rolling quantile?**
- A low quantile (10th percentile) tracks the noise floor, not the signal peaks
- Rolling windows adapt to time-varying baseline
- Robust to outliers from signal pulses (unlike mean or median)

**Output**: `waveform_baseline_removed/data.npz` with baseline-subtracted waveforms.

**Performance**: All 50k waveforms are processed at once using a DataFrame transpose trick — rows become time bins and columns become waveforms, allowing vectorized rolling operations.

---

### Step 3: Dataset Preparation

**File**: `prepare_ml_dataset.py`

Merges baseline-subtracted waveforms with ground truth labels into two separate training datasets:

- `training_data_counts.npz`: waveforms + integer count labels (for classification)
- `training_data_signals.npz`: waveforms + signal parameter labels (for regression)

---

### Step 4: Count Model Training

**File**: `train_count_model.py`

Trains a Conv1D classifier to predict the number of signals (0–6) in a waveform.

**Architecture**:
```
Input (120, 1)
  → Conv1D(64 filters, kernel=5, relu, padding='same')
  → BatchNormalization
  → Conv1D(128 filters, kernel=5, relu, padding='same')
  → BatchNormalization
  → Flatten
  → Dense(256, relu) → BatchNorm → Dropout(0.3)
  → Dense(128, relu)
  → Dense(7, softmax)                    ← 7 classes: counts 0–6
```

**Training details**:
- Loss: `sparse_categorical_crossentropy`
- Optimizer: Adam (with ReduceLROnPlateau: factor 0.5, patience 3)
- Early stopping: patience 6 on val_accuracy
- Optional class weights: balanced (compensates for unequal count distribution)
- Input normalization: StandardScaler (zero-mean, unit-variance)

**Outputs**: `signal_count_model.keras`, confusion matrix, ROC/PR curves, training history.

---

### Step 5: Signal Model Training

**File**: `train_signal_model.py`

Trains a Conv1D regressor to predict (t0, amplitude) for each of the 7 signal slots.

**Architecture**:
```
Input (120, 1)
  → Conv1D(64, kernel=5, relu, padding='same')
  → BatchNormalization
  → Conv1D(128, kernel=5, relu, padding='same')
  → BatchNormalization
  → Flatten
  → Dense(256, relu) → BatchNorm → Dropout(0.3)
  → Dense(128, relu)
  → Dense(14)                            ← 7 slots × 2 params (t0, amp)
```

**Target normalization**: Separate StandardScalers for t0 and amplitude values, ensuring both are on comparable scales during training.

**Two training modes**:
- **Standard** (`--no-pit`): Uses Keras `model.fit()` with callbacks. Fast but doesn't handle signal ordering ambiguity.
- **PIT** (`--pit`, default): Custom training loop with Hungarian matching at each batch. Slower but handles the permutation problem correctly.

See [Permutation-Invariant Training](#permutation-invariant-training-pit) and [Hungarian Matching](#hungarian-matching) for details.

**Outputs**: `signal_model.keras`, scalers (`t0_scaler.pkl`, `amp_scaler.pkl`, `waveform_scaler.pkl`), per-slot MAE bar charts.

---

### Step 6: Prediction Comparison

**File**: `compare_signal_predictions.py`

Evaluates both models on the full dataset using Hungarian matching for fair signal alignment.

**Process**:
1. Predict counts and signal parameters for all waveforms
2. For each waveform, extract active signals (non-zero) from both prediction and truth
3. Hungarian-match predicted signals to true signals (minimizing L1 distance)
4. Collect all matched pairs and compute metrics

**Metrics**:
- t0: MAE, RMSE, R², Pearson r, Spearman r
- Amplitude: MAE, RMSE, R², Pearson r, Spearman r
- Count accuracy
- Per-count breakdown

**Plots**: Hexbin scatter plots (true vs. predicted) with error histograms, count confusion matrix.

---

### Step 7: Waveform Inspection Plots

**File**: `plot_individual_waveform.py`

Generates per-waveform visualizations showing the raw waveform with true signals (blue dots), predicted signals (green X's), and dashed lines connecting Hungarian-matched pairs.

Default: plots waveforms 1–300. Uses batch prediction (2 model calls instead of 600) for efficiency.

---

### Step 8: Error Analysis

**File**: `error_analysis/analyze.py`

Performs 8 deep diagnostic analyses on experiment results:

| Analysis | What it reveals |
|----------|-----------------|
| **Per-slot MAE** | Accuracy per signal rank (1st signal by time, 2nd, etc.) — later signals are harder |
| **Error vs. count** | How error scales with signal multiplicity (more signals → worse) |
| **Temporal error profile** | Edge effects — are signals near window boundaries predicted poorly? |
| **Amplitude dependency** | Are low-amplitude signals harder to detect/locate? |
| **Spacing analysis** | How does inter-signal spacing affect accuracy? (closely spaced → worse) |
| **Worst cases** | Visual inspection of the 20 most poorly predicted waveforms |
| **Count model analysis** | Confusion matrix, per-class accuracy, systematic biases |
| **Residual QQ plots** | Are residuals normally distributed? Skewness/kurtosis analysis |

**Outputs**: 8 PNG plots, `error_analysis_report.json` (full numerical results), `summary.pdf` (all plots combined).

---

### Step 9: Evaluation Report

**File**: `generate_report.py`

Compiles all metrics, training histories, and error analysis results into a single human-readable markdown report (`report.md`). Includes:

- Configuration table
- Timing breakdown
- Count model performance (accuracy, per-class, misclassifications)
- Signal model performance (MAE, RMSE, correlations)
- Error analysis tables (per-slot, per-count, temporal, spacing, residuals)

Auto-generated at the end of every `run-all` pipeline execution.

---

## Design Choices & Algorithms

### Conv1D Architecture

**Why Conv1D instead of Dense or RNN?**

ADC waveforms are 1D time series with local temporal structure — pulse shapes span ~15–20 bins. Conv1D filters with kernel size 5 act as learnable feature detectors that slide across the time axis, recognizing pulse features regardless of position (translation invariance).

**Architecture rationale**:
- **Two Conv1D layers** (64 → 128 filters): First layer detects basic features (edges, peaks), second layer combines them into higher-level patterns (full pulse shapes, overlapping pulses)
- **`padding='same'`**: Preserves temporal resolution through layers
- **Flatten → Dense**: After convolutional feature extraction, dense layers combine spatial features for the final prediction
- **Dropout (0.3)**: Regularization to prevent overfitting on training data

### Permutation-Invariant Training (PIT)

**The problem**: The signal model outputs 14 values: `[t0_0, amp_0, t0_1, amp_1, ..., t0_6, amp_6]`. But there's no inherent ordering — if a waveform has signals at t=30 ns and t=70 ns, the model could predict them in either slot. Without PIT, the model is penalized for correct predictions in the "wrong" slot, creating a conflicting learning signal.

**Example**:
```
True:     slot 0 = (30 ns, 10.0),  slot 1 = (70 ns, 15.0)
Pred A:   slot 0 = (31 ns, 9.8),   slot 1 = (69 ns, 14.7)  ← correct!
Pred B:   slot 0 = (69 ns, 14.7),  slot 1 = (31 ns, 9.8)   ← also correct!
```

Without PIT, Pred B would incur a large loss despite being equally correct.

**The solution**: Before computing the loss at each training step:
1. Get the model's predictions
2. Find the optimal assignment between predicted slots and true signals using Hungarian matching
3. Reorder the true labels to match the predicted slot ordering
4. Compute loss on the reordered targets

This way, the model is always evaluated against the most favorable permutation of ground truth.

**Trade-off**: PIT requires a custom training loop (can't use standard Keras `model.fit()`) and runs Hungarian matching per-sample, making it ~10–20x slower than standard training. Use `--no-pit` for faster experimentation.

### Hungarian Matching

**What**: The Hungarian algorithm (also called the Kuhn–Munkres algorithm) solves the optimal assignment problem — given a cost matrix, find the assignment of rows to columns that minimizes total cost.

**How it's used in this pipeline**:

1. **Cost matrix construction**: For a waveform with n predicted and m true signals:
   ```
   cost[i, j] = |pred_t0[i] - true_t0[j]| + |pred_amp[i] - true_amp[j]|
   ```
   This is the L1 distance between predicted signal `i` and true signal `j`.

2. **Optimal matching**: `scipy.optimize.linear_sum_assignment(cost)` returns the assignment that minimizes total cost. Complexity: O(n³).

3. **Usage points**:
   - **During PIT training**: Reorder truth labels to match predictions (per batch, per sample)
   - **During evaluation** (compare, plot, error analysis): Align predicted signals to true signals for fair metric computation

**Why not just sort by t0?** Sorting is a heuristic that fails when predictions are noisy — a predicted t0 might be closer to a different true signal than the one in the same sorted position. Hungarian matching finds the globally optimal assignment.

### Weighted Huber Loss

**Standard MAE** treats all errors equally and all output components equally. Two improvements:

1. **Huber loss** (δ = 1.0):
   ```
   L(e) = 0.5 × e²          if |e| ≤ δ
        = δ × (|e| - 0.5δ)   if |e| > δ
   ```
   - Quadratic near zero (smooth gradients, stable training)
   - Linear for large errors (robust to outliers, unlike MSE)
   - Combines benefits of MAE (robustness) and MSE (smoothness)

2. **Per-component weights**: t0 components receive weight 2.0, amplitude components receive weight 1.0. This prioritizes timing accuracy, which is typically the harder and more important quantity.

### Baseline Subtraction: Rolling Quantile

**Why not just subtract the mean?** The baseline may drift over time, and the mean is influenced by signal peaks. The 10th percentile of a rolling window:
- Tracks the noise floor (below signal peaks)
- Adapts to slow baseline drift
- Is robust to outlier values from signals

Window size 31 (±15 bins) balances temporal resolution against noise smoothing.

### BatchNormalization

Added after each Conv1D layer and after the first Dense layer. Benefits:
- **Stabilizes training**: Normalizes activations to zero mean, unit variance
- **Allows higher learning rates**: Reduces internal covariate shift
- **Acts as mild regularization**: Noise from mini-batch statistics

### Class Weights

The count distribution is roughly uniform (0–6), but small imbalances can bias the classifier. Using `sklearn.utils.compute_class_weight('balanced')` assigns inverse-frequency weights, ensuring minority classes contribute equally to the loss.

### Varied Noise Augmentation

When `VARIED_NOISE=True`, each waveform receives a noise level sampled from `[0.2, 0.3, 0.4, 0.5]` instead of a fixed `σ = 0.5`. This trains the model to handle varying noise conditions, improving generalization to real-world data where noise levels aren't constant.

---

## Configuration Reference

All parameters are in `config.py`. Key settings:

### Physics & Generation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `TIME_START` / `TIME_END` | 0 / 120 ns | Time window |
| `SAMPLING_RATE` | 1 ns | 1 GHz digitization |
| `TAU_RISE` / `TAU_FALL` | 2 / 10 ns | Bi-exponential pulse shape |
| `AMPLITUDE_MIN` / `MAX` | 5 / 20 | Signal amplitude range |
| `NUM_WAVEFORMS` | 50,000 | Training dataset size |
| `NOISE_STD` | 0.5 | Gaussian noise σ |
| `BASELINE` | 200.0 | DC baseline offset |
| `MIN_SPACING` | 2.0 ns | Minimum inter-signal spacing |
| `MAX_SIGNALS` | 7 | Max signals per waveform (padding size) |

### Model Architecture
| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONV_FILTERS` | [64, 128] | Filters per Conv1D layer |
| `CONV_KERNEL_SIZE` | 5 | Convolution kernel width |
| `DENSE_UNITS` | [256, 128] | Dense layer sizes |
| `DROPOUT_RATE` | 0.3 | Dropout probability |

### Training
| Parameter | Default | Description |
|-----------|---------|-------------|
| `COUNT_MODEL_EPOCHS` | 30 | Max epochs for count model |
| `SIGNAL_MODEL_EPOCHS` | 20 | Max epochs for signal model |
| `COUNT_MODEL_BATCH_SIZE` | 128 | Batch size (count) |
| `SIGNAL_MODEL_BATCH_SIZE` | 128 | Batch size (signal) |
| `EARLY_STOPPING_PATIENCE` | 6 | Epochs without improvement |
| `LR_REDUCE_PATIENCE` | 3 | Epochs before reducing LR |
| `LR_REDUCE_FACTOR` | 0.5 | LR reduction multiplier |
| `LR_MIN` | 1e-6 | Minimum learning rate |

### Feature Flags
| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_PIT_LOSS` | True | Enable PIT (overridden by `--pit`/`--no-pit`) |
| `USE_BATCHNORM` | True | BatchNormalization layers |
| `USE_CLASS_WEIGHTS` | True | Balanced class weights for count model |
| `VARIED_NOISE` | True | Per-waveform noise level sampling |
| `SIGNAL_LOSS_TYPE` | `"weighted_huber"` | Loss function (`"weighted_huber"` or `"mae"`) |
| `T0_LOSS_WEIGHT` | 2.0 | t0 weight in weighted Huber loss |
| `HUBER_DELTA` | 1.0 | Huber loss transition point |

---

## Experiment Tracking

Every `run-all` execution saves a complete experiment snapshot:

```
experiments/<timestamp>_<name>/
├── config.json               # All pipeline parameters
├── metrics.json              # Training + evaluation metrics + timing
├── pipeline.log              # Full execution log
├── report.md                 # Human-readable evaluation summary
├── training_plots/           # Training curves, scalers, histories
├── comparison_plots/         # True vs. predicted scatter plots
├── waveform_inspection/      # Per-waveform visualizations
├── tensorboard/              # TensorBoard event files
└── error_analysis/           # 8 diagnostic analyses + JSON report + PDF
```

Experiments are fully self-contained and reproducible — `config.json` records all parameters used.

### Comparing experiments

Generate a report for any past experiment:
```bash
python cli.py report --experiment experiments/2026-03-06_my_experiment
```

Run additional error analysis:
```bash
python cli.py analyze --experiment experiments/2026-03-06_my_experiment
```

---

## TensorBoard Integration

Training metrics are logged to TensorBoard for real-time monitoring.

### Launch TensorBoard
```bash
tensorboard --logdir tensorboard_logs/
```

### What's logged
- **Count model**: accuracy, loss, val_accuracy, val_loss, learning_rate (via Keras TensorBoard callback)
- **Signal model (standard)**: loss, mae, val_loss, val_mae, learning_rate (via Keras TensorBoard callback)
- **Signal model (PIT)**: epoch_loss, epoch_mae, epoch_val_loss, epoch_val_mae, epoch_learning_rate (via manual `tf.summary.scalar()` since the custom PIT loop bypasses Keras callbacks)

Logs are organized as `tensorboard_logs/<experiment_name>/{count_model,signal_model}/` and copied to the experiment folder.

---

## Performance Optimizations

| Optimization | Before | After | Speedup |
|-------------|--------|-------|---------|
| **File format** | 100k+ individual .txt files | Single `.npz` per step | ~50x I/O |
| **Waveform generation** | Python loop per waveform | Vectorized NumPy broadcasting per count group | ~100x |
| **Baseline subtraction** | Per-row Pandas Series | Bulk DataFrame transpose trick | ~10x |
| **Model training** | Sequential (count then signal) | Parallel via `ProcessPoolExecutor` | ~2x |
| **Waveform plotting** | 598 individual `model.predict()` calls | 2 batch `predict()` calls | ~100x |
| **PIT disabled** | ~40 min training (50k, 20 epochs) | ~5 min standard `model.fit()` | ~8x |

---

## Typical Results

With default configuration (50k waveforms, 30/20 epochs, PIT enabled):

- **Count model accuracy**: ~92–95%
- **t0 MAE**: ~3–5 ns
- **Amplitude MAE**: ~0.7–1.3
- **Total pipeline time**: ~5–7 min (no PIT), ~30–60 min (with PIT)

Performance degrades with higher signal counts (5–6 signals) and closely spaced signals (< 5 ns apart), as expected from the physics of overlapping pulses.
