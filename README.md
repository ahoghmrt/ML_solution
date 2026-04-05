# ADC Waveform Signal Extraction

A machine learning pipeline that extracts signal parameters from simulated ADC (Analog-to-Digital Converter) waveforms. Given a digitized waveform containing 0-6 overlapping pulses with noise and baseline offset, the pipeline predicts:

- **How many signals** are present (classification)
- **Where and how large** each signal is &mdash; timing (t0) and amplitude (regression)

## How It Works

Waveforms are modeled as sums of bi-exponential pulses (&tau;<sub>rise</sub> = 2 ns, &tau;<sub>fall</sub> = 10 ns) on a 120-bin time window at 1 GHz sampling, with Gaussian noise and a DC baseline offset. Two Conv1D neural networks handle the extraction:

1. **Count model** &mdash; classifies the number of signals (0&ndash;6) in a waveform
2. **Signal model** &mdash; regresses (t0, amplitude) pairs for up to 7 signals, trained with permutation-invariant loss (Hungarian matching)

### Pipeline

```
gen_wave.py          Generate synthetic waveforms with known truth
       |
baseline_subtract.py Rolling-quantile baseline removal
       |
prepare_ml_dataset.py   Create .npz training datasets
       |
  +----+----+
  |         |            (trained in parallel)
train_count  train_signal
  |         |
  +----+----+
       |
compare_signal_predictions.py   Evaluation plots & metrics
       |
plot_individual_waveform.py     Per-waveform visualization
       |
error_analysis/analyze.py      Deep error diagnostics
```

## Quick Start

### Requirements

- Python 3.10+
- TensorFlow 2.19+
- GPU optional (auto-detected, CUDA-compatible)

```bash
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python cli.py run-all --experiment-name my_experiment
```

This generates 50,000 synthetic waveforms, trains both models, evaluates, and saves everything to `experiments/<timestamp>_my_experiment/`.

### Run Individual Steps

```bash
python cli.py generate          # Step 1: Generate waveforms
python cli.py baseline          # Step 2: Subtract baselines
python cli.py prepare           # Step 3: Prepare ML datasets
python cli.py train-count       # Step 4: Train count classifier
python cli.py train-signal      # Step 5: Train signal regressor
python cli.py compare           # Step 6: Evaluation plots
python cli.py plot              # Step 7: Waveform inspection plots
python cli.py analyze --experiment experiments/<folder>  # Step 8: Error analysis
```

### Key Options

| Flag | Description | Default |
|------|-------------|---------|
| `--num-waveforms` | Number of training waveforms | 50,000 |
| `--epochs` | Training epochs | 30 (count) / 20 (signal) |
| `--batch-size` | Training batch size | 128 |
| `--pit` / `--no-pit` | Permutation-invariant training | Enabled |
| `--gpu` / `--no-gpu` | GPU usage | Auto-detect |
| `--experiment-name` | Name tag for experiment folder | None |
| `--varied-noise` | Sample per-waveform noise levels | Enabled |

## Configuration

All parameters are centralized in [config.py](config.py): waveform physics, generation defaults, model architecture, training hyperparameters, and directory paths.

### Model Architecture

Both models share a common backbone:

```
Input (120,1) -> Conv1D(64, k=5) -> BatchNorm -> Conv1D(128, k=5) -> BatchNorm
             -> Flatten -> Dense(256) -> Dropout(0.3) -> Dense(128)
```

- **Count model**: Softmax(7) output, sparse categorical cross-entropy, balanced class weights
- **Signal model**: Dense(14) output (7 slots x 2 params), weighted Huber loss with t0 emphasis

## Experiment Tracking

Each `run-all` execution creates a timestamped folder in `experiments/` containing:

- `config.json` &mdash; all parameters used
- `metrics.json` &mdash; training/evaluation metrics and timing
- `pipeline.log` &mdash; full execution log
- `training_plots/` &mdash; loss curves, scalers
- `comparison_plots/` &mdash; hexbin scatter, delta histograms
- `waveform_inspection/` &mdash; individual waveform plots with predictions
- `error_analysis/` &mdash; 8 diagnostic analyses + summary PDF

TensorBoard logs are written to `tensorboard_logs/` and can be viewed with:

```bash
tensorboard --logdir tensorboard_logs/
```

## Error Analysis

The error analysis module ([error_analysis/analyze.py](error_analysis/analyze.py)) produces 8 diagnostics:

1. Per-slot MAE/RMSE breakdown
2. Error vs. signal count (boxplots)
3. Temporal error profile (t0 accuracy across time window)
4. Amplitude dependency analysis
5. Signal spacing impact
6. Worst-case waveform identification
7. Count model confusion analysis
8. Residual QQ plots

## Results

Latest experiment (50k waveforms, full training):

| Metric | Value |
|--------|-------|
| Count model accuracy | 92.2% |
| t0 MAE | 4.27 ns |
| Amplitude MAE | 1.24 |

## Project Structure

```
ML_solution/
├── cli.py                          # CLI entry point
├── config.py                       # Central configuration
├── gen_wave.py                     # Waveform generation
├── baseline_subtract.py            # Baseline removal
├── prepare_ml_dataset.py           # Dataset preparation
├── train_count_model.py            # Count classifier training
├── train_signal_model.py           # Signal regressor training
├── compare_signal_predictions.py   # Model evaluation
├── plot_individual_waveform.py     # Waveform visualization
├── generate_report.py              # LaTeX report generation
├── error_analysis/
│   └── analyze.py                  # Error diagnostics module
├── experiments/                    # Timestamped experiment results
├── reports/                        # Generated Beamer presentations
├── colab_run.ipynb                 # Google Colab notebook (T4 GPU)
└── requirements.txt
```

## Google Colab

To run on a free T4 GPU, use the provided [colab_run.ipynb](colab_run.ipynb) notebook.
