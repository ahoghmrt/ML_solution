# Central configuration for the ADC waveform signal extraction pipeline.
# Edit this file to control all pipeline parameters from one place.

# ── Waveform Physics ──────────────────────────────────────────────
TIME_START = 0          # ns
TIME_END = 120          # ns
SAMPLING_RATE = 1       # ns per sample (1 GHz)

TAU_RISE = 2            # pulse rise time constant (ns)
TAU_FALL = 10           # pulse fall time constant (ns)

AMPLITUDE_MIN = 5       # signal amplitude range
AMPLITUDE_MAX = 20

SIGNAL_COUNTS = [0, 1, 2, 3, 4, 5, 6]   # possible number of signals per waveform
NOISE_STD_CHOICES = [0.2, 0.3, 0.4, 0.5] # when noise_std is not fixed

T0_MIN = 5              # earliest signal start time (ns)
T0_MARGIN = 10          # margin from time_end for signal placement (ns)

# ── Generation Defaults ──────────────────────────────────────────
NUM_WAVEFORMS = 50000
NOISE_STD = 0.5
BASELINE = 200.0
MIN_SPACING = 0.0001    # minimum spacing between signals (ns)
MAX_SIGNALS = 7         # max signals per waveform (padding size)

# ── Baseline Subtraction ─────────────────────────────────────────
WINDOW_SIZE = 31
QUANTILE = 0.1

# ── Training ─────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

COUNT_MODEL_EPOCHS = 40
COUNT_MODEL_BATCH_SIZE = 128

SIGNAL_MODEL_EPOCHS = 30
SIGNAL_MODEL_BATCH_SIZE = 64

# ── Model Architecture ───────────────────────────────────────────
CONV_FILTERS = [32, 64]
CONV_KERNEL_SIZE = 5
DENSE_UNITS = [128, 64]
DROPOUT_RATE = 0.3

EARLY_STOPPING_PATIENCE = 6
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5
LR_MIN = 1e-6

# ── Plotting ─────────────────────────────────────────────────────
PLOT_START = 1
PLOT_END = 300

# ── Directories ──────────────────────────────────────────────────
DIR_RAW = "waveform_raw"
DIR_BASELINE_REMOVED = "waveform_baseline_removed"
DIR_ML_DATA = "ml_training_data"
DIR_TRAINING_PLOTS = "training_plots"
DIR_COMPARISON_PLOTS = "comparison_plots"
DIR_WAVEFORM_INSPECTION = "waveform_inspection"
DIR_LOGS = "logs"
DIR_EXPERIMENTS = "experiments"
