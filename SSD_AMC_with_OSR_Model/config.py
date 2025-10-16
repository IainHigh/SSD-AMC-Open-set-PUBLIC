"""
config.py:
Configuration file for the SSD AMC OSR model.
All modifiable parameters are grouped here.
"""

#####################
# System Parameters
#####################
RNG_SEED = 2027

#####################
# Dataset Parameters
#####################

# Number of classes. Should match number of modulation schemes in training_set.json config file.
NUM_CLASSES = 9

SAMPLING_FREQUENCY = (
    1e9  # Sampling frequency of the dataset in Hz. Should match config file.
)

# Should match savepath in the config json files.
TRAINING_SAVE_NAME = "training"
TESTING_SAVE_NAME = "testing"

#####################
# Plotting Parameters
#####################
PRINT_CONFIG_FILE = True  # Print the configuration file to the console.
WRITE_TEST_RESULTS = False  # Write the test results to a file.

GENERATE_CONFUSION_MATRIX = True  # Generate a confusion matrix after training.

# Plot a subset of embedding features during training/testing.
PLOT_FEATURE_DISTRIBUTION = True
# Number of embeddings from each set to include in the feature plots.
NUMBER_OF_FEATURES_PLOTTED = 10000
# SNR windows used for t-SNE feature plots.
TSNE_SNR_RANGES = [(-20, -2), (0, 30), (20, 30)]
# Base perplexity for t-SNE (auto-clamped if a filtered subset is too small).
TSNE_PERPLEXITY = 80

PLOT_TEST_SAMPLES = 0  # Frequency domain representation of results - number to plot.


#####################
#  Filtering Parameters
#####################

# Window to use when filtering model outputs. Options are:
# "Rectangular", "Triangular", "Hanning", "Hamming", "Blackman", "Kaiser-Bessel".
WINDOW_CHOICE = "Hamming"
NUMTAPS = 101  # Number of taps for the filter.

#####################
# Model Parameters
#####################

S = 8  # Number of grid cells.
B = 4  # Anchors / Boxes per cell.
EMBED_DIM = 24

#####################
# Open-Set Recognition Parameters
#####################

OPENSET_ENABLE = True  # master switch for open-set recognition.
OPENSET_COVERAGE = 0.85  # tail kept inside each class Gaussian.
UNKNOWN_CLASS_NAME = "unknown"
OPENSET_THRESHOLD = None

#####################
# Training Parameters
#####################

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001  # Initial learning rate.

# Final learning rate multiplier for the learning rate scheduler.
FINAL_LR_MULTIPLE = 0.005

########################
# Loss Function Weights
########################

DETAILED_LOSS_PRINT = (
    True  # If True, will print detailed loss information during training.
)

CONFIDENCE_THRESHOLD = 0.125  # Confidence threshold for filtering predictions.

LAMBDA_NOOBJ = 1.0  # Weight for confidence loss in no-object cells.
IOU_LOSS = 4.0  # Weight for IoU loss (centre offset and bandwidths).
LAMBDA_CLASS = 2.0  # Weight for classification loss.

# Triplet loss parameters used when OPENSET_ENABLE is True.
LAMBDA_TRIPLET = 1.0  # Weight for triplet loss.
TRIPLET_MARGIN = 1.0  # Margin value for triplet loss.


def print_config_file():
    """
    Print the configuration file to the console.
    """
    print("Configuration File:")
    print("\tPLOT_TEST_SAMPLES:", PLOT_TEST_SAMPLES)
    print("\tPLOT_FEATURE_DISTRIBUTION:", PLOT_FEATURE_DISTRIBUTION)
    print("\tNUMBER_OF_FEATURES_PLOTTED:", NUMBER_OF_FEATURES_PLOTTED)
    print("\tTSNE_SNR_RANGES:", TSNE_SNR_RANGES)
    print("\tTSNE_PERPLEXITY:", TSNE_PERPLEXITY)
    print("\tWRITE_TEST_RESULTS:", WRITE_TEST_RESULTS)
    print("\tGENERATE_CONFUSION_MATRIX:", GENERATE_CONFUSION_MATRIX)
    print("\tWINDOW_CHOICE:", WINDOW_CHOICE)
    print("\tNUMTAPS:", NUMTAPS)
    print("\tS:", S)
    print("\tB:", B)
    print("\tEMBED_DIM:", EMBED_DIM)
    print("\tBATCH_SIZE:", BATCH_SIZE)
    print("\tEPOCHS:", EPOCHS)
    print("\tLEARNING_RATE:", LEARNING_RATE)
    print("\tFINAL_LR_MULTIPLE:", FINAL_LR_MULTIPLE)
    print("\tCONFIDENCE_THRESHOLD:", CONFIDENCE_THRESHOLD)
    print("")
    print("\tLOSS WEIGHT LAMBDAS:")
    print("\t\tLAMBDA_NOOBJ:", LAMBDA_NOOBJ)
    print("\t\tIOU_LOSS:", IOU_LOSS)
    print("\t\tLAMBDA_CLASS:", LAMBDA_CLASS)

    if OPENSET_ENABLE:
        print("")
        print("\tOPENSET RECOGNITION PARAMETERS:")
        print("\t\tOPENSET_COVERAGE:", OPENSET_COVERAGE)
        print("\t\tUNKNOWN_CLASS_NAME:", UNKNOWN_CLASS_NAME)
        print("\t\tLAMBDA_TRIPLET:", LAMBDA_TRIPLET)
        print("\t\tTRIPLET_MARGIN:", TRIPLET_MARGIN)
    else:
        print("")
        print("\tOPENSET RECOGNITION IS DISABLED")


#####################
# CONSTANTS - Do not modify
#####################

CELL_WIDTH = (SAMPLING_FREQUENCY / 2) / S  # width of a “bin” in Hz
UNKNOWN_IDX = NUM_CLASSES
MODULATION_CLASSES = []
