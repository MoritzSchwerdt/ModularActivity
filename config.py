# Configs defined
from typing import Dict

DATASET_PATH = "../../data"
CHECKPOINT_PATH = "../../saved_models/stl10_simclr"

BATCH_SIZE = 256
NUM_EPOCHS = 500
LR = 5e-4
HIDDEN_DIM = 128
TEMPERATURE = 0.07
WEIGHT_DECAY = 1e-4




# estimate of the size of the retina sensitive to central eight degrees, given an
# estimate of roughly 300 microns per visual degree
RETINA_SIZE = 2.4  # mm
# estimate of how big, in mm, V1 in a single hemisphere is, from Benson et al
#  (sqrt 13.5cm^2 = 36.75mm)
V1_SIZE = 36.75  # mm
# V2 estimate also from Benson et al, slightly smaller than V1
V2_SIZE = 35.0  # mm
# human V4 estimates are hard to find, but this seems like a reasonable approximation
V4_SIZE = 22.4  # mm
# this is measured from the average size of the responsive VTC ROI in the NSD
VTC_SIZE = 70.0  # mm

brain_mapping = {
        "block1_0": "retina",
        "block1_1": "retina",
        "block2_0": "V1",
        "block2_1": "V1",
        "layer2.2": "V1",
        "block3_0": "V2",
        "block3_1": "V2",
        "block4_0": "V4",
        "block4_1": "V4"
    }

TISSUE_SIZES: Dict[str, float] = {
    "retina": RETINA_SIZE,
    "V1": V1_SIZE,
    "V2": V2_SIZE,
    "V4": V4_SIZE,
    "VTC": VTC_SIZE,
}

NEIGHBORHOOD_WIDTHS: Dict[str, float] = {
    "retina": 0.0475,
    "V1": 1.626,
    "V2": 3.977,
    "V4": 2.545,
    "VTC": 31.818,
}

RF_OVERLAP = 0.2
N_NEIGHBORHOODS = 500
#n_neighborhoods = 500
N_NEIGHBORS = 200
#n_neighbors = 200
