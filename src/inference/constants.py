CLOSE_TO_EDGE_DISTANCE = 20  # In pixels
CLOSE_TO_EDGE_SIZE_THRESHOLD = 0.85
DENSITY_KEYS = ["image_name", "n_stomata", "density", "g_max"]
DENSITY_OUTPUT_COLUMNS = ["image name", "number of stomata", "density", "g max"]
DIFFUSIVITY_OF_WATER_IN_AIR_25C = 0.0282
IOU_THRESHOLD = 0.5
MEASUREMENT_OUTPUT_COLUMN_NAMES = [
    "id",
    "image name",
    "class",
    "pore length",
    "pore width",
    "pore area",
    "pore width/length",
    "subsidiary cell area",
    "guard cell area",
    "confidence",
]
MEASUREMENT_KEYS = [
    "stoma_id",
    "image_name",
    "class",
    "pore_length",
    "pore_width",
    "pore_area",
    "pore_width_to_length_ratio",
    "subsidiary_cell_area",
    "guard_cell_area",
    "confidence",
]
MINIMUM_LENGTH = 5  # In pixels
MOLAR_VOLUME_OF_WATER_IN_AIR_25C = 0.02241
NAMES_TO_CATEGORY_ID = {
    "Closed Stomata": 0,
    "Open Stomata": 1,
    "Stomatal Pore": 2,
    "Subsidiary cells": 3,
}
ORPHAN_AREA_THRESHOLD = 0.5
SIZE_THRESHOLD = 0.3
WIDTH_OVER_LENGTH_THRESHOLD = 0.85
