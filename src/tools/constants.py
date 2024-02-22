# In pixels per micron
CAMERA_CALIBRATION = {
    "Barley": 4.2736,
    "Arabidopsis": 10.25131,
}
DENSITY_KEYS = ["image_name", "n_stomata", "density"]
DENSITY_OUTPUT_COLUMNS = [
    "image name",
    "number of stomata",
    "density",
]
IMAGE_AREA = {
    "Barley": 0.3229496,
    "Arabidopsis": 0.04794822,
}
IS_ONLINE = False
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
OPENCV_FILE_SUPPORT = [
    "png",
    "bmp",
    "jpeg",
    "jpg",
    "jpe",
    "jp2",
    "tiff",
    "tif",
]
PLANT_OPTIONS = [
    "Arabidopsis",
    "Barley",
]
