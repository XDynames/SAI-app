# In pixels per micron
CAMERA_CALIBRATION = {
    "Barley": 4.2736,
    "Arabidopsis": 10.25131,
}
# Pixels between bounding box and edge
CLOSE_TO_EDGE_DISTANCE = 20
# Percentile threshold
CLOSE_TO_EDGE_SIZE_THRESHOLD = 0.85
DENSITY_KEYS = ["image_name", "n_stomata", "density"]
DENSITY_OUTPUT_COLUMNS = ["image_name", "number_of_stomata", "density"]
IOU_THRESHOLD = 0.2
IMAGE_AREA = {
    "Barley": 0.3229496,
    "Arabidopsis": 0.04794822,
}
IS_ONLINE = False
MEASUREMENT_OUTPUT_COLUMN_NAMES = [
    "id",
    "image_name",
    "class",
    "length",
    "width",
    "area",
    "width/length",
    "confidence",
]
MEASUREMENT_KEYS = [
    "stoma_id",
    "image_name",
    "class",
    "length",
    "width",
    "area",
    "width/length",
    "confidence",
]
# In pixels
MINIMUM_LENGTH = 5
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
# Percentile threshold
SIZE_THRESHOLD = 0.3
WIDTH_OVER_LENGTH_THRESHOLD = 0.85
