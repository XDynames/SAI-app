import os
import json
import copy

import torch
import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
import shapely
from shapely import affinity
import shapely.geometry as shapes
import matplotlib.pyplot as plt
from rasterio.transform import IDENTITY
from mask_to_polygons.vectorification import geometries_from_mask

"""
    Records predictions and ground truth pairs for an image as a JSON file
"""

IOU_THRESHOLD = 0.5

class AnnotationStore:
    def __init__(self, dir_annotations, retrieval=False):
        self.ground_truth = self.load_ground_truth(dir_annotations)
        self.filename_id_map = self.create_filename_id_map(self.ground_truth)
        self.coco = COCO(dir_annotations)
        if retrieval:
            self._pore_id_to_filename = self._create_pore_to_file_map(self.ground_truth)

    def load_ground_truth(self, gt_filepath):
        with open(gt_filepath) as file:
            raw_gt = json.load(file)
        # Associate image-id with filenames
        gt_formatted = dict()
        for image_details in raw_gt["images"]:
            image_id = image_details["id"]
            gt_formatted[image_id] = {
                "filename": image_details["file_name"],
                "annotations": list(),
            }
        # Assign instance annotations to images
        for annotation in raw_gt["annotations"]:
            image_id = annotation["image_id"]
            gt_formatted[image_id]["annotations"].append(annotation)
        return gt_formatted

    def create_filename_id_map(self, ground_truth_dict):
        file_id_map = dict()
        for image_id in ground_truth_dict.keys():
            name = ground_truth_dict[image_id]["filename"]
            file_id_map[name] = image_id
        return file_id_map

    def _create_pore_to_file_map(self, gt_dict):
        pore_to_file_map = dict()
        for image_id in gt_dict:
            image_gt = gt_dict[image_id]
            file = image_gt["filename"]
            for pore_gt in image_gt["annotations"]:
                pore_to_file_map[pore_gt["id"]] = file
        return pore_to_file_map

    def retrieve_pore(self, pore_id):
        filename = self._pore_id_to_filename[pore_id]
        image_id = self.filename_id_map[filename]
        image_annotations = self.ground_truth[image_id]["annotations"]

        for image_annotation in image_annotations:
            if image_annotation["id"] == pore_id:
                pore_annotation = image_annotation
                break
        return [filename, pore_annotation]


def record_predictions(predictions, filename, stoma_annotations):
    if stoma_annotations is None:
        print(f"# of predictions: {len(predictions.pred_boxes)}")
        predictions = convert_predictions_to_list_of_dictionaries(predictions)
        with open(".".join([filename[:-4] + "-predictions", "json"]), "w") as file:
            json.dump({"detections": predictions}, file)
        return

    ground_truth = stoma_annotations.ground_truth
    filename_id_map = stoma_annotations.filename_id_map

    coco = stoma_annotations.coco

    image_name = filename.split("/")[-1]
    image_gt = ground_truth[filename_id_map[image_name]]["annotations"]

    print(f"# of ground truth: {len(image_gt)}")
    image_gt = convert_gt_to_list_of_dictionaries(image_gt, coco)
    pairs = assign_preds_gt(predictions, image_gt)
    image_json = {"detections": pairs}
    print(f"Matched: {len(image_json['detections'])}")

    with open(".".join([filename[:-4], "json"]), "w") as file:
        json.dump(image_json, file)
    with open(".".join([filename[:-4] + "-gt", "json"]), "w") as file:
        json.dump({"detections": image_gt}, file)


def assign_preds_gt(predictions, image_gt):
    gt_prediction_pairs = []
    for i, prediction in enumerate(predictions):
        bbox = prediction["bbox"]
        gt_prediction_pairs.extend(
            [
                create_pair(predictions[i], instance_gt)
                for instance_gt in image_gt
                if gt_overlaps(bbox, instance_gt["bbox"])
            ]
        )
    return gt_prediction_pairs


def convert_gt_to_list_of_dictionaries(image_gt, coco):
    image_gt = [convert_gt_to_dictionary(instance_gt, coco) for instance_gt in image_gt]
    return image_gt


def convert_predictions_to_list_of_dictionaries(predictions):
    predictions = [
        convert_predictions_to_dictionary(i, predictions)
        for i in range(len(predictions.pred_boxes))
    ]
    return predictions


def convert_predictions_to_dictionary(i, predictions):
    # Extract and format predictions
    pred_mask = predictions.pred_masks[i].cpu().numpy()
    pred_AB = predictions.pred_keypoints[i].flatten().tolist()
    pred_class = predictions.pred_classes[i].item()
    if pred_class == 1:
        # Processes prediction mask
        pred_polygon = mask_to_poly(pred_mask)
        pred_CD = find_CD(pred_polygon, gt=False)
        pred_width = l2_dist(pred_CD)
        pred_area = pred_mask.sum().item()
    else:
        pred_polygon = []
        pred_CD = [-1, -1, 1.0 - 1, -1, 1]
        pred_width = 0
        pred_area = 0

    prediction_dict = {
        "bbox": predictions.pred_boxes[i].tensor.tolist()[0],
        "area": pred_area,
        "AB_keypoints": pred_AB,
        "length": l2_dist(pred_AB),
        "CD_keypoints": pred_CD,
        "width": pred_width,
        "category_id": pred_class,
        "segmentation": [pred_polygon],
        "confidence": predictions.scores[i].item(),
    }
    return prediction_dict


def convert_gt_to_dictionary(instance_gt, coco):
    if instance_gt["category_id"] == 1:
        # Process ground truth mask
        gt_CD = find_CD(instance_gt["segmentation"][0], instance_gt["keypoints"])
        gt_width = l2_dist(gt_CD)
        instance_gt["area"] = calc_area(instance_gt, coco)
    else:
        # Placehold values for no prediction
        gt_width = 0
        gt_CD = [-1, -1, 1, -1, -1, 1]
        instance_gt["area"] = 0
    # Add additional keys to ground truth
    instance_gt["width"] = gt_width
    instance_gt["CD_keypoints"] = gt_CD
    instance_gt["length"] = l2_dist(instance_gt["keypoints"])
    return instance_gt


def create_pair(predictions, instance_gt):
    detection_pair = {"gt": instance_gt, "pred": predictions}
    return detection_pair


def find_CD(polygon, keypoints=None, gt=True):
    # If no mask is predicted
    if len(polygon) < 1:
        #    counter += 1
        return [-1, -1, 1, -1, -1, 1]

    x_points = [x for x in polygon[0::2]]
    y_points = [y for y in polygon[1::2]]

    if keypoints == None:
        keypoints = extract_polygon_AB(x_points, y_points)
    # Convert to shapely linear ring
    polygon = [[x, y] for x, y in zip(x_points, y_points)]
    mask = shapes.LinearRing(polygon)
    # Find line perpendicular to AB
    A = shapes.Point(keypoints[0], keypoints[1])
    B = shapes.Point(keypoints[3], keypoints[4])

    l_AB = shapes.LineString([A, B])
    l_perp = affinity.rotate(l_AB, 90)
    l_perp = affinity.scale(l_perp, 10, 10)
    # Find intersection with polygon
    try:
        intersections = l_perp.intersection(mask)
    except:
        intersections = shapes.collection.GeometryCollection()

    """
    # Visual check
    plt.plot(*mask.xy)
    plt.plot(*l_AB.xy)
    plt.plot(*l_perp.xy)
    if counter % 2 == 0:
        plt.savefig('visualised_plots/out-'+str(int(counter/2))+'.png')
        plt.clf()
    elif gt:
        plt.clf()
    counter += 1
    """
    # If there is no intersection
    if intersections.is_empty:
        return [-1, -1, 1, -1, -1, 1]

    if intersections[0].coords.xy[1] > intersections[1].coords.xy[1]:
        D = intersections[0].coords.xy
        C = intersections[1].coords.xy
    else:
        D = intersections[1].coords.xy
        C = intersections[0].coords.xy
    return [*C[0], *C[1], 1, *D[0], *D[1], 1]


def l2_dist(keypoints):
    A, B = [keypoints[0], keypoints[1]], [keypoints[3], keypoints[4]]
    return pow((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2, 0.5)


def mask_to_poly(mask):
    if mask.sum() == 0:
        return []
    poly = geometries_from_mask(np.uint8(mask), IDENTITY, "polygons")
    poly = poly[0]["coordinates"][0]
    flat_poly = []
    for point in poly:
        flat_poly.extend([point[0], point[1]])
    return flat_poly


# Convert polygon -> binary mask COCO.annToMask
#  Count pixels as area
def calc_area(gt_annotation, coco):
    gt_mask = torch.Tensor(coco.annToMask(gt_annotation)) == 1
    gt_area = gt_mask.sum()
    return gt_area.item()


def intersects(bbox_1, bbox_2):
    is_overlap = not (
        bbox_2[0] > bbox_1[2]
        or bbox_2[2] < bbox_1[0]
        or bbox_2[1] > bbox_1[3]
        or bbox_2[3] < bbox_1[1]
    )
    return is_overlap


def is_overlapping(bbox1, bbox2):
    if intersects(bbox1, bbox2):
        return overlaps(bbox1, bbox2)
    return False


def overlaps(bbox_1, bbox_2):
    iou = intersection_over_union(bbox_1, bbox_2)
    if iou > IOU_THRESHOLD:
        return True
    return False


def gt_overlaps(bbox, gt_bbox):
    # GT boxes [x, y, width, height] -> [x1, y1, x2, y2]
    gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]

    return is_overlapping(bbox, gt_bbox)


def intersection_over_union(bbox, gt_bbox):
    x_max, y_max = max(bbox[0], gt_bbox[0]), max(bbox[1], gt_bbox[1])
    x_min, y_min = min(bbox[2], gt_bbox[2]), min(bbox[3], gt_bbox[3])
    intersecting_area = max(0, x_min - x_max + 1) * max(0, y_min - y_max + 1)

    pred_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    gt_area = (gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1)
    iou = intersecting_area / float(pred_area + gt_area - intersecting_area)
    return iou


def extract_polygon_AB(x_values, y_values):
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_extent = x_max - x_min
    y_extent = y_max - y_min
    # Enables pores of arbitrary orientation
    if x_extent > y_extent:
        major_axis_values = x_values
        minor_axis_values = y_values
        maximum_major_value = x_max
        minimum_major_value = x_min
    else:
        major_axis_values = y_values
        minor_axis_values = x_values
        maximum_major_value = y_max
        minimum_major_value = y_min
    # Left/Right along major axis
    left_hand_values, right_hand_values = [], []
    for i, minor_value in enumerate(minor_axis_values):
        if maximum_major_value == major_axis_values[i]:
            right_hand_values.append(minor_value)
        if minimum_major_value == major_axis_values[i]:
            left_hand_values.append(minor_value)
    # Use midpoint of extreme values as keypoint value
    right_hand_value = (right_hand_values[0] + right_hand_values[-1]) / 2
    left_hand_value = (left_hand_values[0] + left_hand_values[-1]) / 2

    if x_extent > y_extent:
        keypoints = [
            minimum_major_value,
            left_hand_value,
            1,
            maximum_major_value,
            right_hand_value,
            1,
        ]
    else:
        keypoints = [
            left_hand_value,
            minimum_major_value,
            1,
            right_hand_value,
            maximum_major_value,
            1,
        ]
    return keypoints
