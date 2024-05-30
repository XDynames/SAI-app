from typing import Dict, List, Tuple, Union

import streamlit as st
from shapely.geometry import MultiPolygon, Polygon

from app.annotation_retrieval import get_ground_truth
from inference.constants import NAMES_TO_CATEGORY_ID, ORPHAN_AREA_THRESHOLD
from inference.utils import (
    calculate_midpoint_of_keypoints,
    convert_measurements,
    find_AB,
    find_CD,
    is_bbox_a_in_bbox_b,
    is_bbox_a_mostly_in_bbox_b,
    l2_dist,
)
from tools.draw import format_polygon_coordinates


def retrieve():
    raw_ground_truth = get_ground_truth()
    ground_truth = process_ground_truth(raw_ground_truth)
    return ground_truth


def process_ground_truth(raw_ground_truth: List[Dict]) -> List[Dict]:
    format_bboxes(raw_ground_truth)
    complexes, structures = separate_annotations(raw_ground_truth)
    annotations = format_annotations(complexes, structures)
    convert_measurements(annotations)
    return annotations


def format_bboxes(raw_ground_truth: List[Dict]):
    for annotation in raw_ground_truth:
        if "bbox" in annotation:
            xyhw_bbox = annotation["bbox"]
            annotation["bbox"] = convert_bbox_xyhw_to_xyxy(xyhw_bbox)


def convert_bbox_xyhw_to_xyxy(xywh_bbox: List[float]) -> List[float]:
    x1, y1, w, h = xywh_bbox
    return [x1, y1, x1 + w, y1 + h]


def separate_annotations(raw_ground_truth: List[Dict]) -> Tuple:
    complexes, structures = [], []
    for annotation in raw_ground_truth:
        if is_stomata_complex(annotation):
            complexes.append(annotation)
        else:
            structures.append(annotation)
    return complexes, structures


def format_annotations(complexes: List[Dict], structures: List[Dict]) -> List[Dict]:
    formatted_annotation = []
    for complex_annotation in complexes:
        annotation = format_annotation(complex_annotation, structures)
        formatted_annotation.append(annotation)
    return formatted_annotation


def is_stomata_complex(annotation: Dict) -> bool:
    class_label = annotation["category_id"]
    complex_categories = [
        NAMES_TO_CATEGORY_ID["Closed Stomata"],
        NAMES_TO_CATEGORY_ID["Open Stomata"],
    ]
    return class_label in complex_categories


def format_annotation(annotation: Dict, structures: List[Dict]) -> Dict:
    formatted = {}
    add_guard_cells(annotation, formatted)
    maybe_add_subsidiary_cells(annotation, formatted, structures)
    maybe_add_stomata_pore(annotation, formatted, structures)
    add_pore_keypoints(annotation, formatted)
    add_guard_cell_keypoints(formatted)
    return formatted


def add_guard_cells(annotation: Dict, formatted: Dict):
    guard_cell_polygon = get_guard_cell_polygon(annotation["segmentation"])
    guard_cell = {
        "bbox": annotation["bbox"],
        "category_id": annotation["category_id"],
        "guard_cell_area": guard_cell_polygon.area,
        "guard_cell_polygons": annotation["segmentation"],
    }
    formatted.update(guard_cell)


def get_guard_cell_polygon(polygons: List) -> MultiPolygon:
    guard_cell_polygon = MultiPolygon(
        [
            Polygon(format_polygon_coordinates(polygons[0])),
            Polygon(format_polygon_coordinates(polygons[1])),
        ]
    )
    return guard_cell_polygon


def flatten_shapely_coords(coords: List) -> List[float]:
    flat_coords = []
    for coord in coords:
        flat_coords.extend(coord)
    return flat_coords


def maybe_add_subsidiary_cells(
    annotation: Dict,
    formatted: Dict,
    structures: List[Dict],
):
    subsidiary_area, subsidiary_cell_polygons = 0.0, []
    subsidiary_cells = find_subsidiary_cells(annotation, structures)

    for subsidiary_cell in subsidiary_cells:
        subsidiary_cell_polygon = subsidiary_cell["segmentation"][0]
        subsidiary_area += get_polygon_area(subsidiary_cell_polygon)
        subsidiary_cell_polygons.append(subsidiary_cell_polygon)
    subsidiary_cells = {
        "subsidiary_cell_polygons": subsidiary_cell_polygons,
        "subsidiary_cell_area": subsidiary_area,
    }
    formatted.update(subsidiary_cells)


def find_subsidiary_cells(complex_annotation: Dict, structures: List[Dict]) -> List:
    subsidiary_cells = []
    stomata_bbox = complex_annotation["bbox"]
    for structure in structures:
        if is_subsidiary_cell(structure):
            cell_bbox = structure["bbox"]
            if is_bbox_a_mostly_in_bbox_b(
                cell_bbox,
                stomata_bbox,
                ORPHAN_AREA_THRESHOLD,
            ):
                subsidiary_cells.append(structure)
    return subsidiary_cells


def is_subsidiary_cell(annotation: Dict) -> bool:
    class_label = annotation["category_id"]
    return class_label == NAMES_TO_CATEGORY_ID["Subsidiary cells"]


def maybe_add_stomata_pore(
    complex_annotation: Dict,
    formatted: Dict,
    structures: List[Dict],
):
    if is_closed_stomata(complex_annotation):
        pore = {"pore_area": 0.0, "pore_polygon": []}
    else:
        pore = find_stomata_pore(complex_annotation, structures)
        if pore is None:
            pore = {"pore_area": 0.0, "pore_polygon": []}
        else:
            polygon = pore["segmentation"][0]
            pore = {"pore_area": get_polygon_area(polygon), "pore_polygon": polygon}
    formatted.update(pore)


def get_polygon_area(polygon: List[float]) -> float:
    return Polygon(format_polygon_coordinates(polygon)).area


def is_closed_stomata(annotation: Dict) -> bool:
    class_label = annotation["category_id"]
    return class_label == NAMES_TO_CATEGORY_ID["Closed Stomata"]


def find_stomata_pore(
    complex_annotation: Dict, structures: List[Dict]
) -> Union[Dict, None]:
    stomata_bbox = complex_annotation["bbox"]
    for structure in structures:
        if is_stomata_pore(structure):
            pore_bbox = structure["bbox"]
            if is_bbox_a_in_bbox_b(pore_bbox, stomata_bbox):
                return structure


def is_stomata_pore(annotation: Dict) -> bool:
    class_label = annotation["category_id"]
    return class_label == NAMES_TO_CATEGORY_ID["Stomatal Pore"]


def add_pore_keypoints(annotation: Dict, formatted: Dict):
    keypoints_AB = annotation["keypoints"]
    pore_length = l2_dist(keypoints_AB)
    if is_closed_stomata(annotation):
        keypoints_CD = [-1, -1, 1, -1, -1, 1]
        pore_width = 0.0
    else:
        pore_polygon = formatted["pore_polygon"]
        keypoints_CD = find_CD(pore_polygon, keypoints_AB)
        pore_width = l2_dist(keypoints_CD)
    pore_keypoints = {
        "AB_keypoints": keypoints_AB,
        "CD_keypoints": keypoints_CD,
        "pore_length": pore_length,
        "pore_width": pore_width,
    }
    formatted.update(pore_keypoints)


def add_guard_cell_keypoints(formatted: Dict):
    add_guard_cell_width_keypoints(formatted)
    add_guard_cell_groove_keypoints(formatted)


def add_guard_cell_width_keypoints(formatted: Dict):
    guard_cell_polygon = get_guard_cell_polygon(formatted["guard_cell_polygons"])
    guard_cell_polygon = guard_cell_polygon.convex_hull.exterior.coords
    guard_cell_polygon = flatten_shapely_coords(guard_cell_polygon)
    keypoints_AB = formatted["AB_keypoints"]
    keypoints = find_CD(guard_cell_polygon, keypoints_AB)
    keypoints_CD = formatted["CD_keypoints"]
    if keypoints_CD == [-1, -1, 1, -1, -1, 1]:
        midpoint = calculate_midpoint_of_keypoints(keypoints_AB)
        keypoints_CD = [*midpoint, 1, *midpoint, 1]
    keypoint_1 = [*keypoints[:3], *keypoints_CD[:3]]
    keypoint_2 = [*keypoints_CD[3:], *keypoints[3:]]
    width = (l2_dist(keypoint_1) + l2_dist(keypoint_2)) / 2
    guard_cell_width = {
        "guard_cell_width_keypoints": [keypoint_1, keypoint_2],
        "guard_cell_width": width,
    }
    formatted.update(guard_cell_width)


def add_guard_cell_groove_keypoints(formatted: Dict):
    guard_cell_polygon = get_guard_cell_polygon(formatted["guard_cell_polygons"])
    guard_cell_polygon = guard_cell_polygon.convex_hull.exterior.coords
    guard_cell_polygon = flatten_shapely_coords(guard_cell_polygon)
    keypoints_AB = formatted["AB_keypoints"]
    keypoints = find_AB(guard_cell_polygon, keypoints_AB)
    keypoint_1 = [*keypoints[:3], *keypoints_AB[:3]]
    keypoint_2 = [*keypoints_AB[3:], *keypoints[3:]]
    length = (l2_dist(keypoint_1) + l2_dist(keypoint_2)) / 2
    guard_cell_grooves = {
        "guard_cell_groove_keypoints": [keypoint_1, keypoint_2],
        "guard_cell_groove_length": length,
    }
    formatted.update(guard_cell_grooves)
