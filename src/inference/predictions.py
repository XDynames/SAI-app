import copy
from typing import Dict, List, Union

import numpy as np
import shapely
from detectron2.utils.visualizer import GenericMask
from shapely.geometry import Polygon

from inference.constants import (
    NAMES_TO_CATEGORY_ID,
    MINIMUM_LENGTH,
    WIDTH_OVER_LENGTH_THRESHOLD,
    ORPHAN_AREA_THRESHOLD,
)
from inference.initial_filter import (
    filter_invalid_predictions,
    is_stomatal_pore,
)
from inference.utils import (
    calculate_bbox_height,
    calculate_bbox_width,
    calculate_midpoint_of_keypoints,
    calulate_width_over_length,
    extract_AB_from_polygon,
    find_AB,
    find_CD,
    is_stomata_complex,
    is_bbox_a_in_bbox_b,
    is_bbox_a_mostly_in_bbox_b,
    l2_dist,
)
from tools.draw import format_polygon_coordinates


class ModelOutput:
    def __init__(self, predictions, n_stoma: int):
        self._predictions = predictions
        self._formatted_predictions = []
        self.pore_lengths = []
        self.bounding_box_dimensions = []
        self._n_stoma_processed = n_stoma
        self._process_predictions()

    def _process_predictions(self):
        filter_invalid_predictions(self._predictions)
        self._format_predictions()

    @property
    def n_predictions(self) -> int:
        return len(self._formatted_predictions)

    @property
    def detections(self) -> List[Dict]:
        return self._formatted_predictions

    @property
    def _n_predictions(self) -> int:
        return len(self._predictions.pred_boxes)

    def _format_predictions(self):
        for i in range(self._n_predictions):
            if self._is_stomata_complex(i):
                self._format_prediction(i)

    def _is_stomata_complex(self, i: int) -> bool:
        return is_stomata_complex(i, self._predictions)

    def _format_prediction(self, i: int):
        prediction = {}
        self._add_complex_detction(i, prediction)
        self._add_guard_cells(i, prediction)
        self._add_pore(i, prediction)
        self._add_subsidiary_cells(i, prediction)
        self._add_pore_keypoints(i, prediction)
        self._add_guard_cell_keypoints(i, prediction)
        self._add_width_over_length(prediction)
        if prediction["width_over_length"] > WIDTH_OVER_LENGTH_THRESHOLD:
            return
        prediction["stoma_id"] = self._n_stoma_processed
        self._n_stoma_processed += 1
        self._formatted_predictions.append(prediction)
        self.pore_lengths.append(prediction["pore_length"])
        self._add_bounding_box_dimensions(prediction["bbox"])

    def _add_complex_detction(self, i: int, prediction: Dict):
        prediction["category_id"] = self._get_class(i)
        prediction["bbox"] = self._get_bounding_box(i)
        prediction["confidence"] = self._get_confidence(i)

    def _add_guard_cells(self, i: int, prediction: Dict):
        interior = []
        guard_cell_mask = self._get_mask(i)
        guard_cell_polygons = copy.deepcopy(guard_cell_mask.polygons)
        exterior_polygon = self._select_largest_polygon(guard_cell_polygons)
        shapely_exterior = Polygon(format_polygon_coordinates(exterior_polygon))
        for guard_cell_polygon in guard_cell_polygons:
            polygon = Polygon(format_polygon_coordinates(guard_cell_polygon))
            if shapely.within(polygon, shapely_exterior):
                interior = guard_cell_polygon.tolist()
        prediction.update(
            {
                "guard_cell_area": guard_cell_mask.area(),
                "guard_cell_polygon": {
                    "exterior": exterior_polygon,
                    "interior": interior,
                },
            }
        )

    def _select_largest_polygon(self, polygons: List[GenericMask]) -> GenericMask:
        tmp_polygons = [format_polygon_coordinates(polygon) for polygon in polygons]
        i_max = np.argmax([Polygon(polygon).area for polygon in tmp_polygons])
        return polygons.pop(i_max).tolist()

    def _add_pore(self, i: int, prediction: Dict):
        if self._is_open_stomata(prediction):
            pore = self._maybe_find_pore(i)
            if pore is None:
                prediction["category_id"] = NAMES_TO_CATEGORY_ID["Closed Stomata"]
        if self._is_closed_stomata(prediction):
            pore = {"pore_area": 0.0, "pore_polygon": []}
        prediction.update(pore)

    def _is_closed_stomata(self, prediction: Dict) -> bool:
        return prediction["category_id"] == NAMES_TO_CATEGORY_ID["Closed Stomata"]

    def _is_open_stomata(self, prediction: Dict) -> bool:
        return prediction["category_id"] == NAMES_TO_CATEGORY_ID["Open Stomata"]

    def _maybe_find_pore(self, i: int) -> Union[Dict, None]:
        pore_index = self._find_pore(i)
        if pore_index is not None:
            return self._format_pore_prediction(pore_index)
        return None

    def _find_pore(self, i: int) -> Union[int, None]:
        stomata_bbox = self._get_bounding_box(i)
        for j in range(self._n_predictions):
            if is_stomatal_pore(j, self._predictions):
                pore_bbox = self._get_bounding_box(j)
                if is_bbox_a_in_bbox_b(pore_bbox, stomata_bbox):
                    return j
        return None

    def _format_pore_prediction(self, i: int) -> Dict:
        mask = self._get_mask(i)
        pore = {
            "pore_area": mask.area(),
            "pore_polygon": self._select_largest_polygon(mask.polygons),
        }
        return pore

    def _add_guard_cell_keypoints(self, i: int, prediction: Dict):
        self._add_guard_cell_width_keypoints(prediction)
        self._add_guard_cell_groove_keypoints(prediction)

    def _add_guard_cell_groove_keypoints(self, prediction: Dict):
        guard_cell_polygon = prediction["guard_cell_polygon"]["exterior"]
        keypoints_AB = prediction["AB_keypoints"]
        keypoints = find_AB(guard_cell_polygon, keypoints_AB)
        prediction.update(
            {
                "guard_cell_groove_keypoints": [
                    [*keypoints[:3], *keypoints_AB[:3]],
                    [*keypoints_AB[3:], *keypoints[3:]],
                ]
            }
        )

    def _add_guard_cell_width_keypoints(self, prediction: Dict):
        guard_cell_polygon = prediction["guard_cell_polygon"]["exterior"]
        keypoints_AB = prediction["AB_keypoints"]
        keypoints = find_CD(guard_cell_polygon, keypoints_AB)
        keypoints_CD = prediction["CD_keypoints"]
        if keypoints_CD == [-1, -1, 1, -1, -1, 1]:
            midpoint = calculate_midpoint_of_keypoints(keypoints_AB)
            keypoints_CD = [*midpoint, 1, *midpoint, 1]
        prediction.update(
            {
                "guard_cell_width_keypoints": [
                    [*keypoints[:3], *keypoints_CD[:3]],
                    [*keypoints_CD[3:], *keypoints[3:]],
                ]
            }
        )

    def _add_pore_keypoints(self, i: int, prediction: Dict):
        keypoints_AB = self._get_keypoints(i)
        pore_length = l2_dist(keypoints_AB)
        if self._is_open_stomata(prediction):
            pore_polygon = prediction["pore_polygon"]
            # Sanity check for keypoint prediction
            if self._is_pore_length_extremly_small(i):
                keypoints_AB = self._extract_AB_from_polygon(pore_polygon)
                pore_length = l2_dist(keypoints_AB)

            keypoints_CD = find_CD(pore_polygon, keypoints_AB)
            # Retry using polygon keypoints
            if keypoints_CD == [-1, -1, 1, -1, -1, 1]:
                keypoints_AB = self._extract_AB_from_polygon(pore_polygon)
                keypoints_CD = find_CD(pore_polygon, keypoints_AB)
                pore_length = l2_dist(keypoints_AB)

            pore_width = l2_dist(keypoints_CD)
            width_length_ratio = calulate_width_over_length(pore_length, pore_width)
            if width_length_ratio > WIDTH_OVER_LENGTH_THRESHOLD:
                keypoints_AB = self._extract_AB_from_polygon(pore_polygon)
                keypoints_CD = find_CD(pore_polygon, keypoints_AB)
                pore_length = l2_dist(keypoints_AB)

            pore_width = l2_dist(keypoints_CD)
        else:
            keypoints_CD = [-1, -1, 1, -1, -1, 1]
            pore_width = 0

        # Stoma length is always the longest measurement
        if pore_width > pore_length:
            keypoints_AB, keypoints_CD = keypoints_CD, keypoints_AB
            pore_length, pore_width = pore_width, pore_length

        prediction.update(
            {
                "AB_keypoints": keypoints_AB,
                "CD_keypoints": keypoints_CD,
                "pore_length": pore_length,
                "pore_width": pore_width,
            }
        )

    def _is_pore_length_extremly_small(self, i) -> bool:
        keypoints_AB = self._get_keypoints(i)
        pore_length = l2_dist(keypoints_AB)
        return pore_length < MINIMUM_LENGTH

    def _extract_AB_from_polygon(self, pore_polygon: List[float]) -> List[float]:
        x_points = [x for x in pore_polygon[0::2]]
        y_points = [y for y in pore_polygon[1::2]]
        return extract_AB_from_polygon(x_points, y_points)

    def _add_subsidiary_cells(self, i: int, prediction: Dict):
        subsidiary_polygons = []
        subsidiary_area = 0.0
        for subsidiary_cell in self._get_subsidiary_cells(i):
            polygon = self._select_largest_polygon(subsidiary_cell.polygons)
            subsidiary_area += subsidiary_cell.area()
            subsidiary_polygons.append(polygon)
        # If only one cell was detected estimate total area as double
        if len(subsidiary_polygons) == 1:
            subsidiary_area *= 2
        prediction.update(
            {
                "subsidiary_cell_polygons": subsidiary_polygons,
                "subsidiary_cell_area": subsidiary_area,
            }
        )

    def _get_subsidiary_cells(self, i: int) -> List[GenericMask]:
        subsidiary_cells = []
        subsidiary_cell_indices = self._find_subsidiary_cells(i)
        if subsidiary_cell_indices is not None:
            if len(subsidiary_cell_indices) > 0:
                for index in subsidiary_cell_indices:
                    subsidiary_cells.append(self._get_mask(index))
        return subsidiary_cells

    def _find_subsidiary_cells(self, i: int) -> Union[List[int], None]:
        subsidiary_cell_indices = []
        stomata_bbox = self._get_bounding_box(i)
        for j in range(self._n_predictions):
            if self._is_subsidiary_cell(j):
                cell_bbox = self._get_bounding_box(j)
                if is_bbox_a_mostly_in_bbox_b(
                    cell_bbox,
                    stomata_bbox,
                    ORPHAN_AREA_THRESHOLD,
                ):
                    subsidiary_cell_indices.append(j)
        return subsidiary_cell_indices

    def _is_subsidiary_cell(self, i) -> bool:
        return self._get_class(i) == NAMES_TO_CATEGORY_ID["Subsidiary cells"]

    def _add_width_over_length(self, prediction: Dict):
        length, width = prediction["pore_length"], prediction["pore_width"]
        length_over_width = calulate_width_over_length(length, width)
        prediction["width_over_length"] = length_over_width

    def _add_bounding_box_dimensions(self, bbox: List[float]):
        bbox_dimensions = {
            "height": calculate_bbox_height(bbox),
            "width": calculate_bbox_width(bbox),
        }
        self.bounding_box_dimensions.append(bbox_dimensions)

    def _get_class(self, i: int) -> int:
        return self._predictions.pred_classes[i].item()

    def _get_mask(self, i: int) -> GenericMask:
        mask = self._predictions.pred_masks[i].cpu().numpy()
        return GenericMask(mask, *mask.shape)

    def _get_keypoints(self, i: int) -> List[float]:
        return self._predictions.pred_keypoints[i].flatten().tolist()

    def _get_bounding_box(self, i: int) -> List[float]:
        return self._predictions.pred_boxes[i].tensor.tolist()[0]

    def _get_confidence(self, i: int) -> float:
        return self._predictions.scores[i].item()
