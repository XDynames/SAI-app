import matplotlib as mpl
import streamlit as st
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon


def masks(mpl_axis, annotations, gt):
    for annotation in annotations:
        if "guard_cell_polygon" in annotation:
            if "exterior" in annotation["guard_cell_polygon"]:
                exterior = annotation["guard_cell_polygon"]["exterior"]
                interior = annotation["guard_cell_polygon"]["interior"]
                draw_mask_with_holes(mpl_axis, exterior, interior, "xkcd:orange")

        if "guard_cell_polygons" in annotation:
            polygons = annotation["guard_cell_polygons"]
            draw_mask(mpl_axis, polygons[0], "xkcd:orange")
            draw_mask(mpl_axis, polygons[1], "xkcd:orange")

        if annotation["pore_polygon"]:
            polygon = annotation["pore_polygon"]
            draw_mask(mpl_axis, polygon, "xkcd:teal")

        if annotation["subsidiary_cell_polygons"]:
            for polygon in annotation["subsidiary_cell_polygons"]:
                draw_mask(mpl_axis, polygon, "xkcd:fuchsia")


def keypoints(mpl_axis, annotations, gt):
    for annotation in annotations:
        draw_keypoints(mpl_axis, annotation, gt)


def bboxes(mpl_axis, annotations, gt):
    for annotation in annotations:
        edgecolour = "xkcd:green" if annotation["category_id"] else "xkcd:purple"
        draw_bbox(mpl_axis, annotation, edgecolour, gt)


def legend(mpl_axis):
    mpl_axis.set(ylim=[mpl_axis.get_ylim()[0] + 350, 0])
    proxy_handles, labels = [], []
    proxy_handles.extend(
        [
            mpl.patches.Patch(color="xkcd:green"),
            mpl.patches.Patch(color="xkcd:purple"),
            mpl.patches.Patch(color="xkcd:blue"),
            mpl.patches.Patch(color="xkcd:red"),
            mpl.patches.Patch(color="xkcd:teal"),
            mpl.patches.Patch(color="xkcd:orange"),
            mpl.patches.Patch(color="xkcd:fuchsia"),
        ]
    )
    labels.extend(
        [
            "Open Pores",
            "Closed Pores",
            "Pore Keypoints",
            "Guard Cell Keypoints",
            "Pore",
            "Guard Cells",
            "Subsidiary Cells",
        ]
    )
    mpl_axis.legend(
        proxy_handles,
        labels,
        loc="lower center",
        frameon=False,
        fontsize=7,
        ncol=3,
    )


def draw_bbox(mpl_axis, annotation, colour, gt):
    label = f"{annotation['stoma_id']}: " if "stoma_id" in annotation else ""
    label += "Open" if annotation["category_id"] else "Closed"
    x1, y1, x2, y2 = annotation["bbox"]
    width = x2 - x1
    height = y2 - y1
    if gt:
        text_position = (x1 + width, y1 + height)
        alignment = "right"
        text = f"{label}"
    else:
        text_position = (x1, y1)
        alignment = "left"
        confidence = round(annotation["confidence"] * 100, 0)
        text = f"{label} {confidence}%"

    draw_box(mpl_axis, (x1, y1), width, height, colour)
    draw_label(mpl_axis, text_position, text, alignment)


def draw_mask(mpl_axis, polygon, colour):
    polygon = format_polygon_coordinates(polygon)
    mpl_axis.add_patch(
        mpl.patches.Polygon(
            polygon,
            fill=True,
            alpha=0.5,
            facecolor=colour,
            edgecolor=colour,
            linewidth=0.0,
        )
    )
    mpl_axis.add_patch(
        mpl.patches.Polygon(
            polygon,
            fill=False,
            alpha=1.0,
            edgecolor=colour,
            linewidth=0.5,
        )
    )


def format_polygon_coordinates(polygon):
    indices = range(0, len(polygon), 2)
    return [[polygon[i], polygon[i + 1]] for i in indices]


def draw_keypoints(mpl_axis, annotation, gt):
    draw_guard_cell_width_keypoints(mpl_axis, annotation, gt)
    draw_guard_cell_groove_keypoints(mpl_axis, annotation, gt)
    draw_pore_length_keypoints(mpl_axis, annotation, gt)
    draw_pore_width_keypoints(mpl_axis, annotation, gt)


def draw_pore_width_keypoints(mpl_axis, annotation, gt):
    colour = "xkcd:blue"
    keypoints_width = annotation["CD_keypoints"]
    # Dummy value for closed stoma -> no width
    if keypoints_width[0] == -1:
        return
    keypoints_x = extract_x_keypoints(keypoints_width)
    keypoints_y = extract_y_keypoints(keypoints_width)
    draw_points_and_lines(mpl_axis, keypoints_x, keypoints_y, colour)


def draw_guard_cell_groove_keypoints(mpl_axis, annotation, gt):
    colour = "xkcd:red"
    for keypoints_groove in annotation["guard_cell_groove_keypoints"]:
        keypoints_x = extract_x_keypoints(keypoints_groove)
        keypoints_y = extract_y_keypoints(keypoints_groove)
        draw_points_and_lines(mpl_axis, keypoints_x, keypoints_y, colour)


def draw_pore_length_keypoints(mpl_axis, annotation, gt):
    colour = "xkcd:blue"
    keypoints_length = annotation["AB_keypoints"]
    keypoints_x = extract_x_keypoints(keypoints_length)
    keypoints_y = extract_y_keypoints(keypoints_length)
    draw_points_and_lines(mpl_axis, keypoints_x, keypoints_y, colour)


def draw_guard_cell_width_keypoints(mpl_axis, annotation, gt):
    colour = "xkcd:red"
    for keypoints_width in annotation["guard_cell_width_keypoints"]:
        keypoints_x = extract_x_keypoints(keypoints_width)
        keypoints_y = extract_y_keypoints(keypoints_width)
        draw_points_and_lines(mpl_axis, keypoints_x, keypoints_y, colour)


def extract_x_keypoints(keypoints):
    return [keypoints[0], keypoints[3]]


def extract_y_keypoints(keypoints):
    return [keypoints[1], keypoints[4]]


def draw_points_and_lines(mpl_axis, keypoints_x, keypoints_y, colour):
    draw_lines(mpl_axis, [keypoints_x, keypoints_y], colour)
    draw_points(mpl_axis, zip(keypoints_x, keypoints_y), colour)


def draw_points(mpl_axis, keypoints, colour, radius=1.0):
    for keypoint in keypoints:
        mpl_axis.add_patch(
            mpl.patches.Circle(
                keypoint,
                radius=radius,
                fill=True,
                color=colour,
            )
        )


def draw_lines(mpl_axis, keypoints, colour):
    mpl_axis.add_line(
        mpl.lines.Line2D(
            keypoints[0],
            keypoints[1],
            linewidth=0.5,
            color=colour,
        )
    )


def draw_box(mpl_axis, position, width, height, edgecolour):
    mpl_axis.add_patch(
        mpl.patches.Rectangle(
            position,
            width,
            height,
            fill=False,
            edgecolor=edgecolour,
            linewidth=0.75,
            alpha=1,
        )
    )


def draw_label(mpl_axis, position, text, horizontal_alignment):
    mpl_axis.text(
        position[0],
        position[1],
        text,
        size=2.5,
        family="sans-serif",
        bbox={
            "facecolor": "black",
            "alpha": 1.0,
            "pad": 0.3,
            "edgecolor": "none",
        },
        verticalalignment="bottom",
        color="white",
        horizontalalignment=horizontal_alignment,
    )


def draw_legend_text(mpl_axis, position, text):
    mpl_axis.text(
        position[0],
        position[1],
        text,
        size=10,
        family="sans-serif",
        bbox={
            "facecolor": "white",
            "alpha": 1,
            "pad": 0.7,
            "edgecolor": "none",
        },
        verticalalignment="top",
        color="black",
        horizontalalignment="left",
    )


def draw_mask_with_holes(mpl_axis, exterior, interior, colour):
    exterior_polygon = format_polygon_coordinates(exterior)
    if interior:
        interior_polygon = [format_polygon_coordinates(interior)]
    else:
        interior_polygon = None
    polygon = Polygon(exterior_polygon, holes=interior_polygon)
    plot_polygon(
        polygon,
        ax=mpl_axis,
        add_points=False,
        facecolor=colour,
        alpha=0.5,
        linewidth=0.0,
    )
    mpl_axis.add_patch(
        mpl.patches.Polygon(
            exterior_polygon,
            fill=False,
            alpha=1.0,
            edgecolor=colour,
            linewidth=0.5,
        )
    )
    if interior_polygon is not None:
        mpl_axis.add_patch(
            mpl.patches.Polygon(
                interior_polygon[0],
                fill=False,
                alpha=1.0,
                edgecolor=colour,
                linewidth=0.5,
            )
        )
