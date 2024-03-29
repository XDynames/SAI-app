import matplotlib as mpl


def masks(mpl_axis, annotations, gt):
    colour = "red" if gt else "blue"
    for annotation in annotations:
        draw_masks(mpl_axis, annotation, colour)


def keypoints(mpl_axis, annotations, gt):
    for annotation in annotations:
        draw_keypoints(mpl_axis, annotation, gt)


def bboxes(mpl_axis, annotations, gt):
    for annotation in annotations:
        edgecolour = "green" if annotation["category_id"] else "orange"
        draw_bbox(mpl_axis, annotation, edgecolour, gt)


def legend(mpl_axis, human_flag=True):
    mpl_axis.set(ylim=[mpl_axis.get_ylim()[0] + 300, 0])
    proxy_handles, labels = [], []
    if human_flag:
        proxy_handles.append(mpl.patches.Patch(color="red"))
        labels.append("Human Measurements")
    proxy_handles.extend(
        [
            mpl.patches.Patch(color="blue"),
            mpl.patches.Patch(color="green"),
            mpl.patches.Patch(color="orange"),
        ]
    )
    labels.extend(["Model Estimates", "Open Pores", "Closed Pores"])
    num_cols = 2 if human_flag else 3
    mpl_axis.legend(
        proxy_handles,
        labels,
        loc="lower center",
        frameon=False,
        fontsize=8,
        ncol=num_cols,
    )


def draw_bbox(mpl_axis, annotation, colour, gt):
    label = f"{annotation['stoma_id']}: " if "stoma_id" in annotation else ""
    label += "Open" if annotation["category_id"] else "Closed"
    if gt:
        x1, y1, width, height = annotation["bbox"]
        text_position = (x1 + width, y1)
        alignment = "right"
        text = f"{label}"
    else:
        x1, y1, x2, y2 = annotation["bbox"]
        width = x2 - x1
        height = y2 - y1
        text_position = (x1, y1)
        alignment = "left"
        confidence = round(annotation["confidence"] * 100, 1)
        text = f"{label} {confidence}%"

    draw_box(mpl_axis, (x1, y1), width, height, colour)
    draw_label(mpl_axis, text_position, text, alignment)


def draw_masks(mpl_axis, annotation, colour):
    segment = annotation["segmentation"][0]
    if not segment:
        return
    segment = format_polygon_coordinates(segment)
    mpl_axis.add_patch(
        mpl.patches.Polygon(
            segment,
            fill=True,
            alpha=0.5,
            facecolor=colour,
            edgecolor=colour,
            linewidth=0.5,
        )
    )


def format_polygon_coordinates(polygon):
    indices = range(0, len(polygon), 2)
    return [[polygon[i], polygon[i + 1]] for i in indices]


def draw_keypoints(mpl_axis, annotation, gt):
    draw_length_keypoints(mpl_axis, annotation, gt)
    draw_width_keypoints(mpl_axis, annotation, gt)


def draw_width_keypoints(mpl_axis, annotation, gt):
    keypoints_width = annotation["CD_keypoints"]
    # Dummy value for closed stoma -> no width
    if keypoints_width[0] == -1:
        return
    keypoints_x = extract_x_keypoints(keypoints_width)
    keypoints_y = extract_y_keypoints(keypoints_width)
    draw_points_and_lines(mpl_axis, keypoints_x, keypoints_y, gt)


def draw_length_keypoints(mpl_axis, annotation, gt):
    key = "keypoints" if gt else "AB_keypoints"
    keypoints_length = annotation[key]
    keypoints_x = extract_x_keypoints(keypoints_length)
    keypoints_y = extract_y_keypoints(keypoints_length)
    draw_points_and_lines(mpl_axis, keypoints_x, keypoints_y, gt)


def extract_x_keypoints(keypoints):
    return [keypoints[0], keypoints[3]]


def extract_y_keypoints(keypoints):
    return [keypoints[1], keypoints[4]]


def draw_points_and_lines(mpl_axis, keypoints_x, keypoints_y, gt):
    colour = "red" if gt else "blue"
    draw_lines(mpl_axis, [keypoints_x, keypoints_y], colour)
    draw_points(mpl_axis, zip(keypoints_x, keypoints_y), colour)


def draw_points(mpl_axis, keypoints, colour, radius=2):
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
            linewidth=1,
            alpha=0.5,
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
            "alpha": 0.8,
            "pad": 0.7,
            "edgecolor": "none",
        },
        verticalalignment="top",
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
