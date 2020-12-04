import matplotlib as mpl

def bboxes(mpl_axis, annotations, gt):
    key = 'category_id' if gt else 'class'
    for annotation in annotations:
        edgecolour = 'green' if annotation[key] else 'orange'
        draw_box(mpl_axis, annotation['bbox'], edgecolour, gt)
    return mpl_axis

def draw_box(mpl_axis, bbox, edgecolour, gt):
        if gt:
            x1, y1, width, height = bbox
        else:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

        mpl_axis.add_patch(
            mpl.patches.Rectangle(
                (x1, y1),
                width,
                height,
                fill=False,
                edgecolor=edgecolour,
                linewidth=1,
                alpha=0.5,
            )
        )
        return mpl_axis