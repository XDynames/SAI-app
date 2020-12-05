import matplotlib as mpl

def masks(mpl_axis, annotations, gt):
    pass

def keypoints(mpl_axis, annotations, gt):
    pass

def bboxes(mpl_axis, annotations, gt):
    key = 'category_id' if gt else 'class'
    for annotation in annotations:
        edgecolour = 'green' if annotation[key] else 'orange'
        draw_bbox(mpl_axis, annotation, edgecolour, gt)

def draw_bbox(mpl_axis, annotation, colour, gt):
        if gt:
            x1, y1, width, height = annotation['bbox']
            text_position = (x1+width,y1) 
            alignment = 'right'
            label = 'Open' if annotation['category_id'] else 'Closed'
            text = f'{label}'
        else:
            x1, y1, x2, y2 = annotation['bbox']
            width = x2 - x1
            height = y2 - y1
            text_position = (x1,y1)
            alignment = 'left'
            label = 'Open' if annotation['class'] else 'Closed'
            confidence = round(annotation['confidence'] * 100, 1)
            text = f'{label} {confidence}%'

        draw_box(mpl_axis, (x1,y1), width, height, colour)
        draw_label(mpl_axis, text_position, text, alignment)

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
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            color='white',
            horizontalalignment=horizontal_alignment
        )