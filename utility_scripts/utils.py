import numpy as np


def crop_padding(img, roi, pad_value):
    """
    img: HxW or HxWxC np.ndarray
    roi: (x,y,w,h)
    pad_value: [b,g,r]
    """
    need_squeeze = False
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        need_squeeze = True
    assert len(pad_value) == img.shape[2]
    x, y, w, h = roi
    x, y, w, h = int(x), int(y), int(w), int(h)
    H, W = img.shape[:2]
    output = np.tile(np.array(pad_value), (h, w, 1)).astype(img.dtype)
    if bbox_iou((x, y, x + w, y + h), (0, 0, W, H)) > 0:
        output[max(-y, 0) : min(H - y, h), max(-x, 0) : min(W - x, w), :] = img[
            max(y, 0) : min(y + h, H), max(x, 0) : min(x + w, W), :
        ]
    if need_squeeze:
        output = np.squeeze(output)
    return output


def bbox_iou(b1, b2):
    """
    b: (x1,y1,x2,y2)
    """
    lx = max(b1[0], b2[0])
    rx = min(b1[2], b2[2])
    uy = max(b1[1], b2[1])
    dy = min(b1[3], b2[3])
    if rx <= lx or dy <= uy:
        return 0.0
    else:
        interArea = (rx - lx) * (dy - uy)
        a1 = float((b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = float((b2[2] - b2[0]) * (b2[3] - b2[1]))
        return interArea / (a1 + a2 - interArea)


class EraserSetter(object):
    def __init__(self):
        self.min_overlap = 0.4
        self.max_overlap = 0.9
        self.min_cut_ratio = 0.001
        self.max_cut_ratio = 0.7

    def __call__(self, mask, eraser):
        return place_eraser_in_ratio(
            mask,
            eraser,
            self.min_overlap,
            self.max_overlap,
            self.min_cut_ratio,
            self.max_cut_ratio,
            100,
        )


def place_eraser_in_ratio(
    mask,
    eraser,
    min_overlap,
    max_overlap,
    min_ratio,
    max_ratio,
    max_iter,
):
    for i in range(max_iter):
        shift_eraser, ratio = place_eraser(mask, eraser, min_overlap, max_overlap)
        if ratio >= min_ratio and ratio < max_ratio:
            break
    return shift_eraser, ratio


def place_eraser(mask, eraser, min_overlap, max_overlap):
    assert len(mask.shape) == 2
    assert len(eraser.shape) == 2
    assert min_overlap <= max_overlap
    h, w = mask.shape
    overlap = np.random.uniform(low=min_overlap, high=max_overlap)
    offx = np.random.uniform(overlap - 1, 1 - overlap)
    if offx < 0:
        over_y = overlap / (offx + 1)
        if np.random.rand() > 0.5:
            offy = over_y - 1
        else:
            offy = 1 - over_y
    else:
        over_y = overlap / (1 - offx)
        if np.random.rand() > 0.5:
            offy = over_y - 1
        else:
            offy = 1 - over_y
    assert offy > -1 and offy < 1
    bbox = (int(offx * h), int(offy * h), w, h)
    shift_eraser = crop_padding(eraser, bbox, pad_value=(0,))
    assert shift_eraser.max() <= 1, "eraser max: {}".format(eraser.max())
    ratio = ((mask == 1) & (shift_eraser == 1)).sum() / float((mask == 1).sum() + 1e-5)
    return shift_eraser, ratio
