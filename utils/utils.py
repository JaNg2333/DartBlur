import cv2
import numpy as np
from PIL import Image

def letterbox_image(image, size):
    ih, iw, _ = np.shape(image)
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(image, (nw, nh))
    new_image = np.ones([size[1], size[0], 3]) * 128
    new_image[(h - nh) // 2:nh + (h - nh) // 2, (w - nw) // 2:nw + (w - nw) // 2] = image
    return new_image


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def preprocess_input(image):
    image -= np.array((104, 117, 123), np.float32)
    return image


def process_img(image, target, input_shape):
    iw, ih = image.size
    h, w = input_shape, input_shape
    box = target

    new_ar = w / h

    if new_ar < 1:
        nh = int(h)
        nw = int(nh * new_ar)
    else:
        nw = int(w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    dx = int(w - nw)
    dy = int(h - nh)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image
    image_data = np.array(image, np.uint8)

    if len(box) > 0:
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        center_x = (box[:, 0] + box[:, 2]) / 2
        center_y = (box[:, 1] + box[:, 3]) / 2
        box = box[np.logical_and(np.logical_and(center_x > 0, center_y > 0),
                                    np.logical_and(center_x < w, center_y < h))]
        box[:, 0:4][box[:, 0:4] < 0] = 0
        box[:, [0, 2]][box[:, [0, 2]] > w] = w
        box[:, [1, 3]][box[:, [1, 3]] > h] = h

        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
    # print(box)
    box[:, [0, 2]] = box[:, [0, 2]] / float(w)
    box[:, [1, 3]] = box[:, [1, 3]] / float(h)
    box_data = box
    # print(box_data)
    # modified here
    image_data = np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1))

    return image_data, box_data, dx, dy

 