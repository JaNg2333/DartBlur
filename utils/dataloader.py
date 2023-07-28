import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image

from utils.utils import preprocess_input


class DataGenerator(data.Dataset):
    def __init__(self, txt_path, img_size, origin_image_path='images/', blurred_image_path='images_G2/'):
        self.img_size = img_size
        self.txt_path = txt_path

        self.o_path = origin_image_path
        self.b_path = blurred_image_path

        self.imgs_path, self.words = self.process_labels()

    def __len__(self):
        return len(self.imgs_path)

    def get_len(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = Image.open(self.imgs_path[index])
        img_blur = Image.open(self.imgs_path[index].replace(self.o_path, self.b_path))
        labels = self.words[index]
        annotations = np.zeros((0, 15))

        if len(labels) == 0:
            return img, annotations

        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4] < 0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)

        img, img_blur, target = self.get_random_data(img, img_blur, target, [self.img_size, self.img_size])

        img = np.transpose(preprocess_input(np.array(img, np.float32)), (2, 0, 1))
        img_blur = np.transpose(preprocess_input(np.array(img_blur, np.float32)), (2, 0, 1))

        # modified here
        return img, img_blur, target

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, image_blur, targes, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4):
        iw, ih = image.size
        h, w = input_shape
        box = targes

        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)

        scale = self.rand(0.95, 1.05)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        image_blur = image_blur.resize((nw, nh), Image.BICUBIC)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # modified here
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image_blur, (dx, dy))
        image_blur = new_image

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # modified here
            image_blur = image_blur.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        image_blur_data = np.array(image_blur, np.uint8)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2, 4, 6, 8, 10, 12]] = box[:, [0, 2, 4, 6, 8, 10, 12]] * nw / iw + dx
            box[:, [1, 3, 5, 7, 9, 11, 13]] = box[:, [1, 3, 5, 7, 9, 11, 13]] * nh / ih + dy
            if flip:
                box[:, [0, 2, 4, 6, 8, 10, 12]] = w - box[:, [2, 0, 6, 4, 8, 12, 10]]
                box[:, [5, 7, 9, 11, 13]] = box[:, [7, 5, 9, 13, 11]]

            center_x = (box[:, 0] + box[:, 2]) / 2
            center_y = (box[:, 1] + box[:, 3]) / 2

            box = box[np.logical_and(np.logical_and(center_x > 0, center_y > 0),
                                     np.logical_and(center_x < w, center_y < h))]

            box[:, 0:14][box[:, 0:14] < 0] = 0
            box[:, [0, 2, 4, 6, 8, 10, 12]][box[:, [0, 2, 4, 6, 8, 10, 12]] > w] = w
            box[:, [1, 3, 5, 7, 9, 11, 13]][box[:, [1, 3, 5, 7, 9, 11, 13]] > h] = h

            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        box[:, 4:-1][box[:, -1] == -1] = 0
        box[:, [0, 2, 4, 6, 8, 10, 12]] /= w
        box[:, [1, 3, 5, 7, 9, 11, 13]] /= h
        box_data = box

        return image_data, image_blur_data, box_data

    def process_labels(self):
        imgs_path = []
        words = []
        f = open(self.txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = self.txt_path.replace('label.txt', self.o_path) + path
                imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
        words.append(labels)
        return imgs_path, words


def detection_collate(batch):
    images = []
    images_blur = []
    targets = []
    for img, img_blur, box in batch:
        if len(box) == 0:
            continue
        images.append(img)
        # modified here
        images_blur.append(img_blur)
        targets.append(box)
    images = np.array(images)
    images_blur = np.array(images_blur)

    return images, images_blur, targets
