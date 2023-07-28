# coding=utf-8
from copy import deepcopy
import os
import cv2
import sys
import json
import numpy as np
from alive_progress import alive_bar
# Unet
import torch
from nets.UNet import UNet
from utils.config import cfg_mnet
from utils.utils import process_img
from PIL import Image

# data
dataset = { "info": {
            "description": "WIDER face in COCO format.",
            "url": "",
            "version": "1.1",
            "contributor": "aimhabo",
            "date_created": "2020-09-29"},
            "images": [],
            "annotations": [],
            "categories": [{'id': 1, 'name': 'face'}],
}

def get_mask(img, targets):
    with torch.no_grad():
        binary_mask = torch.zeros(img.shape[0], 1, img.shape[2], img.shape[3]).type(torch.FloatTensor).to(device)
        for b in range(len(targets)):
            for f in range(targets[b].shape[0]):
                box = targets[b][f, 0:4]
                bbox = [int(box[x] * 768) for x in range(len(box))]
                binary_mask[b, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.0
    return binary_mask

def save_mask(images, output):
    images = images.data.cpu().numpy()
    img_item = np.clip(images[0].transpose(1, 2, 0)*255., 0, 255)
    cv2.imwrite(os.path.join(output, 'mask.jpg'), img_item)

def save_gx(images):
    images = images.data.cpu().numpy()
    img_item = np.clip(images[0].transpose(1, 2, 0) + np.array((104, 117, 123), np.float32), 0, 255)
    return img_item

def get_gx(x, b, g):
    x_b = torch.cat([x / 127.5, b], dim=1)
    g_x = (g(x_b) * 127.5) * b + x * (1 - b)
    return g_x

def get_blurred_image(x, b, g):
    offset = torch.tensor([[[[104.]], [[117.]], [[123.]]]], device=x.device)

    def normalize(x):
        return (x + offset) / 127.5 - 1

    def inverse_normalize(x):
        return (x + 1) * 127.5 - offset

    with torch.no_grad():
        x_b = torch.cat([normalize(x), b], dim=1)
    g_x = inverse_normalize(g(x_b)) * b + x * (1 - b)

    return g_x

def inv_process_img(o_x, g_x, box, dx, dy):
    res = deepcopy(o_x)
    g_x_copy = deepcopy(g_x)
    iw, ih = o_x.size
    h, w, _ = g_x.shape

    nw = int(w - dx)
    nh = int(w - dy)

    g_x_copy = Image.fromarray(np.uint8(g_x_copy))
    g_x_copy = g_x_copy.crop((dx, dy, nw, nh)).resize((iw, ih))

    for i in range(box.shape[0]):
        r_x1, r_y1, r_x2, r_y2 = int(box[i, 0]), int(box[i, 1]), int(box[i, 2]), int(box[i, 3])
        res.paste(g_x_copy.crop((r_x1, r_y1, r_x2, r_y2)), (r_x1, r_y1))

    return res

modes = ['val','train']

for mode in modes:
    print(mode)
    image_root = './data/widerface/WIDER_' + mode + '/images_G2/'
    mosaic_root = './data/widerface/WIDER_' + mode + '/images_dartblur/'
    # Unet
    model_path = './model_data/dartblur.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unet = UNet(input_channel=4)
    unet.load_state_dict(torch.load(model_path))
    unet = unet.to(device)
    with open('./data/widerface/wider_face_split/wider_face_' + mode + '_bbx_gt.txt','r') as f:
        lines = f.readlines()
        num_lines = len(lines)
        i_l=0
        img_id=1
        anno_id=1
        imagepath=None

        i = 0
        total_img = 0
        while i < num_lines:
            if len(lines[i]) < 1:
                    break
            if '--' in lines[i]:
                total_img += 1
            i+=1

        with alive_bar(total_img) as bar:
            while i_l < num_lines:
                # print(num_lines, '\\', i_l, '\t-', img_id)
                if len(lines[i_l]) < 1:
                    break
                if '--' in lines[i_l]:
                    imagepath=lines[i_l].strip()
                    im=image_root+imagepath
                    img = Image.open(im)
                    i_l+=1
                    num_gt=int(lines[i_l])
                    bbox = np.zeros((num_gt, 4))
                    while num_gt>0:
                        i_l+=1
                        x1,y1,wid,hei=list(map(int, lines[i_l].split()))[:4]
                        num_gt-=1
                        anno_id = anno_id + 1
                        bbox[num_gt,:] = np.array([x1,y1,x1+wid,y1+hei])
                    image, targets, dx, dy = process_img(img, np.array(bbox), 768)
                    image = torch.from_numpy(image).type(torch.FloatTensor).to(device).unsqueeze(0)
                    targets = [torch.from_numpy(targets).type(torch.FloatTensor).to(device)]
                    binary_mask = get_mask(image, targets)
                    gx = get_blurred_image(image, binary_mask, unet)
                    im = save_gx(gx)

                    res = inv_process_img(img, im, np.array(bbox), dx, dy)
                    img_id+=1
                    if not os.path.exists(mosaic_root+os.path.dirname(imagepath)):
                        os.makedirs(mosaic_root+os.path.dirname(imagepath))
                    res.save(mosaic_root+imagepath, quality=95, subsampling=0)
                    bar()
                i_l+=1
                
