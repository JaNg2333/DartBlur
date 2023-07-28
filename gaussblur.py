# coding=utf-8
import os
import cv2
import sys
import json
import numpy as np

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

modes = ['val','train']

for mode in modes:

    print(mode)

    image_root = './data/widerface/'+mode+'/images/'
    mosaic_root = './data/widerface/'+mode+'/images_G2_M80/'

    with open('./data/widerface/wider_face_split/wider_face_'+mode+'_bbx_gt.txt','r') as f:
        lines = f.readlines()
        num_lines = len(lines)
        i_l=0
        img_id=1
        anno_id=1
        imagepath=None
        while i_l < num_lines:
            # print(num_lines, '\\', i_l, '\t-', img_id)
            if len(lines[i_l]) < 1:
                break
            if '--' in lines[i_l]:
                imagepath=lines[i_l].strip()
                im=image_root+imagepath
                im = cv2.imread(im)
                height, width, channels = im.shape
                dataset["images"].append({"file_name": imagepath, "coco_url": "local", "height": height, "width": width, "flickr_url": "local", "id": img_id})
                i_l+=1
                num_gt=int(lines[i_l])
                while num_gt>0:
                    i_l+=1
                    x1,y1,wid,hei=list(map(int, lines[i_l].split()))[:4]
                    num_gt-=1
                    dataset["annotations"].append({
                        "segmentation": [],
                        "iscrowd": 0,
                        "area": wid * hei,
                        "image_id": img_id,
                        "bbox": [x1, y1, wid, hei],
                        "category_id": 1,
                        "id": anno_id})
                    anno_id = anno_id + 1

                    # gauss
                    kernel_width = (wid//2) | 1
                    kernel_height = (hei//2) | 1
                    # print(kernel_width, ' ', kernel_height)
                    if hei>1 and wid>1:
                        blurred_face = cv2.GaussianBlur(im[y1:y1+hei, x1:x1+wid], (kernel_width, kernel_height), 0)
                        im[y1:y1+hei, x1:x1+wid] = blurred_face

                img_id+=1
                if not os.path.exists(mosaic_root+os.path.dirname(imagepath)):
                    os.makedirs(mosaic_root+os.path.dirname(imagepath))
                cv2.imwrite(mosaic_root+imagepath, im, [int(cv2.IMWRITE_JPEG_QUALITY),80])
            i_l+=1
