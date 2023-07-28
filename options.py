from __future__ import absolute_import, division, print_function

import os
import argparse
import time

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in
_time = time.strftime('%Y%m%d%H', time.localtime(time.time()))

class DartBlurOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="DartBlur options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the widerface",
                                 default=os.path.join(file_dir, "data"))
        self.parser.add_argument("--blurred_image_path",
                                 type=str,
                                 help="path to the blurred data",
                                 default='images_G2_M80/')
        self.parser.add_argument("--training_dataset_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, 'data/widerface/train/label.txt'))

        # TRAINING options
        
        self.parser.add_argument("--input_blurred_data",
                                 type=bool,
                                 help="load Blurred images",
                                 default=True)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=8)
        self.parser.add_argument("--start_epoch",
                                 type=int,
                                 help="start from ? epoch",
                                 default=0)
        self.parser.add_argument("--num_epoch",
                                 type=int,
                                 help="number of epochs",
                                 default=100)
        self.parser.add_argument("--num_fg_epoch",
                                 type=int,
                                 help="number of RetinaFace training epochs",
                                 default=3)
        self.parser.add_argument("--lr_org_g",
                                 type=float,
                                 help="lr of Unet",
                                 default=2e-4)
        self.parser.add_argument("--gamma_g",
                                 type=float,
                                 help="gamma of Unet",
                                 default=0.925)
        self.parser.add_argument("--lr_org_fg",
                                 type=float,
                                 help="lr of pretrained RetinaFace",
                                 default=5e-4)
        self.parser.add_argument("--gamma_fg",
                                 type=float,
                                 help="gamma of pretrained RetinaFace",
                                 default=0.925)
        self.parser.add_argument("--budget",
                                 type=float,
                                 default=20)
        self.parser.add_argument("--label_smoothing",
                                 type=float,
                                 default=0.2)

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)

        # LOADING options
        self.parser.add_argument("--load_checkpoint_g",
                                 type=bool,
                                 help="load Unet checkpoint",
                                 default=True)
        self.parser.add_argument("--g_path",
                                 type=str,
                                 help="path to the Unet weight",
                                 default=os.path.join(file_dir, 'model_data/model_g.pt'))
        self.parser.add_argument("--load_checkpoint_fg",
                                 type=bool,
                                 help="load pretrained RetinaFace checkpoint",
                                 default=False)
        self.parser.add_argument("--fg_path",
                                 type=str,
                                 help="path to the pretrained Retinaface model",
                                 default=os.path.join(file_dir, 'model_data/model_fg.pt'))
        self.parser.add_argument("--retinaface_path",
                                 type=str,
                                 help="path to the Retinaface model",
                                 default=os.path.join(file_dir, 'model_data/Retinaface_mobilenet0.25.pth'))
        
        # LOGGING options
        self.parser.add_argument("--pth_save_dir",
                                 type=str,
                                 help="weights log directory",
                                 default='./logs/pth_' + _time)
        self.parser.add_argument("--debug_save_dir",
                                 type=str,
                                 help="blurred-img log directory",
                                 default='./logs/img_' + _time)


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
