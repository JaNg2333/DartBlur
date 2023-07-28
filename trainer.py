##########################
# packages
##########################
from __future__ import absolute_import, division, print_function

import os
import cv2
import higher
import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from nets.UNet import UNet
from nets.retinaface import RetinaFace
from nets.retinaface_training import MultiBoxLoss

from utils.config import cfg_mnet
from utils.dataloader import DataGenerator, detection_collate
from utils.anchors import Anchors
from utils.utils_bbox import decode, decode_landm, non_max_suppression


class Trainer:
    def __init__(self, options):
        self.opt = options
        # self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        
        ##########################
        # base settings
        ##########################
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.start_epoch = self.opt.start_epoch
        self.load_checkpoint_g = self.opt.load_checkpoint_g
        self.load_checkpoint_fg = self.opt.load_checkpoint_fg
        self.input_blurred_data = self.opt.input_blurred_data
        self.budget = self.opt.budget
        self.label_smoothing = self.opt.label_smoothing

        self.num_epoch = self.opt.num_epoch
        self.num_fg_epoch = self.opt.num_fg_epoch
        self.batch_size = self.opt.batch_size
        self.num_workers = self.opt.num_workers
        self.lr_org_g = self.opt.lr_org_g
        self.gamma_g = self.opt.gamma_g
        self.lr_org_fg = self.opt.lr_org_fg
        self.gamma_fg = self.opt.gamma_fg

        self.blurred_image_path = self.opt.blurred_image_path
        self.cfg = cfg_mnet
        self.image_size = self.cfg['train_image_size']

        self.training_dataset_path = self.opt.training_dataset_path
        self.retinaface_path = self.opt.retinaface_path
        self.g_path = self.opt.g_path
        self.fg_path = self.opt.fg_path

        self.pth_save_dir = self.opt.pth_save_dir
        self.debug_save_dir = self.opt.debug_save_dir


        ##########################
        # foler preparation
        ##########################
        if not os.path.exists(self.debug_save_dir):
            os.makedirs(self.debug_save_dir)
        if not os.path.exists(self.pth_save_dir):
            os.makedirs(self.pth_save_dir)

    ##########################
    # utils
    ##########################
    def m_save(self, model, model_path):
        torch.save(model.state_dict(), model_path)
        print('Model saved.')


    def m_load(self, model, model_path):
        model.load_state_dict(torch.load(model_path))
        print('Best checkpoint loaded.')


    def m_get_binary_mask(self, image_shape, targets):
        with torch.no_grad():
            binary_mask = torch.zeros(image_shape[0], 1, image_shape[2],
                                    image_shape[3]).type(torch.FloatTensor).to(self.device)
            for b in range(len(targets)):
                for f in range(targets[b].shape[0]):
                    box = targets[b][f, 0:4]
                    bbox = [int(box[x] * self.image_size) for x in range(len(box))]
                    binary_mask[b, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.0
        return binary_mask


    def m_get_blurred_image(self, x, b, g):
        offset = torch.tensor([[[[104.]], [[117.]], [[123.]]]], device=x.device)

        def normalize(x):
            return (x + offset) / 127.5 - 1

        def inverse_normalize(x):
            return (x + 1) * 127.5 - offset

        with torch.no_grad():
            x_b = torch.cat([normalize(x), b], dim=1)
        g_x = inverse_normalize(g(x_b)) * b + x * (1 - b)

        return g_x


    def m_loss_rev_with_mask(self, x1, x2, mask, target=0.0, mode='below', reduction=True):
        loss = F.smooth_l1_loss(x1, x2, reduction='none')  # [B*C*W*H]

        rescale_factor = x1.shape[2] * x1.shape[3] * 0.01
        loss = torch.sum((loss * mask).view(loss.shape[0], -1) / rescale_factor / 3,
                        dim=1) / (torch.sum(mask.view(mask.shape[0], -1) / rescale_factor, dim=1) + 1e-7)  # [B]

        if target > 0.0:
            if mode == 'below':
                # optimize until the loss below target
                loss = F.relu(loss - target)
            elif mode == 'exact':
                # optimize the loss to be exactly taget
                loss = torch.abs(loss - target)
            else:
                raise NotImplementedError

        if reduction:
            loss = torch.mean(loss)

        return loss


    def m_save_g_x(self, images, gaussian_images, blurred_images, output, epoch, step):
        images = images.data.cpu().numpy()
        gaussian_images = gaussian_images.data.cpu().numpy()
        blurred_images = blurred_images.data.cpu().numpy()

        def numpy2img(array):
            img_item = np.clip(array.transpose(1, 2, 0) + np.array((104, 117, 123), np.float32), 0, 255)
            img_item = cv2.cvtColor(img_item, cv2.COLOR_RGB2BGR)
            return img_item

        for i in range(len(images)):
            result = np.hstack((numpy2img(images[i]), numpy2img(gaussian_images[i]), numpy2img(blurred_images[i])))
            cv2.imwrite(os.path.join(output, 'ep_' + str(epoch) + '_st_' + str(step) + '_ba_' + str(i) + '.jpg'), result)


    def m_get_final_bbox(self, preds, anchors, conf_threshold=0.0, nms_iou=0.45):
        with torch.no_grad():
            preds_boxes, preds_conf, preds_landms = preds
            batch_size = preds_boxes.shape[0]
            boxes_conf_landms_list = []

            for i in range(batch_size):
                boxes = decode(preds_boxes[i].data.squeeze(0), anchors, self.cfg['variance'])
                conf = preds_conf[i].data.squeeze(0)[:, 1:2]
                landms = decode_landm(preds_landms[i].data.squeeze(0), anchors, self.cfg['variance'])

                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, conf_threshold, nms_iou)

                if len(boxes_conf_landms) <= 0:
                    boxes_conf_landms = np.zeros((1, 15))
                else:
                    # change the correspnding position
                    boxes_conf_landms[:, 4:14] = boxes_conf_landms[:, 5:15]
                    boxes_conf_landms[:, 14] = np.sign(boxes_conf_landms[:, 4])

                boxes_conf_landms_list.append(boxes_conf_landms)

        return boxes_conf_landms_list


    ##########################
    # pipelines
    ##########################
    def m_run_train(self, g, f, fg, criterion, anchors, dataloader, num_epoch):
        # define optimizers
        opt_g = optim.Adam(g.parameters(), lr=self.lr_org_g * self.gamma_g ** self.start_epoch, betas=(0.5, 0.999))
        sch_g = optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.gamma_g)
        opt_fg = optim.Adam(fg.parameters(), lr=self.lr_org_fg * self.gamma_fg ** self.start_epoch, betas=(0.5, 0.999))
        sch_fg = optim.lr_scheduler.ExponentialLR(opt_fg, gamma=self.gamma_fg)
        opt_fg_fixed = optim.Adam(fg.parameters(), lr=self.lr_org_fg, betas=(0.5, 0.999))

        # wrapped loss function
        def wrapped_detection_loss(preds, targets, label_smoothing=0.0):
            r_loss, c_loss, _ = criterion(preds, anchors, targets, label_smoothing)
            return self.cfg['loc_weight'] * r_loss + c_loss

        # begin training
        for epoch in range(self.start_epoch, num_epoch):
            # optimizing fg epoch
            loss_dict = {'fg': 0, 'step': 0}
            with tqdm(total=len(dataloader) * self.num_fg_epoch, desc=f'Epoch {epoch + 1}/{num_epoch} for fg', postfix=dict, mininterval=0.3) as pbar:
                for i in range(self.num_fg_epoch):
                    for data in dataloader:
                        # prepare the data
                        x, x_blur, y = data
                        with torch.no_grad():
                            x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
                            x_blur = torch.from_numpy(x_blur).type(torch.FloatTensor).to(self.device)
                            y = [torch.from_numpy(ann).type(torch.FloatTensor).to(self.device) for ann in y]

                        if self.input_blurred_data:
                            x_input = x_blur
                        else:
                            x_input = x

                        # get faces mask
                        b = self.m_get_binary_mask(x.shape, y)

                        with torch.no_grad():
                            g_x_no_grad = self.m_get_blurred_image(x_input, b, g)

                        fg.train()
                        fg_g_x = fg(g_x_no_grad)

                        loss_f_m = wrapped_detection_loss(fg_g_x, y, label_smoothing=self.label_smoothing)

                        opt_fg_fixed.zero_grad()
                        loss_f_m.backward()
                        opt_fg_fixed.step()

                        # output metrics
                        loss_dict['fg'] += loss_f_m.detach().data.cpu().numpy()
                        loss_dict['step'] += 1
                        disp_fg = loss_dict['fg'] / loss_dict['step']

                        pbar.set_postfix(**{
                            'fg': disp_fg,
                        })
                        pbar.update(1)

            # optimizing g epoch
            loss_dict = {'rev': 0, 'fid_proc': 0, 'fid_post': 0, 'fid_cycl': 0, 'fg': 0, 'step': 0}
            with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epoch} for g', postfix=dict, mininterval=0.3) as pbar:
                for data in dataloader:
                    try:
                        # prepare the data
                        x, x_blur, y = data
                        with torch.no_grad():
                            x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
                            x_blur = torch.from_numpy(x_blur).type(torch.FloatTensor).to(self.device)
                            y = [torch.from_numpy(ann).type(torch.FloatTensor).to(self.device) for ann in y]

                        if self.input_blurred_data:
                            x_input = x_blur
                        else:
                            x_input = x

                        # get faces mask
                        b = self.m_get_binary_mask(x.shape, y)

                        # aligning target
                        f.eval()
                        with torch.no_grad():
                            f_x = f(x)
                            # the conf_threshold need to match with test setting
                            f_x = self.m_get_final_bbox(f_x, anchors, conf_threshold=0.01)
                            f_x = [torch.from_numpy(ann).type(torch.FloatTensor).to(self.device) for ann in f_x]

                        # get blurred images
                        g_x = self.m_get_blurred_image(x_input, b, g)
                        if loss_dict['step'] % 50 == 0:
                            self.m_save_g_x(x, x_blur, g_x, self.debug_save_dir, epoch, loss_dict['step'])

                        # obtain f(g(x))
                        f.eval()
                        f_g_x = f(g_x)

                        # obtain first order predictions
                        fg.eval()
                        fgt_g_x = fg(g_x)
                        # fgt_x = fg(x)

                        # inner loop for second order
                        with higher.innerloop_ctx(fg, opt_fg, track_higher_grads=True) as (fgt, opt_fgt):
                            fgt.train()
                            for _ in range(1):
                                loss_fgt = wrapped_detection_loss(fgt(g_x), y, label_smoothing=self.label_smoothing)
                                opt_fgt.step(loss_fgt)

                            fgt.eval()
                            # fgt_g_x = fgt(g_x)
                            fgt_x = fgt(x)

                        # compute alignment losses
                        loss_rev = self.m_loss_rev_with_mask(g_x, x_blur, b, target=self.budget)
                        loss_f_g_x = wrapped_detection_loss(f_g_x, f_x)
                        loss_fgt_g_x = wrapped_detection_loss(fgt_g_x, f_x, label_smoothing=self.label_smoothing)
                        loss_fgt_x = wrapped_detection_loss(fgt_x, f_x, label_smoothing=self.label_smoothing)

                        loss_total = loss_rev + loss_f_g_x + loss_fgt_g_x + loss_fgt_x

                        # optimize g
                        opt_g.zero_grad()
                        loss_total.backward()
                        nn.utils.clip_grad_norm_(g.parameters(), max_norm=500, norm_type=2)
                        opt_g.step()

                        # optimize fg
                        with torch.no_grad():
                            g_x_no_grad = self.m_get_blurred_image(x_input, b, g)

                        fg.train()
                        fg_g_x = fg(g_x_no_grad)

                        loss_f_m = wrapped_detection_loss(fg_g_x, y, label_smoothing=self.label_smoothing)

                        opt_fg.zero_grad()
                        loss_f_m.backward()
                        opt_fg.step()

                        # output metrics
                        loss_dict['rev'] += loss_rev.detach().data.cpu().numpy()
                        loss_dict['fid_proc'] += loss_f_g_x.detach().data.cpu().numpy()
                        loss_dict['fid_post'] += loss_fgt_g_x.detach().data.cpu().numpy()
                        loss_dict['fid_cycl'] += loss_fgt_x.detach().data.cpu().numpy()
                        loss_dict['fg'] += loss_f_m.detach().data.cpu().numpy()
                        loss_dict['step'] += 1

                        disp_rev = loss_dict['rev'] / loss_dict['step']
                        disp_proc = loss_dict['fid_proc'] / loss_dict['step']
                        disp_post = loss_dict['fid_post'] / loss_dict['step']
                        disp_cycl = loss_dict['fid_cycl'] / loss_dict['step']
                        disp_fg = loss_dict['fg'] / loss_dict['step']

                        pbar.set_postfix(**{
                            'rev': disp_rev,
                            'fid_proc': disp_proc,
                            'fid_post': disp_post,
                            'fid_cycl': disp_cycl,
                            'fg': disp_fg,
                        })
                        pbar.update(1)
                    except RuntimeError as e:
                        print(e)
                        opt_g.zero_grad()
                        opt_fg.zero_grad()

            # reduce lr
            sch_g.step()
            sch_fg.step()

            # save the checkpoint
            self.m_save(g, os.path.join(self.pth_save_dir, 'g_epoch_%d_loss_%.4f_%.4f_%.4f_%.4f_%.4f.pt') %
                (epoch, disp_rev, disp_proc, disp_post, disp_cycl, disp_fg))
            self.m_save(fg, os.path.join(self.pth_save_dir, 'fg_epoch_%d_loss_%.4f_%.4f_%.4f_%.4f_%.4f.pt') %
                (epoch, disp_rev, disp_proc, disp_post, disp_cycl, disp_fg))
            
    def train(self):
        print('Load retinaface')
        retinaface = RetinaFace(cfg=self.cfg, pretrained=True)
        if retinaface != '':
            print('Load weights {}.'.format(self.retinaface_path))
            retinaface_dict = retinaface.state_dict()
            pretrained_dict = torch.load(self.retinaface_path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(retinaface_dict[k]) == np.shape(v)}
            retinaface_dict.update(pretrained_dict)
            retinaface.load_state_dict(retinaface_dict)

        criterion = MultiBoxLoss(2, 0.35, 7, self.cfg['variance'], cuda=True)  # False if only 'cpu'
        anchors = Anchors(self.cfg, image_size=(self.image_size, self.image_size)).get_anchors()

        # get models
        retinaface.to(self.device)
        retinaface_otf = deepcopy(retinaface)
        unet = UNet(input_channel=4)
        if self.load_checkpoint_g:
            self.m_load(unet, self.g_path)
        if self.load_checkpoint_fg:
            self.m_load(retinaface_otf, self.fg_path)

        # to device
        retinaface = retinaface.to(self.device)
        retinaface_otf = retinaface_otf.to(self.device)
        unet = unet.to(self.device)
        anchors = anchors.to(self.device)

        # data generator
        training_dataset = DataGenerator(self.training_dataset_path, self.image_size, blurred_image_path=self.blurred_image_path)
        training_loader = DataLoader(training_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=detection_collate)

        print('Begin training pipeline')
        self.m_run_train(unet, retinaface, retinaface_otf, criterion, anchors, training_loader, self.num_epoch)


