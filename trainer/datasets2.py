"""
Copyright (C) 2019, 2020 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# pylint: disable=C0111, R0913, R0903, R0914, W0511
import random
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from PIL import Image
from skimage import img_as_float32
from skimage.exposure import rescale_intensity

from models_model_lib import load_train_image_and_annot
from file_utils import ls
import im_utils


class TrainDataset(Dataset):
    def __init__(self, train_annot_dir, dataset_dir, in_w, out_w):
        """
        in_w and out_w are the tile size in pixels
        """
        self.in_w = in_w
        self.out_w = out_w
        self.train_annot_dir = train_annot_dir
        self.dataset_dir = dataset_dir

    def __len__(self):
        # use at least 612 but when dataset gets bigger start to expand
        # to prevent validation from taking all the time (relatively)
        return max(612, len(ls(self.train_annot_dir)) * 2)

    def __getitem__(self, _):
        image, annot, fname = load_train_image_and_annot(self.dataset_dir,
                                                         self.train_annot_dir)

        tile_pad = (self.in_w - self.out_w) // 2

        # ensures each pixel is sampled with equal chance
        im_pad_w = self.out_w + tile_pad
        padded_w = image.shape[1] + (im_pad_w * 2)
        padded_h = image.shape[0] + (im_pad_w * 2)
        padded_im = im_utils.pad(image, im_pad_w)

        # This speeds up the padding.
        annot = annot[:, :, :2] #RG no B
        padded_annot = im_utils.pad(annot, im_pad_w)
        right_lim = padded_w - self.in_w
        bottom_lim = padded_h - self.in_w

        # TODO:
        # Images with less annoations will still give the same number of
        # tiles in the training procedure as images with more annotation.
        # Further empirical investigation into effects of
        # instance selection required are required.
        while True:
            x_in = math.floor(random.random() * right_lim)
            y_in = math.floor(random.random() * bottom_lim)
            annot_tile = padded_annot[y_in:y_in+self.in_w,
                                      x_in:x_in+self.in_w]
            if np.sum(annot_tile) > 0:
                break

        im_tile = padded_im[y_in:y_in+self.in_w,
                            x_in:x_in+self.in_w]

        assert annot_tile.shape == (self.in_w, self.in_w, 2), (
            f" shape is {annot_tile.shape} for tile from {fname}")

        assert im_tile.shape == (self.in_w, self.in_w, 4), (
            f" shape is {im_tile.shape} for tile from {fname}")



        seg = np.array(im_tile[:,:,3:])/255
        im_tile = np.array(im_tile[:,:,:3])


        im_tile = img_as_float32(im_tile)

        im_tile = np.concatenate((im_tile,seg), axis=2)


        im_tile = im_utils.normalize_tile(im_tile)
        #im_tile, annot_tile = self.augmentor.transform(im_tile, annot_tile)
        #im_tile = im_utils.normalize_tile(im_tile)

        foreground = np.array(annot_tile)[:, :, 0]
        background = np.array(annot_tile)[:, :, 1]

        # Annotion is cropped post augmentation to ensure
        # elastic grid doesn't remove the edges.
        foreground = foreground[tile_pad:-tile_pad, tile_pad:-tile_pad]
        background = background[tile_pad:-tile_pad, tile_pad:-tile_pad]
        # mask specified pixels of annotation which are defined
        mask = foreground + background
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        foreground = foreground.astype(np.int64)
        foreground = torch.from_numpy(foreground)
        im_tile = im_tile.astype(np.float32)
        im_tile = np.moveaxis(im_tile, -1, 0)
        im_tile = torch.from_numpy(im_tile)
        return im_tile, foreground, mask