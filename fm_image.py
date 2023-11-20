"""This module provides a class to save an image with the corresponding ground truth and 
other parameters"""
import os
import PIL
import torch
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt

from dip.utils import denoising_utils

dtype = torch.cuda.FloatTensor


class Image():
    """Image with corresponding ground truth (if available), paths, crop regions, ...
    """
    def __init__(self,
                 data_noisy,
                 path_noisy,
                 save_dir_noisy,
                 save_dir_denoised,
                 data_gt = None,
                 path_gt = None,
                 crop_region = None,
                 frame_number = None,
                 images_min = None,
                 images_max = None):

        self.data_noisy = data_noisy
        self.img_noisy_torch = denoising_utils.np_to_torch(data_noisy.astype('float')).type(dtype)
        self.data_gt = data_gt
        self.path_noisy = path_noisy
        self.path_gt = path_gt
        self.save_dir_noisy = save_dir_noisy
        self.save_dir_denoised = save_dir_denoised
        self.crop_region = crop_region
        self.frame_number = frame_number

        self.base_file_name = self.get_base_file_name()

        if path_noisy.endswith(".png"):
            self.file_type = "png"
        elif path_noisy.endswith(".tif") or path_noisy.endswith(".tiff"):
            self.file_type = "tif"
        else:
            self.file_type = "unknown"

        self.save_file_noisy = None
        self.data_denoised = None
        self.save_file_denoised = None

        self.images_min = images_min
        self.images_max = images_max


    def save_noisy(self):
        """save noisy image in color and as png 
        """
        if not self.save_dir_noisy:
            return

        file_name = self.get_base_file_name()

        if not os.path.exists(self.save_dir_noisy):
            os.makedirs(self.save_dir_noisy)

        # save colored image
        self.save_file_noisy = f"{self.save_dir_noisy}/{file_name}_color.png"
        cm = plt.get_cmap('viridis')
        colored_image = cm(np.clip(self.data_noisy.squeeze(), 0, 1))
        img_pil_color = PIL.Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
        img_pil_color.save(self.save_file_noisy)


    def save_denoised(self, current_time):
        """save denoised image in color and in gray scale as png/tif 
        (depending on the original data type)

        Args:
            current_time (str): string that contains the current time
        """
        base_file_name = self.get_base_file_name()
        file_name = f"{base_file_name}_{current_time}"

        if not os.path.exists(self.save_dir_denoised):
            os.makedirs(self.save_dir_denoised)

        # save colored image
        self.save_file_denoised = f"{self.save_dir_denoised}/{file_name}_color.png"
        cm = plt.get_cmap('viridis')
        colored_image = cm(np.clip(self.data_denoised.squeeze(), 0, 1))
        save_img_color = PIL.Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
        save_img_color.save(self.save_file_denoised)

        if self.file_type == "png":
            # save gray image
            save_file_denoised_gray = f"{self.save_dir_denoised}/{file_name}.png"
            save_img = PIL.Image.fromarray(
                np.clip(self.data_denoised.squeeze(), 0, 1) * 255).convert('L')
            save_img.save(save_file_denoised_gray)

        # save image as tif
        if self.file_type == 'tif':
            save_data_denoised = self.data_denoised.squeeze()
            if not self.images_min is None and not self.images_max is None:
                save_data_denoised = (save_data_denoised * (self.images_max - self.images_min) +
                    self.images_min)
            save_file_denoised_tif = f"{self.save_dir_denoised}/{file_name}.tif"
            tf.imwrite(save_file_denoised_tif, save_data_denoised)


    def get_base_file_name(self):
        """extract base file name from path

        Returns:
            str: file name
        """
        file_name_prefix = os.path.splitext(os.path.basename(self.path_noisy))[0]
        if self.crop_region:
            file_name = (f"{file_name_prefix}_{self.crop_region[0]}_{self.crop_region[1]}" +
                         f"_{self.crop_region[2]}_{self.crop_region[3]}")
        else:
            file_name = file_name_prefix
        if self.frame_number > -1:
            file_name = f"{file_name}_frame{self.frame_number:03}"
        return file_name
