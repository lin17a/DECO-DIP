"""This module defines an Image Sequence that contains images of type fm_image."""
import os
import sys
import PIL
import numpy as np
import tifffile as tf

import fm_image

class ImageSequence():
    """Image sequences containing multiple images of type fm_image
    """
    def __init__(self,
                 path_noisy,
                 save_path_noisy,
                 save_path_denoised,
                 time_series_params,
                 frame_range = None,
                 path_gt = None,
                 crop_region = None):

        self.images = []
        self.path_noisy = path_noisy
        self.save_path_noisy = save_path_noisy
        self.save_path_denoised = save_path_denoised

        if path_gt == "NULL":
            self.path_gt = None
        else:
            self.path_gt = path_gt
        if crop_region == "NULL":
            self.crop_region = None
        else:
            self.crop_region = crop_region

        self.is_time_series = time_series_params["is_series"]
        if not self.is_time_series:
            self.back_and_forth = False
        else:
            self.back_and_forth = time_series_params["back_and_forth"]

        if frame_range == "NULL":
            frame_range = None
        assert(frame_range is None or len(frame_range) == 2)
        self.frame_range = frame_range

        self.load_images()


    def load_images(self):
        """load tif or png images (with their corresponding ground truth image if available)
        """
        # both, noisy images and ground truths are folders
        if os.path.isdir(self.path_noisy) and os.path.isdir(self.path_gt):
            img_paths = list(self.get_image_paths(self.path_noisy))
            img_gt_paths = list(self.get_image_paths(self.path_noisy))

            for i, (file, file_gt) in enumerate(zip(img_paths, img_gt_paths)):
                assert(f"{file}"[-6:-4] == f"{file_gt}"[-6:-4] or file_gt == "NULL")
                img_list = self.prepare_imgs(file,
                                        self.save_path_noisy,
                                        self.save_path_denoised,
                                        file_gt = file_gt,
                                        frame_number = i)
                self.images.extend(img_list)

        # there are multiple noisy images in one folder, ground truth not available
        elif os.path.isdir(self.path_noisy) and not self.path_gt:
            img_paths = list(self.get_image_paths(self.path_noisy))
            for i, file in enumerate(img_paths):
                img_list = self.prepare_imgs(file,
                                        self.save_path_noisy,
                                        self.save_path_denoised,
                                        file_gt = None,
                                        frame_number = i)
                self.images.extend(img_list)

        # noisy images are in one folder and there is only one ground truth image
        elif os.path.isdir(self.path_noisy) and not os.path.isdir(self.path_gt):
            img_paths = list(self.get_image_paths(self.path_noisy))
            for i, file in enumerate(img_paths):
                img_list = self.prepare_imgs(file,
                                        self.save_path_noisy,
                                        self.save_path_denoised,
                                        frame_number = i)
                self.images.extend(img_list)

        # there is only one noisy image
        else:
            img_list = self.prepare_imgs(self.path_noisy,
                                    self.save_path_noisy,
                                    self.save_path_denoised)
            self.images.extend(img_list)

        # adjust frame range
        if self.frame_range:
            self.images = self.images[self.frame_range[0] : self.frame_range[1]]


    def load_png(self, path, crop_region = None):
        """load a png image

        Args:
            path (str): path to image
            crop_region (list, optional): list with the crop region: [left, top, right, bottom].
                Defaults to None.

        Returns:
            np.Array: numpy array with the image, shape: (1, shape_x, shape_y)
        """
        # convert to grayscale
        img = PIL.Image.open(path).convert('L')
        if crop_region:
            # crop region: [left, top, right, bottom]
            img = img.crop(tuple(crop_region))
        img_np = np.array(img)[None, ...].astype(np.float32) / 255
        return img_np

    def load_tif(self, path, imgs_min = None, imgs_max = None):
        """load tif image

        Args:
            path (str): path to tif file
            imgs_min (float, optional): minimum value of all images in the tiff file. 
                Defaults to None.
            imgs_max (float, optional): maximum value of all images in the tiff file. 
                Defaults to None.

        Returns:
            (List of np.Array, float, float): list of images as np.Array(1, shape_x, shape_y),
                imgs_min, imgs_max
        """
        imgs = tf.imread(path)
        imgs = imgs.astype(np.float32)
        if len(imgs.shape) < 3:
            imgs = imgs[None, ...]
        if self.crop_region:
            imgs = imgs[:, self.crop_region[1]:self.crop_region[3],
                        self.crop_region[0]:self.crop_region[2]]

        img_list = []
        num_imgs = imgs.shape[0]

        if imgs_min is None or imgs_max is None:
            imgs_min = imgs.min()
            imgs_max = imgs.max()
        if num_imgs == 1:
            img = (imgs - imgs_min) / (imgs_max - imgs_min)
            img = img.clip(min=0, max=1)
            img_list.append(img)
        else:
            for i in range(num_imgs):
                img = (imgs[i][None, ...] - imgs_min) / (imgs_max - imgs_min)
                img = img.clip(min=0, max=1)
                img_list.append(img)
        return img_list, imgs_min, imgs_max


    def prepare_imgs_png(self, file_noisy, save_dir_noisy, save_dir_denoised, file_gt = None,
                         frame_number = 0):
        """create an fm_image for the image

        Args:
            save_dir_noisy (str): path where to save the noisy image
            save_dir_denoised (str): path where to save the denoised image
            frame_number (int): number of the image

        Returns:
            List[fm_image.Image]: list with one fm_image
        """
        img_noisy = self.load_png(file_noisy, self.crop_region)

        img_gt = None
        if file_gt:
            assert file_noisy.endswith(".png")
            img_gt = self.load_png(file_gt, self.crop_region)

        image = fm_image.Image(data_noisy = img_noisy,
                                path_noisy = file_noisy,
                                save_dir_noisy = save_dir_noisy,
                                save_dir_denoised = save_dir_denoised,
                                data_gt = img_gt,
                                path_gt = file_gt,
                                crop_region = self.crop_region,
                                frame_number = frame_number)
        image.save_noisy()
        return [image]


    def prepare_imgs_tif(self, file_noisy, save_dir_noisy, save_dir_denoised):
        """create fm_images for each image

        Args:
            save_dir_noisy (str): path where to save noisy images
            save_dir_denoised (str): path where to save the denoised images

        Returns:
            (List[fm_image.Image]): list of fm_images
        """
        imgs_noisy, imgs_min, imgs_max = self.load_tif(file_noisy)
        if self.path_gt:
            assert file_noisy.endswith(".tif")
            imgs_gt, _, _ = self.load_tif(self.path_gt, imgs_min, imgs_max)

            images = []
            for i, (img_noisy, img_gt) in enumerate(zip(imgs_noisy, imgs_gt)):
                image = fm_image.Image(data_noisy = img_noisy,
                        path_noisy = file_noisy,
                        save_dir_noisy = save_dir_noisy,
                        save_dir_denoised = save_dir_denoised,
                        data_gt = img_gt,
                        path_gt = self.path_gt,
                        crop_region = self.crop_region,
                        frame_number = i,
                        images_min = imgs_min,
                        images_max = imgs_max)
                image.save_noisy()
                images.append(image)
        else:
            images = []
            for i, img_noisy in enumerate(imgs_noisy):
                image = fm_image.Image(data_noisy = img_noisy,
                        path_noisy = file_noisy,
                        save_dir_noisy = save_dir_noisy,
                        save_dir_denoised = save_dir_denoised,
                        data_gt = None,
                        path_gt = self.path_gt,
                        crop_region = self.crop_region,
                        frame_number = i,
                        images_min = imgs_min,
                        images_max = imgs_max)
                image.save_noisy()
                images.append(image)

        return images


    def prepare_imgs(self, path_noisy, save_dir_noisy, save_dir_denoised, file_gt = None,
                     frame_number = 0):
        """create an fm_image for each image

        Args:
            path_noisy (str): path to the noisy image
            save_dir_noisy (str): path to where the noisy images are stored
            save_dir_denoised (str): path to where the denoised images are stored
            frame_number (int, optional): number of the image/frame. Defaults to 0.

        Returns:
            List[fm_image]: list of fm_images
        """
        if path_noisy.endswith(".png"):
            image_list = self.prepare_imgs_png(path_noisy, save_dir_noisy, save_dir_denoised,
                                               file_gt, frame_number)
            return image_list
        if path_noisy.endswith(".tif"):
            image_list = self.prepare_imgs_tif(path_noisy, save_dir_noisy, save_dir_denoised)
            return image_list
        print("error: file ends neither with png nor tif")
        sys.exit()


    def get_image_paths(self, path, rev=False):
        """iterator over all png and tif files in a directory

        Args:
            path (str): path to the directory
            rev (bool, optional): iterate in reverse alphabetic order. Defaults to False.

        Yields:
            str: path to png/tif file
        """
        if rev:
            files = reversed(sorted(os.listdir(path)))
        else:
            files = sorted(os.listdir(path))
        for file in files:
            if os.path.isfile(os.path.join(path, file)) and (f"{file}".endswith(".png") or
                                                             f"{file}".endswith(".tif")):
                yield f"{path}/{file}"
