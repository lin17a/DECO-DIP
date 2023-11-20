"""This module provides a loss function containing MSE, a convolution with a PSF 
or the Lucy-Richardson functional"""
import sys
import numpy as np
from torchmetrics.image import TotalVariation
import torch

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, './TDEntropyDeconvolution')
from psf import psf

class LossFunction(torch.nn.Module):
    """loss function

    Args:
        torch (nn.Module): torch module
    """
    def __init__(self, dtype, loss_type_main, loss_type2, loss2_fact = 0, loss2_incr_fact = 0,
                 regularizer = None, regularizer_fact = 0, regularizer_incr_fact = 0,
                 psf_params = None, superres_factor = 1, downsampler = None):
        super().__init__()

        self.superres_factor = superres_factor
        self.downsampler = downsampler

        self.dtype = dtype
        if psf_params:
            self.dims = 2
            self.psf_params = psf_params
            self.init_psf()

        self.loss_func_main = self.get_loss_func(loss_type_main)
        self.loss_func2 = self.get_loss_func(loss_type2)

        self.loss2_incr_fact = loss2_incr_fact
        self.loss2_fact = loss2_fact

        self.regularizer = self.get_regularizer(regularizer)
        self.regularizer_fact = regularizer_fact
        self.regularizer_incr_fact = regularizer_incr_fact


    def forward(self, output, target, *args):
        """calculate the loss between output and target

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target

        Returns:
            torch.Tensor: loss
        """
        loss = self.general_loss_func(output, target, *args)
        return loss

    def general_loss_func(self, output, target, epoch):
        """calls the configured loss function(s)

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target
            epoch (int): number of the epoch

        Returns:
            torch.Tensor: loss
        """
        loss_main = self.loss_func_main(output, target)
        if self.loss_func2:
            loss2 = self.loss_func2(output, target)
        else:
            loss2 = 0
        if self.regularizer:
            regularization = self.regularizer(output)
        else:
            regularization = 0

        fact_loss2 = self.loss2_fact + (self.loss2_incr_fact * epoch)
        fact_loss_main = 1 - fact_loss2
        regularizer_fact = self.regularizer_fact + (self.regularizer_incr_fact * epoch)
        loss = fact_loss_main * loss_main + fact_loss2 * loss2 +  regularizer_fact * regularization

        return loss

    ## loss functions
    def loss_mse(self, output, target):
        """calculate the mean sqaured error between output and target

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target

        Returns:
            torch.Tensor: MSE
        """
        # downsample img if superresolution factor is given
        if self.superres_factor > 1:
            output = self.downsampler(output)

        return self.mse(output, target)

    def loss_psf(self, output, target):
        """Calculates the MSE between the target and the output convolved with the PSF

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target

        Returns:
            torch.Tensor: MSE(output * PSF, target)
        """
        conv = self.conv_with_psf(output)
        # downsample img if superresolution factor is given
        if self.superres_factor > 1:
            conv = self.downsampler(conv)
        return self.mse(torch.squeeze(conv), target)

    def loss_richardson_lucy(self, output, target):
        """calculate the Richardson-Lucy functional

        Args:
            output (torch.Tensor): network output
            target (toch.Tensor): target

        Returns:
            torch.Tensor: RL functional result
        """
        # downsample img if superresolution factor is given
        #if self.superres_factor > 1:
        #    output = self.downsampler(output)
        conv = torch.squeeze(self.conv_with_psf(output))
        integrant = conv - target * torch.log(conv + sys.float_info.epsilon)
        return torch.sum(integrant)

    def ssim(self, output, target):
        """Calculate the structural similarity index between output and target

        Args:
            output (np.ndarray): network output
            target (np.ndarray): target

        Returns:
            float: SSIM
        """
        # downsample img if superresolution factor is given
        # if self.superres_factor > 1:
        #     output_torch = torch.tensor(output, device='cuda')[None, :]
        #     output_upscaled = self.downsampler(output_torch)
        #     output = output_upscaled.detach().cpu().numpy()[0]
        return ssim(output, target, data_range = 1)


    ## regularizer
    def tv_norm(self, image):
        """calculate the TV-norm of an image

        Args:
            image (np.ndarray): image as numpy array

        Returns:
            float: tv norm
        """
        tv = 0
        for i in range(len(image)-1):
            for j in range(len(image[0])-1):
                tv += torch.sqrt((image[i, j+1] - image[i, j])**2 +
                                 (image[i+1, j] - image[i, j])**2)
        return tv

    def tikhonov_miller(self, img):
        """calculate the Tikhonov-Miller norm of an image

        Args:
            img (np.ndarray): image as numpy array

        Returns:
            float: Tikhonov-Miller norm of the image
        """
        diff1 = (img[..., 1:, :] - img[..., :-1, :])**2
        diff2 = (img[..., :, 1:] - img[..., :, :-1])**2

        res1 = diff1.abs().sum([1, 2, 3])
        res2 = diff2.abs().sum([1, 2, 3])
        tm = res1 + res2
        return tm


    ## helper functions
    def get_loss_func(self, loss_type):
        """choose the correct loss function

        Args:
            loss_type (str): loss type string, possible values: mse, psf, kl, rl

        Returns:
            fm_loss.LossFunction: loss funtion
        """
        match loss_type:
            case 'mse':
                return self.loss_mse
            case 'psf':
                return self.loss_psf
            case 'kl':
                return self.loss_kullback_leibler
            case 'rl':
                return self.loss_richardson_lucy
            case _:
                return None

    def get_regularizer(self, regularizer):
        """get the regularizer function

        Args:
            regularizer (str): regularizer type, possible values: tv, tm, sparse

        Returns:
            fm_loss.regularizer: regularizer function
        """
        match regularizer:
            case 'tv':
                tv = TotalVariation().to(device='cuda')
                return tv
            case 'tm':
                return self.tikhonov_miller
            case '_':
                return None

    def init_psf(self):
        """initialize point spread function

        Returns:
            np.ndarray: PSF as numpy array
        """
        psf_obj = psf.PSF(self.dims, **self.psf_params)
        psf_arr = psf_obj.data
        if self.superres_factor > 1:
            psf_arr = np.kron(psf_arr, np.ones((self.superres_factor, self.superres_factor)))
        psf_arr_torch = torch.tensor(psf_arr, device='cuda:0').type(self.dtype)
        self.psf_arr = psf_arr_torch
        return psf_arr_torch

    def mse(self, output, target):
        """Calculate the mean squared error

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target

        Returns:
            torch.Tensor: MSE
        """
        return torch.mean((output - target)**2 + sys.float_info.epsilon)

    def conv_with_psf(self, image):
        """convolve the given image with a PSF

        Args:
            image (torch.Tensor): image

        Returns:
            torch.Tensor: image convolved with the PSF
        """
        psf_size = self.psf_params['xysize'] * self.superres_factor
        in_psf = self.psf_arr.view(1, 1, psf_size, psf_size).repeat(1, 1, 1, 1)
        in_img = image.repeat(1, 1, 1, 1)
        conv = torch.real(torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftn(in_img) *
                                                              torch.fft.fftn(in_psf))))
        return conv

    def get_final_loss(self, output, target, num_epoch):
        """calculate different losses between output and target: MSE, the configured one, 
        PSNR, SSIM, tv norm of the output and tv norm of the target

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target
            num_epoch (int): number of the epoch

        Returns:
            dict: dict with different losses
        """
        target_np = target.detach().cpu().numpy()[0][0]
        if self.superres_factor > 1:
            output_np = self.downsampler(output).detach().cpu().numpy()[0][0]
        else:
            output_np = output.detach().cpu().numpy()[0][0]
        psnr = peak_signal_noise_ratio(target_np, output_np)
        total_variation = TotalVariation().to(device='cuda')
        return {
            'mse_loss' : self.loss_mse(output, target).item(),
            'configured_loss' : self.forward(output, target, num_epoch).item(),
            'psnr' : psnr,
            'ssim' : self.ssim(output_np, target_np),
            'tv_norm_out' : total_variation(output).item(),
            'tv_norm_target' : total_variation(target).item()
        }
