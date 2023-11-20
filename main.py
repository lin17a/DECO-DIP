#!/usr/bin/env python

import os
import sys
import logging
import argparse
from pprint import pprint
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio

import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from dip import models
from dip.utils import denoising_utils
from dip.models.downsampler import Downsampler

import save_results
import fm_loss
from init_params import get_parameters
from fm_image_sequence import ImageSequence


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
torch.manual_seed(0)

SEED = 13579


def write_tensorboard_summary(writer, fm_image, loss_function, output_image, iteration):
    """write loss to tensorboard 

    Args:
        writer (SummaryWriter): tensorboard summary writer
        fm_image (fm_image.Image): fm_image to compare output to
        LossFunction (fm_loss.LossFunction): loss funtion
        output_image (torch.Tensor): network output
        iteration (int): number of the iteration
    """
    loss_noisy_dict = loss_function.get_final_loss(
        output_image, torch.tensor(fm_image.data_noisy, device='cuda')[None, :], iteration)
    if not fm_image.data_gt is None:
        loss_gt_dict = loss_function.get_final_loss(output_image, torch.tensor(
            fm_image.data_gt, device='cuda')[None, :], iteration)
    else:
        loss_gt_dict = {'mse_loss': 0, 'configured_loss': 0,
                        'psnr': 0, 'ssim': 0, 'tv_norm_target': 0}
    writer.add_scalars('Loss MSE', {
        "noisy": loss_noisy_dict['mse_loss'], "gt": loss_gt_dict['mse_loss']}, iteration)
    writer.add_scalars('Loss configured', {
        "noisy": loss_noisy_dict['configured_loss'], "gt": loss_gt_dict['configured_loss']}, 
        iteration)
    writer.add_scalars('Loss PSNR', {
        "noisy": loss_noisy_dict['psnr'], "gt": loss_gt_dict['psnr']}, iteration)
    writer.add_scalars('Loss SSIM', {
        "noisy": loss_noisy_dict['ssim'], "gt": loss_gt_dict['ssim']}, iteration)
    writer.add_scalars('TV Norm', {
        "out": loss_noisy_dict['tv_norm_out'], "gt": loss_gt_dict['tv_norm_target']}, iteration)


def check_pnsr_developement(fm_image, output_image, status_dict, verbosity = 0):
    """check if the psnr got better since the last call, 
    resets the network to last checkpoint if the PSNR got worse by more than 5

    Args:
        fm_image (fm_image.Image): fm_image containing the noisy image
        output_image (torch.Tensor): network output
        status_dict (dict): contains the current status of the training
        verbosity (int, optional): verbosity. Defaults to 0.

    Returns:
        dict: status dict with the updated status of the network and PSNR
    """
    psnr = peak_signal_noise_ratio(fm_image.data_noisy, output_image.detach().cpu().numpy()[0])

    if psnr - status_dict["psnr_last"] < -5:
        if verbosity > 0:
            print('Falling back to previous checkpoint.')
        for new_param, net_param in zip(status_dict["last_net"], status_dict["net"].parameters()):
            net_param.data.copy_(new_param.cuda())
    else:
        status_dict["last_net"] = [x.detach().cpu() for x in status_dict["net"].parameters()]
        status_dict["psnr_last"] = psnr

    return status_dict


def closure(image, params, status, iteration):
    """perform one iteration of training

    Args:
        image (fm_image): image of class fm_image to be reconstructed
        params (dict): dict with parameters for training
        status (dict): saves the current status of training, like nets and old psnrs
        iteration (int): number of the iteration

    """
    if params["reg_noise_std"] > 0:
        status["net_input"] = status["net_input_saved"] + \
            (status["noise"].normal_() * params["reg_noise_std"])
    else:
        status["net_input"] = status["net_input_saved"]

    net = status["net"]
    out = net(status["net_input"])

    # Smoothing
    if status["out_avg"] is None:
        status["out_avg"] = out
    else:
        exp_weight = params["exp_weight"]
        status["out_avg"] = status["out_avg"] * \
            exp_weight + out * (1 - exp_weight)

    loss = params["loss"]

    total_loss = loss.forward(out, image.img_noisy_torch, iteration)
    total_loss.backward()

    if params["superres_factor"] > 1:
        out = params["downsampler"](out)

    if status["writer"]:
        write_tensorboard_summary(status["writer"], image, loss, out, iteration)

    if iteration % params["checkpoint_interval"] == 0:
        if params["verbosity"] > 0:
            logging.info("Iteration %d Loss %f PSNR_noisy: %f",
                        iteration, total_loss.item(), status['psnr_last'])

        if status["writer"]:
            status["writer"].add_image(tag="images", img_tensor=out[0][0], global_step=iteration,
                                       walltime=None, dataformats='HW')

        status = check_pnsr_developement(image, out, status, params["verbosity"])


def init_network(params):
    """initialize network

    Args:
        params (dict): parameters for network

    Returns:
        : neural network
    """
    return models.get_net(input_depth=params['input_depth'],
                             NET_TYPE=params['net_type'],
                             pad=params['pad'],
                             n_channels=params['n_channels'],
                             act_fun=params['act_fun'],
                             skip_n33d=params['skip_n33d'],
                             skip_n33u=params['skip_n33u'],
                             skip_n11=params['skip_n11'],
                             num_scales=params['num_scales'],
                             upsample_mode=params['upsample_mode'],
                             downsample_mode=params['downsample_mode']).type(dtype)


def init_writer(parameters):
    """initialize tensorboard summary writer

    Args:
        parameters (dict): parameters for the summary writer

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter: SummaryWriter object
    """
    writer = None
    if parameters["save_and_log"]['tensorboard']:
        folder_name = parameters["save_and_log"]['tensorboard_logdir']
        # define summary writer
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter(log_dir=f"./{folder_name}/{current_time}")
    return writer


def get_final_losses(fm_image, output_image, loss_function, num_iter):
    """calculate loss based on the given loss function and three different image pairs:
       - output and noisy image
       - output and ground truth
       - ground truth and noisy image

    Args:
        fm_image (fm_image.Image): image of class fm_image
        output_image (torch.Tensor): network output
        LossFunction (fm_loss.LossFunction): loss function
        num_iter (int): number of completed iterations

    Returns:
        (dict, dict, dict): final_loss_noisy, final_loss_gt, orig_noisy_difference
    """
    # get final loss
    final_loss_noisy = loss_function.get_final_loss(
        output_image, torch.tensor(fm_image.data_noisy, device='cuda')[None, :], num_iter)
    print("network loss (output - noisy):")
    pprint(final_loss_noisy)

    # get final loss compared to ground truth
    final_loss_gt = {}
    if fm_image.data_gt is not None:
        final_loss_gt = loss_function.get_final_loss(
            output_image, torch.tensor(fm_image.data_gt, device='cuda')[None, :], num_iter)
        print("loss compared to ground truth (output - gt):")
        pprint(final_loss_gt)

    orig_noisy_difference = {}
    if fm_image.data_gt is not None:
        orig_noisy_difference = loss_function.get_final_loss(
                torch.tensor(fm_image.data_gt, device='cuda')[None, :],
                torch.tensor(fm_image.data_noisy, device='cuda')[None, :],
                num_iter)
        print("difference between noisy image and ground truth (noisy - gt):")
        pprint(orig_noisy_difference)

    return final_loss_noisy, final_loss_gt, orig_noisy_difference


def run(parameters, image, first_image=False, net=None, net_input=None):
    """load everything and start training the network

    Args:
        parameters (dict): parameters for training
        image (fm_image): image of type fm_image 
        first_image (bool, optional): defines if it is the first image of a series. 
            Defaults to False.
        net (torch.nn.modules.container.Sequential, optional): network to train. Defaults to None.
        net_input (torch.Tensor, optional): network input. Defaults to None.

    Returns:
        (torch.nn.modules.container.Sequential, torch.Tensor): tuple of current network and last 
            network input
    """

    # get parameters
    net_params = parameters['net']
    superres_params = parameters['superresolution']
    superres_factor = superres_params['superres_factor']

    if net_params["checkpoint_interval"]:
        checkpoint_interval = net_params["checkpoint_interval"]
    else:
        checkpoint_interval = 100

    # define a generator for setting always the same seed
    generator = torch.Generator()
    generator.manual_seed(SEED)
    torch.manual_seed(SEED)

    # init network for the first picture or if we do not have a time series
    if first_image or parameters['time_series']['is_series'] != "True":
        # configure network
        net = init_network(net_params)

        # get random net input
        net_input = denoising_utils.get_noise(
            input_depth=net_params['input_depth'],
            method=net_params["INPUT"],
            spatial_size=(image.data_noisy.shape[1] * superres_factor,
                          image.data_noisy.shape[2] * superres_factor),
            generator=generator).type(dtype).detach()

    downsampler = None
    if superres_factor > 1:
        downsample_kernel = superres_params['downsample_kernel']
        # define downsampler for superresolution
        downsampler = Downsampler(n_planes=1, factor=superres_factor,
                                kernel_type=downsample_kernel, phase=0.5,
                                preserve_size=True).type(dtype)

    # define loss function
    loss = fm_loss.LossFunction(dtype, **parameters['loss'],
                                 psf_params=parameters['psf'],
                                 superres_factor=superres_factor,
                                 downsampler=downsampler)

    # get parameters for optimizer
    params_optimizer = denoising_utils.get_params(
        net_params['OPT_OVER'], net, net_input)

    # set learning rate and number of iterations depending on if it is a time series or not
    if first_image or parameters['time_series']['is_series'] != "True":
        learning_rate = net_params["learning_rate"]
        num_iter = net_params["num_iter"]
    else:
        series_params = parameters["time_series"]
        learning_rate = series_params["learning_rate"]
        num_iter = series_params["num_iter"]

    # dict that holds the current network status
    current_net_status = {
        "iter": 0,
        "out_avg": None,
        "psnr_last": 0,
        "net": net,
        "last_net": None,
        "net_input": None,
        "net_input_saved": net_input.detach().clone(),
        "writer": init_writer(parameters),
        "noise": net_input.detach().clone()
    }

    # define dict with traning parameters
    params = {
        "lr": learning_rate,
        "num_iter": num_iter,
        "reg_noise_std": net_params['reg_noise_std'],
        "exp_weight": net_params['exp_weight'],
        "loss": loss,
        "superres_factor": superres_factor,
        "downsampler": downsampler,
        "checkpoint_interval": checkpoint_interval,
        "verbosity" : parameters["save_and_log"]["verbosity"]
    }

    # train network
    denoising_utils.optimize(net_params['optimizer'],
                             params_optimizer,
                             closure,
                             image,
                             params,
                             current_net_status)

    # get final output
    #out = current_net_status["net"](current_net_status["net_input"])
    out = current_net_status["out_avg"]
    image.data_denoised = denoising_utils.torch_to_np(out)

    # get final loss
    final_loss_noisy, final_loss_gt, orig_noisy_difference = get_final_losses(image, out, loss,
                                                                              params['num_iter'])

    if parameters['save_and_log']['save_imgs']:
        save_results.save_results(image, parameters, final_loss_noisy,
                                  final_loss_gt, orig_noisy_difference)

    if current_net_status["writer"]:
        current_net_status["writer"].close()

    return current_net_status["net"], current_net_status["net_input"]


def get_config_files():
    """parse config file names from arguments

    Returns:
        list: a list of config file names
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_path',
                        nargs=1,
                        default=['parameters.yaml'],
                        help='Path to parameter file or folder with parameter files',
                        type=str)
    param_path = parser.parse_args().param_path[0]

    if os.path.isfile(param_path):
        configs = [param_path]
    elif os.path.isdir(param_path):
        configs = [f"{param_path}/{config_file}" for config_file
                   in sorted(os.listdir(param_path)) if config_file.endswith(".yaml")]
    else:
        print(f"file/directory {param_path} does not exist")
        sys.exit()

    return configs


if __name__ == "__main__":

    config_list = get_config_files()

    for cfg_num, config_file in enumerate(config_list):

        print(f"processing config number {cfg_num}: {config_file}")

        parameters = get_parameters(config_file)
        parameters['save_and_log']['save_imgs'] = True

        image_sequence = ImageSequence(
            path_noisy=parameters['image']['path'],
            save_path_noisy=parameters["save_and_log"]['orig_img_path'],
            save_path_denoised=parameters["save_and_log"]['denoised_img_path'],
            time_series_params=parameters['time_series'],
            frame_range=parameters['image']['frame_range'],
            path_gt=parameters['image']['path_gt'],
            crop_region=parameters['image']['crop_region'])

        num_imgs = len(image_sequence.images)

        if image_sequence.is_time_series:
            for r in range(parameters['image']['num_runs']):
                if image_sequence.back_and_forth:
                    parameters['save_and_log']['save_imgs'] = False
                parameters['first_image'] = True

                print("Fitting net to first image ...")
                net, net_input = run(parameters=parameters,
                                    image=image_sequence.images[0])

                parameters['first_image'] = False
                print("Iterating through images in chronological order ...")
                for i, image in enumerate(image_sequence.images[1:]):
                    print(f"Fitting net to image {i+2}/{num_imgs} ...")
                    net, net_input = run(parameters=parameters,
                                        image=image,
                                        net=net,
                                        net_input=net_input)

                parameters['save_imgs'] = True
                if image_sequence.back_and_forth:
                    print("Iterating through images in reverse order...")
                    for i, image in enumerate(reversed(image_sequence.images[:-1])):
                        print(f"Fitting net to image number {num_imgs-i-1}" +
                              f"(finished ({i+1}/{num_imgs}))")
                        net, net_input = run(parameters=parameters,
                                            image=image,
                                            net=net,
                                            net_input=net_input)
        else:
            for i, image in enumerate(image_sequence.images):
                for r in range(parameters['image']['num_runs']):
                    print(f"Fitting net to image {i+1}/{num_imgs} (run {r+1}) ...")
                    run(parameters, image)

        del image_sequence
