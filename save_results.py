"""This module provides functions to save the results of a DIP run as html and csv file."""
import re
import os
import csv
from datetime import datetime
from bs4 import BeautifulSoup

def save_results(image, parameters, final_loss_noisy, final_loss_gt, orig_noisy_difference):
    """put noisy and denoised image into a html file with the corresponding parameters and losses

    Args:
        image (fm_image.Image): image containing noisy and denoised image
        parameters (dict): parameters
        final_loss_noisy (dict): loss based on the noisy image and the denoised image
        final_loss_gt (dict): loss based on the ground truth and the denoised image
        orig_noisy_difference (dict): difference between the noisy image and the ground truth
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    image.save_denoised(current_time)

    html_path = parameters["save_and_log"]['image_html']
    if html_path:
        add_result_to_html(html_path,
                           image,
                           parameters,
                           current_time = current_time,
                           loss_dict_noisy = final_loss_noisy,
                           loss_dict_gt = final_loss_gt,
                           orig_noisy_difference_dict = orig_noisy_difference)

    csv_path = parameters["save_and_log"]['csv_path']
    if csv_path:
        noise_dict = parse_noise_params(image.base_file_name)

        params = {"gt_path" : image.path_gt,
                 "image_path" : image.save_file_noisy,
                 "gaussian noise" : noise_dict['g'],
                 "poisson noise" : noise_dict['p'],
                 "lr" : parameters['net']['learning_rate'], 
                 "num_iter" : parameters['net']['num_iter'],
                 "depth" : parameters['net']['num_scales'],
                 "main_loss" : parameters['loss']['loss_type_main'],
                 "loss2" : parameters['loss']['loss_type2'],
                 "loss2_fact" : parameters['loss'].get('loss2_fact', None),
                 "regularizer" : parameters['loss']['regularizer'],
                 "regularizer_fact" : parameters['loss'].get('regularizer_fact', None)}

        add_result_to_csv(csv_path, params, final_loss_noisy, final_loss_gt,
                          orig_noisy_difference, current_time)


def parse_noise_params(img_name):
    """parse the noise paramters from the file name

    Args:
        img_name (str): file name of the noisy image

    Returns:
        dict: dict containing p, g and psf parameters from the file name
    """
    p = 0
    g = 0

    p_pattern = re.compile("p([\d\.e\-\+]+)")
    p_result = p_pattern.search(img_name)
    if p_result:
        p = float(p_result.group(1))

    g_pattern = re.compile("g([\d\.e\-\+]+)")
    g_result = g_pattern.search(img_name)
    if g_result:
        g = float(g_result.group(1))

    return {"p" : p, "g" : g}


def add_result_to_csv(csv_path, params, loss_dict_noisy, loss_dict_gt, orig_noisy_difference_dict,
                    current_time):
    """write one row in a csv with some parameters and the corresponding losses

    Args:
        csv_path (str): path to the csv file
        params (dict): params to add to the csv
        loss_dict_noisy (dict): loss between noisy and denoised image
        loss_dict_gt (dict): loss between denoised image and ground truth
        orig_noisy_difference_dict (dict): difference between noisy image and ground truth
        current_time (str): string containing the current time
    """
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            param_headers = [param for param in params] #["poisson", "gauss", "psf"]
            csv_writer.writerow(
                param_headers +
                [f"{loss_noisy}_out_noisy" for loss_noisy in loss_dict_noisy] +
                [f"{loss_gt}_out_gt" for loss_gt in loss_dict_gt] +
                [f"{loss_noisy_gt}_noisy_gt" for loss_noisy_gt in orig_noisy_difference_dict] +
                ["time"])

    params = [params[param] for param in params]
    results_out_noisy = [loss_dict_noisy[loss_noisy] for loss_noisy in loss_dict_noisy]
    results_out_gt = [loss_dict_gt[loss_gt] for loss_gt in loss_dict_gt]
    results_noisy_gt = [orig_noisy_difference_dict[loss_noisy_gt]
                        for loss_noisy_gt in orig_noisy_difference_dict]

    with open(csv_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(params + results_out_noisy + results_out_gt + results_noisy_gt +
                            [current_time])


def add_result_to_html(html_path, image, parameters, current_time = "",
                       loss_dict_noisy = None, loss_dict_gt = None,
                       orig_noisy_difference_dict = None):
    """add noisy image, denoised image, the used parameter and the resulting losses to an html file

    Args:
        html_path (str): path to the html file
        image (fm_image): fm_image containing noisy and denoised image
        parameters (dict): parameters used
        current_time (str, optional): string containing the current time. Defaults to "".
        loss_dict_noisy (dict, optional): loss between noisy and denoised image. Defaults to {}.
        loss_dict_gt (dict, optional): loss between denoised image and ground truth. Defaults to {}.
        orig_noisy_difference_dict (dict, optional): difference between noisy image and 
            ground truth. Defaults to {}.
    """

    if os.path.exists(html_path):
        soup = BeautifulSoup(open(html_path), 'html.parser')
    else:
        soup = BeautifulSoup(open("result_html_template.html"), 'html.parser')

    # add original and denoised image
    orig_img_col = soup.new_tag('td')
    orig_img_col.append(soup.new_tag('img',
                                     src = os.path.abspath(image.save_file_noisy),
                                     style = "display:block; max-height:100%; max-width:100%"))

    new_img_col = soup.new_tag('td')
    new_img_col.append(soup.new_tag('img',
                                    src = os.path.abspath(image.save_file_denoised),
                                    style = "display:block; max-height:100%; max-width:100%"))

    # add parameters
    param_col = soup.new_tag('td')

    param_col.append(soup.new_string(f"file_name : {image.base_file_name}"))
    for param_cat in parameters:
        param_col.append(soup.new_tag('br'))
        if parameters[param_cat]:
            for param in parameters[param_cat]:
                param_col.append(soup.new_string(f"{param} : {parameters[param_cat][param]}"))
                param_col.append(soup.new_tag('br'))
    param_col.append(soup.new_tag('br'))
    param_col.append(soup.new_tag('br'))

    # add current time
    comment_col = soup.new_tag('td')
    comment_col.append(soup.new_string(f"added on {current_time}"))
    comment_col.append(soup.new_tag('br'))
    comment_col.append(soup.new_tag('br'))
    # add loss
    comment_col.append(soup.new_string("loss compared to noisy image:"))
    comment_col.append(soup.new_tag('br'))
    if loss_dict_noisy:
        for loss in loss_dict_noisy:
            comment_col.append(soup.new_string(f"{loss} : {loss_dict_noisy[loss]}"))
            comment_col.append(soup.new_tag('br'))
    comment_col.append(soup.new_tag('br'))
    comment_col.append(soup.new_string("loss compared to ground truth:"))
    comment_col.append(soup.new_tag('br'))
    if loss_dict_gt:
        for loss in loss_dict_gt:
            comment_col.append(soup.new_string(f"{loss} : {loss_dict_gt[loss]}"))
            comment_col.append(soup.new_tag('br'))
    comment_col.append(soup.new_tag('br'))
    comment_col.append(soup.new_string("difference between noisy image and ground truth:"))
    comment_col.append(soup.new_tag('br'))
    if orig_noisy_difference_dict:
        for loss in orig_noisy_difference_dict:
            comment_col.append(soup.new_string(f"{loss} : {orig_noisy_difference_dict[loss]}"))
            comment_col.append(soup.new_tag('br'))

    new_row = soup.new_tag('tr')
    new_row.append(orig_img_col)
    new_row.append(new_img_col)
    new_row.append(param_col)
    new_row.append(comment_col)

    if os.path.exists(html_path):
        last_row = soup.find_all("tr")[-1]
        last_row.insert_after(new_row)
    else:
        tbody = soup.find_all("tbody")[-1]
        tbody.append(new_row)

    with open(html_path, "w") as file:
        file.write(str(soup.prettify()))
