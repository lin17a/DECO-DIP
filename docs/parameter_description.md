---
layout: default
title: Parameters
nav_order: 3
---

# Parameters

A short description for every parameter and their default values can be found in [default_parameters.yaml](https://github.com/lin17a/DECO-DIP/blob/main/default_parameters.yaml).

Parameters are given in yaml format. There is one obligatory parameter, namely the path to the noisy image, that you want to denoise and deconvolve. So, the simplest parameter config looks like this:

```
image:
    path: path/to/your/image
```

The parameters are split into 7 groups, which are explained in the following sections. Example configs can be found in [example_configs](https://github.com/lin17a/DECO-DIP/tree/main/example_configs).

## 1. Image
The parameters in the `image` block define the input image(s).

- `path` : path/to/your/image (obligatory)
    
    Path to the noisy image.

- `path_gt` : path/to/the/gound/truth/image

    You can provide a ground truth image (if available) to calculate the real loss between output and ground truth.
- `crop_region` : List[int]

    A crop region can be configured to crop all images to a specific region. The format is a list with pixel positions: [left, top, right, bottom].

- `frame_range`: List[int]

    Define a frame range to exclude some images from the reconstruction (in case there are multiple input images). Format: [first, last].Defaults to all frames.

- `num_iter` : int

    Define a number of repititions (runs are repeated for same image and same parameters). This can produce different results, because there is randomness in the process.

## 2. Time Series
`time series` parameters define the learning process when the input are time-dependent images.
- `is_series`: bool
    
    The images can either be learned separately (`is_series` = `false`) or the time information can be used (`is_series` = `true`). In the latter case, the network for the first frame is trained as usual and for all subsequent frames the network is initialized with the fittet network from the previous frame.

- `learning_rate`: float

    Learing rate that is used for all frames except the first in case `is_series` is `true`

- `num_iter` : int

    Number of iterations that is used for all frames except the first in case `is_series` is `true`

- `back_and_forth` : bool

    If `is_series` is `true` and this parameter is set to `true`, the program iterates over the images in the reverse order after the normal iteration in chronological order. In this case the network from the last frame, that was optimized during the forward iteration, is used as initialization for the previous frame. This is repeated for every frame in anti-chronological order.

## 3. Net
`net` parameters define the network architecture and training process.

For further information see [default_parameters.yaml](https://github.com/lin17a/DECO-DIP/blob/main/default_parameters.yaml) and [DIP](https://github.com/DmitryUlyanov/deep-image-prior).

## 4. Superresolution
`superresolution` parameters configure a superresolution output image. This feature is not enabled per default.

For further information see [default_parameters.yaml](https://github.com/lin17a/DECO-DIP/blob/main/default_parameters.yaml) and [DIP](https://github.com/DmitryUlyanov/deep-image-prior).

## 5. Loss

`loss` parameters define the loss function.

`loss_main` can be one of `mse`, `psf` or `rl`.

- `mse` (Loss function of the original DIP): mean squared error between output image and noisy image

    $$E(x, x_0)=\|f_{\theta}(z)-x_0\|^2$$, 

    where $$f_{\theta}(z)$$ is the network output and $$x_0$$ is the noisy image.

- `psf`: mean squared error between output image convoluted with the point 
spread function (PSF) and noisy image:

    $$E(x, x_0) = \| f_{\theta}(z) * h - x_0 \|^2 $$, 

    where $$f_{\theta}(z)$$ is the network output, $$h$$ is the PSF and $$x_0$$ is the noisy image.

- `rl`: Richardson-Lucy functional:

    $$E(x, x_0) = \sum [(h * x) - x_0 \cdot \log(h * x)]$$

        
### Combination of loss functions
You can also combine two of the loss functions by using `loss_main` and `loss2`:

$$E(x, x_0) = \text{loss_fact2} \cdot \text{loss2} + (1-\text{loss_fact2}) \cdot \text{loss_main}$$

`loss_fact2` defines the ratio of the two loss functions. 

`loss_fact2` can be increased/decreased with every iteration with the parameter loss2_incr_fact:

$$\text{loss2_fact} := \text{loss2_fact} + (\text{loss2_incr_fact} \cdot \text{epoch})$$

### Explicit regularization

You can use one of these two norms as an explicit regularizer (default is none):
- `tv`: Total-Variation Norm

    $$R(x) = \sum |\nabla x| $$ 

- `tm`: Tikhonov-Miller Norm

    $$R(x) = \sum |\nabla x|^2 $$ 

The influence regularizer can be configured with the parameter `regularizer_fact`:

$$r(x) = \text{regularizer_fact} \cdot R(x)$$

Analogous to the regulation of loss2, you can also change the `regularizer_fact` with every iteration:

$$\text{regularizer_fact} := \text{regularizer_fact} +  \text{regularizer_incr_fact} \cdot \text{epoch} $$

The regularization term $$r(x)$$ is added to the loss, so that this is the **full loss function**:

$$E(x, x_0) = \text{loss_fact2} \cdot \text{loss2} + (1-\text{loss_fact2}) \cdot \text{loss_main} + \text{regularizer_fact}\cdot R(x)$$


## 6. PSF
`psf` parameters define the point spread function (PSF). 

For further information see [default_parameters.yaml](https://github.com/lin17a/DECO-DIP/blob/main/default_parameters.yaml) and  [TDEntropyDeconvolution](https://ipmi-icns-uke.github.io/TDEntropyDeconvolution/General/2-usage.html#point-spread-function)

## 7. Saving and Logging

These parameter group is called `save_and_log` and defines where the results are stored and how much output is produced.

- `orig_img_path` : path/to/your/orig/image_png

    Define where to save result images as png to show it in the html file described below. Defaults to `results/imgs/orig`

- `denoised_img_path` : path/to/your/result/image

    Define where to save result images. Defaults to `results/imgs/denoised`.

- `image_html`: path/to/your/result/html

    You can store images with their corresponding output image, all used parameters and the resulting losses in an html file. Defaults to `results/results.html`.

- `csv_path` : path/to/your/csv

    You can store parameters and the resulting losses in a csv file. Defaults to null (don't save).

- `tensorboard` : bool 
    
    You can use tensorboard to log the loss and save an image for every checkpoint (checkpoint interval is defined in net: checkpoint_interval).

- `tensorboard_logdir`: your/tensorboard/logdir

    Define where to save tensorboard files if `tensorboard` is ``true``.

- `verbosity` : int in [0,1]

    Define the amount of prints. 
