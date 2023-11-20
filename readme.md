# DECO-DIP

## About

DECO-DIP denoises and deconvolves microscopy images. It is based on Deep image prior ([DIP](https://github.com/DmitryUlyanov/deep-image-prior)) [1] and does not require any training data. This version of DIP provides a new loss function, that includes an additional term to model the forward model of the imaging process. Additionally, the time dependence of image series can be used.

Synthetic test data can be created using the code from [TDEntropyDeconvolution](https://github.com/IPMI-ICNS-UKE/TDEntropyDeconvolution/), which is a submodule of this repo [2].

## Usage

1. Clone the repo and initialize submodules by running 
    ```bash
    git clone --recurse-submodules https://github.com/lin17a/DECO-DIP
    ```
    in the command line.
    If you already cloned it and forgot to inialize the submodules, run:
    ```bash
    git submodule update --init --recursive
    ```

2. Apply patch files for the deep-image-prior repo:
    ```bash
    (cd dip && git apply ../dip.patch)
    ```

3. Set up the python virtual environment:
    ```bash
    python -m venv ./path/to/new/virtual/environment
    source ./path/to/new/virtual/environment/bin/activate
    pip install -r requirements.txt
    ```

4. Run the program. With the parameter param_file you can specify a yaml file containing the parameters. Default is ./parameters.yaml.
    ```
    ./main.py --param_file parameters.yaml
    ```

    Example config files can be found in ./example_configs and default parameters are stored in [./default_parameters.yaml](./default_parameters.yaml).

    For detailed parameter descriptions see [./default_parameters.yaml](./default_parameters.yaml) and the [./docs](./docs) folder.

## References

[1] Ulyanov, D., Vedaldi, A., & Lempitsky, V. 2020. "Deep Image Prior". International Journal of Computer Vision 128 (7): 1867–88. https://doi.org/10.1007/s11263-020-01303-4.


[2] L. Woelk, S. A. Kannabiran, V. Brock, Ch. E. Gee, Ch. Lohr, A. H. Guse, B. Diercks, and R. Werner. 2021. "Time-Dependent Image Restoration of Low-SNR Live Cell Ca2+ Fluorescence Microscopy Data". International Journal of Molecular Sciences 22 (21): 11792. https://doi.org/10.3390/ijms222111792.

