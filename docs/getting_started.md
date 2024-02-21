---
layout: home
title: Getting Started
nav_order: 2
---

# Getting Started

## Installation

First clone the repo and initialize submodules by running 
```bash
git clone --recurse-submodules https://github.com/lin17a/DECO-DIP
```
in the command line.

If you already cloned it and forgot to inialize the submodules, run
```bash
git submodule update --init --recursive
```

Then apply patch files for the deep-image-prior repo with
```bash
(cd dip && git apply ../dip.patch)
```

and set up the python virtual environment with
```bash
python -m venv ./path/to/new/virtual/environment
source ./path/to/new/virtual/environment/bin/activate
pip install -r requirements.txt
```

## Usage

Start the program with
```
./main.py --param_path parameters.yaml
```
With the parameter `param_path` you can specify a yaml file containing the parameters. Default is `./parameters.yaml`.
If you want to run the program with more than one parameter file, you can specify a folder containing multiple parameter files. All yaml files in that folder are processed successively.

Example config files can be found in [./example_configs](https://github.com/lin17a/DECO-DIP/tree/main/example_configs) and default parameters are stored in [./default_parameters.yaml](https://github.com/lin17a/DECO-DIP/blob/main/default_parameters.yaml).

For detailed parameter descriptions see [./default_parameters.yaml](https://github.com/lin17a/DECO-DIP/blob/main/default_parameters.yaml) and the [parameter section](/parameter_description.html).