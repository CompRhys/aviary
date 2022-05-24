<h1 align="center">Aviary</h1>

<h4 align="center">

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/comprhys/aviary?label=Repo+Size)](https://github.com/comprhys/aviary/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/comprhys/aviary?label=Last+Commit)](https://github.com/comprhys/aviary/commits)
[![Tests](https://github.com/CompRhys/aviary/actions/workflows/test.yml/badge.svg)](https://github.com/CompRhys/aviary/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/CompRhys/aviary/main.svg)](https://results.pre-commit.ci/latest/github/CompRhys/aviary/main)

</h4>

The aviary contains:

* <a href="https://colab.research.google.com/github/CompRhys/aviary/blob/main/examples/colab/Roost.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Roost In Colab" valign="middle"></a> &nbsp;-&nbsp; `roost`,
* <a href="https://colab.research.google.com/github/CompRhys/aviary/blob/main/examples/colab/Wren.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Wren In Colab" valign="middle"></a> &nbsp;-&nbsp; `wren`,
* `cgcnn`.

The aim is to contain multiple models for materials discovery under a common interface

## `conda` Installation

To use `aviary` you need to create an environment with the correct dependencies. The easiest way to get up and running is to use `anaconda`.
A `cudatoolkit=11.1` environment is provided in `environment-gpu-cu111.yml` allowing a working environment to be created with:

```bash
conda env create -f environment-gpu-cu111.yml
```

If you are not using `cudatoolkit=11.1` or do not have access to a GPU this setup will not work for you. If so please check the following pages [PyTorch](https://pytorch.org/get-started/locally), [PyTorch-Scatter](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for how to install the core packages.

The code was developed and tested on Linux Mint 19.1 Tessa. It should work with other operating systems but it has not been tested for such use.

Once you have setup an environment with the correct dependencies you can install `aviary` using the following commands from the top of the directory:

```bash
conda activate aviary
python setup.py sdist
pip install -e .
```

This will install the library in an editable state allowing for advanced users to make changes as desired.

## `pip` Installation

Aviary requires [`torch-scatter`](https://github.com/rusty1s/pytorch_scatter). To `pip install` it, make sure you replace `1.11.0` with your actual `torch.__version__` (`python -c 'import torch; print(torch.__version__)'`) and `cpu` with your CUDA version if applicable.

```sh
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cpu.html

pip install -U git+https://github.com/CompRhys/aviary  # install aviary itself

# or for an editable install
git clone https://github.com/CompRhys/aviary
pip install -e ./aviary
```

## Example Use

To test the input files generation and cleaning/canonicalization please run:

```sh
python examples/inputs/poscar2df.py
```

This script will load and parse a subset of raw POSCAR files from the TAATA dataset and produce the `datasets/examples/examples.csv` file used for the next example.
The raw files have been selected to ensure that the subset contains all the correct endpoints for the 5 elemental species in the `Hf-N-Ti-Zr-Zn` chemical system.
All the models used share can be run on the input file produced by this example code. To test each of the three models provided please run:

```sh
python examples/roost-example.py --train --evaluate --data-path examples/inputs/examples.csv --targets E_f --tasks regression --losses L1 --robust --epoch 10
```

```sh
python examples/wren-example.py --train --evaluate --data-path examples/inputs/examples.csv --targets E_f --tasks regression --losses L1 --robust --epoch 10
```

```sh
python examples/cgcnn-example.py --train --evaluate --data-path examples/inputs/examples.csv --targets E_f --tasks regression --losses L1 --robust --epoch 10
```

Please note that for speed/demonstration purposes this example runs on only ~68 materials for 10 epochs - running all these examples should take < 30s. These examples do not have sufficient data or training to make accurate predictions, however, the same scripts have been used for all experiments conducted.

## Cite This Work

If you use this code please cite the relevant work:

Predicting materials properties without crystal structure: Deep representation learning from stoichiometry. [[Paper]](https://doi.org/10.1038/s41467-020-19964-7) [[arXiv]](https://arxiv.org/abs/1910.00617)

```tex
@article{goodall2020predicting,
  title={Predicting materials properties without crystal structure: Deep representation learning from stoichiometry},
  author={Goodall, Rhys EA and Lee, Alpha A},
  journal={Nature Communications},
  volume={11},
  number={1},
  pages={1--9},
  year={2020},
  publisher={Nature Publishing Group}
}
```

Rapid Discovery of Novel Materials by Coordinate-free Coarse Graining. [[arXiv]](https://arxiv.org/abs/2106.11132)

```tex
@article{goodall2021rapid,
  title={Rapid Discovery of Novel Materials by Coordinate-free Coarse Graining},
  author={Goodall, Rhys EA and Parackal, Abhijith S and Faber, Felix A and Armiento, Rickard and Lee, Alpha A},
  journal={arXiv preprint arXiv:2106.11132},
  year={2021}
}
```

Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. [[Paper]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) [[arXiv]](https://arxiv.org/abs/1710.10324)

```tex
@article{xie2018crystal,
  title={Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties},
  author={Xie, Tian and Grossman, Jeffrey C},
  journal={Physical review letters},
  volume={120},
  number={14},
  pages={145301},
  year={2018},
  publisher={APS}
}
```

## Disclaimer

This research code is provided as-is. We have checked for potential bugs and believe that the code is being shared in a bug-free state.
