<h1 align="center">Aviary</h1>

<h4 align="center">

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/comprhys/aviary?label=Repo+Size)](https://github.com/comprhys/aviary/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/comprhys/aviary?label=Last+Commit)](https://github.com/comprhys/aviary/commits)
[![Tests](https://github.com/CompRhys/aviary/actions/workflows/test.yml/badge.svg)](https://github.com/CompRhys/aviary/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/CompRhys/aviary/main.svg)](https://results.pre-commit.ci/latest/github/CompRhys/aviary/main)
[![This project supports Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)

</h4>

The aim of `aviary` is to contain multiple models for materials discovery under a common interface, over time we hope to add more models with a particular focus on coordinate-free deep learning models.

## Installation

Users can install `aviary` from source with

```sh
pip install -U git+https://github.com/CompRhys/aviary
```

or for an editable source install from a local clone:

```sh
git clone https://github.com/CompRhys/aviary
pip install -e ./aviary
```

## Example Use from CLI

To test the input files generation and cleaning/canonicalization please run:

```sh
python examples/inputs/poscar_to_df.py
```

This script will load and parse a subset of raw POSCAR files from the TAATA dataset and produce the `datasets/examples/examples.csv` and `datasets/examples/examples.json` files used for the next example.
For the coordinate-free `roost` and `wren` models where the inputs are easily expressed as strings we use CSV inputs.
For the structure-based `cgcnn` model we first construct `pymatgen` structures from the raw POSCAR files then determine their dictionary serializations before saving in a JSON format.
The raw POSCAR files have been selected to ensure that the subset contains all the correct endpoints for the 5 elemental species in the `Hf-N-Ti-Zr-Zn` chemical system.
To test each of the three models provided please run:

```sh
python examples/roost-example.py --train --evaluate --data-path examples/inputs/examples.csv --targets E_f --tasks regression --losses L1 --robust --epoch 10
```

```sh
python examples/wren-example.py --train --evaluate --data-path examples/inputs/examples.csv --targets E_f --tasks regression --losses L1 --robust --epoch 10
```

```sh
python examples/wrenformer-example.py --train --evaluate --data-path examples/inputs/examples.csv --targets E_f --tasks regression --losses L1 --robust --epoch 10
```

```sh
python examples/cgcnn-example.py --train --evaluate --data-path examples/inputs/examples.json --targets E_f --tasks regression --losses L1 --robust --epoch 10
```

Please note that for speed/demonstration purposes this example runs on only ~68 materials for 10 epochs - running all these examples should take < 30 sec. These examples do not have sufficient data or training to make accurate predictions, however, the same scripts were used for all experiments conducted as part of the development and publication of these models.
Consequently understanding these examples will ensure you can deploy the models as intended for your research.

## Notebooks

We also provide some notebooks that show more a more pythonic way to interact with the codebase, these examples make use of the TAATA dataset examined in the `wren` manuscript:

|                                                                                          |                                      |                                                                                                                              |
| ---------------------------------------------------------------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| **[Roost](https://github.com/CompRhys/aviary/blob/main/examples/notebooks/Roost.ipynb)** | [![Launch Codespace]][codespace url] | [![Open in Google Colab]](https://colab.research.google.com/github/CompRhys/aviary/blob/main/examples/notebooks/Roost.ipynb) |
| **[Wren](https://github.com/CompRhys/aviary/blob/main/examples/notebooks/Wren.ipynb)**   | [![Launch Codespace]][codespace url] | [![Open in Google Colab]](https://colab.research.google.com/github/CompRhys/aviary/blob/main/examples/notebooks/Wren.ipynb)  |

[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg
[Launch Codespace]: https://img.shields.io/badge/Launch-Codespace-darkblue?logo=github
[codespace url]: https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=411272553

## Cite This Work

If you use this code please cite the relevant work:

`roost` - Predicting materials properties without crystal structure: Deep representation learning from stoichiometry. [[Paper]](https://doi.org/10.1038/s41467-020-19964-7) [[arXiv]](https://arxiv.org/abs/1910.00617)

```bibtex
@article{goodall_2020_predicting,
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

`wren` - Rapid Discovery of Stable Materials by Coordinate-free Coarse Graining. [[Paper]](https://www.science.org/doi/10.1126/sciadv.abn4117) [[arXiv]](https://arxiv.org/abs/2106.11132)

```bibtex
@article{goodall_2022_rapid,
  title={Rapid discovery of stable materials by coordinate-free coarse graining},
  author={Goodall, Rhys EA and Parackal, Abhijith S and Faber, Felix A and Armiento, Rickard and Lee, Alpha A},
  journal={Science Advances},
  volume={8},
  number={30},
  pages={eabn4117},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```

`cgcnn` - Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. [[Paper]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) [[arXiv]](https://arxiv.org/abs/1710.10324)

```bibtex
@article{xie_2018_crystal,
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
