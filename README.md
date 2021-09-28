# Aviary

The aviary contains `roost`, `wren` and `cgcnn`. The aim is to contain multiple models for materials discovery under a common interface

## Environment Setup

To use `aviary` you need to create an environment with the correct dependencies. The easiest way to get up and running is to use `anaconda`.
A `cudatoolkit=11.1` environment file is provided `environment-gpu-cu111.yml` allowing a working environment to be created with:

```bash
conda env create -f environment-gpu-cu111.yml
```

If you are not using `cudatoolkit=11.1` or do not have access to a GPU this setup will not work for you. If so please check the following pages [PyTorch](https://pytorch.org/get-started/locally/), [PyTorch-Scatter](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for how to install the core packages and then install the remaining requirements as detailed in `requirements.txt`.

The code was developed and tested on Linux Mint 19.1 Tessa. The code should work with other Operating Systems but it has not been tested for such use.

## Aviary Setup

Once you have set up an environment with the correct dependencies you can install `aviary` using the following commands from the top of the directory:

```bash
conda activate aviary
python setup.py sdist
pip install -e .
```

This will install the library in an editable state allowing for advanced users to make changes as desired.

## Example Use

To test the input files generation and cleaning/canonicalization please run:

```sh
python examples/inputs/poscar2df.py
```

This script will load and parse a subset of raw POSCAR files from the TAATA dataset and produce the `data/datasets/examples/examples.csv` file used for the next example.
The raw files have been selected to ensure that the subset contains all the correct endpoints for the 5 elemental species in the `Hf-N-Ti-Zr-Zn` chemical system.
All the models used share can be run on the input file produced by this example code. To test each of the three models provided please run:

```sh
python examples/roost-example.py --train --evaluate --data-path data/datasets/examples/examples.csv --targets E_f --tasks regression --losses L1 --robust --epoch 10
```
```sh
python examples/wren-example.py --train --evaluate --data-path data/datasets/examples/examples.csv --targets E_f --tasks regression --losses L1 --robust --epoch 10
```
```sh
python examples/cgcnn-example.py --train --evaluate --data-path data/datasets/examples/examples.csv --targets E_f --tasks regression --losses L1 --robust --epoch 10
```

Please note that for speed/demonstration purposes this example runs on only ~68 materials for 10 epochs - running all these examples should take < 30s. These examples do not have sufficient data or training to make accurate predictions, however, the same scripts in the examples folder were used for all experiments conducted.

## Reproduction of Figures

Three addition sets of data are provided in the `datasets.tar.gz`, `manuscript-results.tar.gz` and `pre-trained.tar.gz`. Please unzip these and place the contents in `data/datasets`, `results/` and `models` respectively.

### Figure 2

Figure 2 is a direct comparison of different classes of model. We compare results from `roost`, `wren` on pre-relaxation inputs, and `cgcnn` on both pre and post-relaxation inputs. This figure demonstrates how the proposed Wyckoff representation can differentiate polymorphs and can be used to screen unrelaxed inputs without significant degradation in performance.

The data needed to reproduce Figure 2 of the manuscript are provided under `data/datasets/taata`.
The `cds3/taata-expt.txt` folder provides the `slurm` submit scripts and CLI inputs for all experiments as well as the CLI inputs to generate the results files.
To facilitate the reproduction of the Figures without having to re-run the ensembles for the different models we provide the necessary results files under `results/manuscript/` and pre-trained models under `models/pre-trained/`.
The Figure can be generated by running the `examples/plots/multi-pred-test-taata-hull.py` script.

### Figure 3

Figure 3 looks at whether we can use a model trained on the complement of a chemical system (i.e. all available data apart from data in that system) to identify the low lying phases more quickly than via the standard prototype-based high-throughput screening approach. To train the model we take the MP data set and exclude all tertiary systems in the `Hf-N-Zn`, `N-Ti-Zn` and `N-Zn-Zr` chemical systems. The enrichment plot shows that the model is capable of zero-shot generalisation to new chemical systems. Importantly we see that the model enriches lower energy structures more strongly which is exactly the behaviour we would desire in practice.

The data needed to reproduce Figure 3 of the manuscript are provided under `data/datasets/chemsys`.
The `cds3/chemsys-expt.txt` folder provides the `slurm` submit scripts and CLI inputs for all experiments as well as the CLI inputs to generate the results files.
To facilitate the reproduction of the Figures without having to re-run the ensembles for the different models we provide the necessary results files under `results/manuscript/` and pre-trained models under `models/pre-trained/`.
The Figure can be generated by running the `examples/plots/enrich-taata.py` script.

### Figures 4 and 5

Figures 4 and 5 look at a simulated materials discovery campaign. Here we make use of two independent data sources, the MP and WBM data sets, that have been prepared using the same DFT settings. The fact that they use the same DFT settings ensures that there is no covariate shift, this can be an issue when using data from different sources. Figure 4 highlights the strong recall that the Wren model achieves ~75%. Figure 5 highlights the shortcomings of using structure-based models to screen pre-relaxation inputs seen in the low accuracy of the model.

The data needed to reproduce Figures 4 and 5 of the manuscript are provided under `data/datasets/mp` and `data/datasets/wbm`.
The `cds3/wbm-expt.txt` folder provides the `slurm` submit scripts and CLI inputs for all experiments as well as the CLI inputs to generate the results files.
To facilitate the reproduction of the Figures without having to re-run the ensembles for the different models we provide the necessary results files under `results/manuscript/` and pre-trained models under `models/pre-trained/`.
For Figures 4 and 5 as we need to compute the distance to the convex hull of the MP training set we also provide this data pre-computed for the WBM splits - calculation of the energy of the hull at each of the compositions in the WBM data set requires the construction of ~48k convex hulls to get the energies for candidates overlapping the MP chemical systems, for candidates that do not overlap the MP chemical system the energy is calculated using a linear algebra approach calculated using `pymatgen.analysis.phase_diagram._get_slsqp_decomp`.
Figure 4 can be generated by running the `examples/plots/hist_clf.py` script and Figure 5 can be generated by running the `examples/plots/moving-mae-multi.py` script.

### Applying The Combined MP+WBM Model To Your Own Data

In order to apply the model to your own data the easiest approach will be to edit the `examples/inputs/poscar2df.py` script based on your needs. If the case that the target variable is unknown the model will require the column to be filled with a placeholder i.e. `42`. After unpacking the pre-trained models and placing the `pre-trained` directory inside the `models` directory the pre-trained comb models can then be called using:

```sh
python examples/wren-example.py --data-id pre-trained/comb --ensemble 10 --evaluate --data-path data/datasets/examples/examples.csv --test-path <path/to/your/input.csv> --targets E_f --tasks regression --losses L1 --robust
```

The `data-path` to `data/datasets/examples/examples.csv` is used as a placeholder that is needed given the current structure of the code.

## Notes

Some of the data files were originally prepared using an older input format, we have converted these inputs into the newer format based on the AFLOW prototype schema. However, in datasets that have been converted the Pearson symbol is filled by the string "pearson" as a placeholder. The Pearson symbol is not used by the __Wren__ model therefore this does not affect model performance.

## Disclaimer

This research code is provided as-is. We have checked for potential bugs and believe that the code is being shared in a bug-free state. As this is an archive version we will not be able to amend the code to fix bugs/edge-cases found at a later date. However, this code will likely continue to be developed at the location described in the metadata.
