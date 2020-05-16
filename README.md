# RooSt

**R**epresentati**o**n Learning fr**o**m **St**oichiometry

## Premise

In materials discovery applications often we know the composition of trial materials but have little knowledge about the structure.

Many current SOTA results within the field of machine learning for materials discovery are reliant on knowledge of the structure of the material. This means that such models can only be applied to systems that have undergone structural characterisation. As structural characterisation is a time-consuming process whether done experimentally or via the use of ab-initio methods the use of structures as our model inputs is a prohibitive bottleneck to many materials screening applications we would like to pursue.

One approach for avoiding the structure bottleneck is to develop models that learn from the stoichiometry alone. In this work, we show that via a novel recasting of how we view the stoichiometry of a material we can leverage a message-passing neural network to learn materials properties whilst remaining agnostic to the structure. The proposed model exhibits increase sample efficiency compared to more widely used descriptor-based approaches. This work draws inspiration from recent progress in using graph-based methods for the study of small molecules and crystalline materials.

## Environment Setup

```bash
conda create --name roost python=3.7.6
conda activate roost
pip install torch==1.5.0+${CUDA} torchvision==0.6.0+${CUDA} \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==latest+${CUDA} \
    -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install scikit-learn tqdm pandas tensorboard
```

`${CUDA}` Should be replaced by either `cpu`, `cu92`, `cu101` or `cu102` depending on your system CUDA version.

You may encounter issues getting the correct installation of either `PyTorch` or `PyTorch_Scatter` for your system requirements if so please check the following pages [PyTorch](https://pytorch.org/get-started/locally/), [PyTorch-Scatter](https://github.com/rusty1s/pytorch_scatter)

## Example Use

```python train.py --train --evaluate```

Runs the default task, this is on the experimental bandgap data of Zhou et al. (See folder for reference). This default task has been set up to work out of the box without any changes and to give a flavour of how the model can be used. 

If you want to use your own data set on a regression task this can be done with:

```python train.py --data-path /path/to/your/data/data.csv --train```

You can then test your model with:

```python train.py --test-path /path/to/testset.csv --evaluate```

The model takes input in the form csv files with materials-ids, composition strings and target values as the columns.

| material-id |  composition |  target | 
|-------------|--------------|---------| 
| foo-1       | Fe2O3        | 2.3     | 
| foo-2       | La2CuO4      | 4.3     | 

Basic hints about more advanced use of the model (i.e. classification, robust losses, ensembles, tensorboard logging etc..)
are available via the command:

```python train.py --help```

This will output the various command-line flags that can be used to control the code. 


## Cite This Work

If you use this code please cite our work for which this model was built:

[Predicting materials properties without crystal structure: Deep representation learning from stoichiometry](https://arxiv.org/abs/1910.00617)

``` 
@article{goodall2019predicting,
  title={Predicting materials properties without crystal structure: Deep representation learning from stoichiometry},
  author={Goodall, Rhys EA and Lee, Alpha A},
  journal={arXiv preprint arXiv:1910.00617},
  year={2019}
}
```

## Work Using Roost

[A critical examination of compound stability predictions from machine-learned formation energies](https://arxiv.org/abs/2001.10591)

If you have used Roost in your work please contact me and I will add your paper here.

## Acknowledgements

The open-source implementation of `cgcnn` available [here](https://github.com/txie-93/cgcnn) provided significant initial inspiration for how to structure this code-base.

## Disclaimer

This is research code shared without support or any guarantee on its quality. However, please do submit issues and pull requests or raise an issue and I will try my best to solve it.
