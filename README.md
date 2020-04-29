# RooSt

**R**epresentati**o**n Learning fr**o**m **St**oichiometry

## Premise

In materials discovery applications often we know the composition of trial materials but have little knowledge about the structure.

Most current SOTA results within the field of machine learning for materials discovery are reliant on already knowing the structure of the material this means that such ML applications are intrinsically dependant on costly structure prediction calculations or can only be applied to systems that have undergone experimental characterisation. The use of structures is a prohibative bottleneck to many materials screening applications we would like to pursue.

To avoid the structure bottle neck we want to develop models that learn from the stiochiometry alone. In this work, we show that we can leverage a message-passing neural network to tackle materials agnostic tasks with increase efficacy compared to more widely used descriptor based approaches. This work draws inspiration from recent progress in the study of small molecules that has made use of very similiar neural network architectures. 

## Environment Setup

```bash
conda create --name roost python=3.7.6
conda activate roost
pip install torch==1.5.0+${CUDA} torchvision==0.6.0+${CUDA} \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==latest+${CUDA} \
    -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install scikit-learn matplotlib tqdm pandas tensorboard
```

`${CUDA}` Should be replaced by either cpu, cu92, cu101 or cu102 depending on your system CUDA version.

You may encounter issues getting the the correct installation of either Pytorch or Pytorch_Scatter for your system requirements if so please check the following pages [Pytorch](https://pytorch.org/get-started/locally/), [Pytorch-Scatter](https://github.com/rusty1s/pytorch_scatter)

## Example Use

```python train.py```

Runs the default task is on the experimental bandgap data set referenced in the pre-print. This default task has been setup to work out of the box without any changes. 

If you want to tune the model the best way to get an understanding of how to run the code with other settings is to run the command:

```python train.py --help```

This will output the various commandline flags that can be used to control the code.

If you want to use your own data set this can be done with:

```python train.py --data-path /path/to/your/data/data.csv```

The model takes input in the form csv files with materials-ids, composition strings and target values as the columns.

| material-id |  composition |  target | 
|-------------|--------------|---------| 
| foo-1       | Fe2O3        | 2.3     | 
| foo-2       | La2CuO4      | 4.3     | 

Note that if no validation set is given the model will evaluate the test set performance after each epoch, **do not use this metric for early stopping**.

## Cite This Work

If you use this code please cite our work for which this model was built:

[Predicting materials properties without crystal structure: Deep representation learning from stoichiometry](https://arxiv.org/abs/1910.00617)

## Work Using Roost

If you have used Roost in your work please contact me and I will add your paper here.

[A critical examination of compound stability predictions from machine-learned formation energies](https://arxiv.org/abs/2001.10591)

## Disclaimer

This is research code shared without support or any guarantee on its quality. However, if you do find an error please submit a pull request or raise an issue and I will try my best to solve it.

