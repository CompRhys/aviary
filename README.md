# RooSt

**R**epresentati**o**n Learning fr**o**m **St**oichiometry

## Premise

In materials discovery applications often we know the composition of trial materials but have little knowledge about the structure.

Most current SOTA results within the field of machine learning for materials discovery are reliant on already knowing the structure of the material this means that such ML applications are intrinsically dependant on costly structure prediction calculations or can only be applied to systems that have undergone experimental characterisation. The use of structures is a prohibative bottleneck to many materials screening applications we would like to pursue.

To avoid the structure bottle neck we want to develop models that learn from the stiochiometry alone. In this work, we show that we can leverage a message-passing neural network to tackle materials agnostic tasks with increase efficacy compared to more widely used descriptor based approaches. This work draws inspiration from recent progress in the study of small molecules that has made use of very similiar neural network architectures. 

## Example Use
To run the code install the necessary dependencies and then simply run `python train.py`. The default task is on the experimental bandgap data set referenced in the paper. The code has been set up to allow most things to be controlled using argparse flags, the flags are listed under `roost/data.py`. 

Note that if no validation set is given the model will evaluate the test set performance after each epoch, **do not use this metric for early stopping**.

## Cite This Work

If you use this code please cite our work for which this model was built:

[Predicting materials properties without crystal structure: Deep representation learning from stoichiometry](https://arxiv.org/abs/1910.00617)

## Work Using Roost

[A critical examination of compound stability predictions from machine-learned formation energies](https://arxiv.org/abs/2001.10591)

## Disclaimer

This is research code shared without support or any guarantee on its quality. However, if you do find an error please submit a pull request or raise an issue and I will try my best to solve it.

