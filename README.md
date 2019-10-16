# RooSt

**R**epresentati**o**n Learning fr**o**m **St**oichiometry

## Premise

In materials discovery applications often we know the composition of trial materials but have little knowledge about the structure.

Most current SOTA results within the field of machine learning for materials discovery are reliant on already knowing the structure of the material this means that our ML applications are be intrinsically dependant on costly structure prediction calculations or experimental characterisation of structures before the models can be applied to new trial compositions.

To avoid the structure bottle neck we want to develop models that learn from the stiochiometry alone. In this work, we show that we can leverage a message-passing neural network to tackle materials agnostic tasks with increase efficacy. This work draws inspiration from recent progress in the study of small molecules that has made use of very similiar neural network architectures. 

## Example Use
To run the code install the necessary dependencies and then simply run `python train.py`. The default task is on the experimental bandgap data set referenced in the paper. The code has been set up to allow most things to be controlled using argparse flags, the flags are listed under `roost/data.py`. 

Note that if no validation set is given the model will evaluate the test set performance after each epoch, **do not use this metric for early stopping**.

## Cite This Work

If you use this code please cite our work for which this model was built:

[Predicting materials properties without crystal structure: Deep representation learning from stoichiometry](https://arxiv.org/abs/1910.00617)

## Disclaimer

This is research code shared without support or any guarantee on its quality. However, if you do find an error please submit a pull request or raise an issue and I will try my best to solve it.

