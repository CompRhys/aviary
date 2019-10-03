# RooSt

**R**epresentati**o**n Learning fr**o**m **St**oichiometry

## Premise

In materials discovery applications often we know the composition of trial materials but have little knowledge about the structure.

Most current SOTA results within the field of machine learning for materials discovery are reliant on already knowing the structure of the material. Whilst this can help interpolate within given systems it means that our ML applications must be intrinsically dependant on costly structure prediction calculations to first find structures for trial composs to extended periodic crystals although notable issues exist in how to represent dopants in such graphs and how much of the structure we can take as prior knowledge.

In this work, we show that we can leverage a message-passing neural network to tackle materials agnostic tasks with increase efficacy. This is done by reformulating the problem itions. Whilst this would not be prohibitive it adds an additional level of complexity.

In a similar vein to other materials agnostic platforms (i.e. [MAGPIE](http://oqmd.org/static/analytics/magpie/doc)) this model aims to be able to determine properties (to reasonable accuracy) based solely on the composition, i.e. stoichiometric formula of a material.

Recent progress in the study of small molecules has built upon the use of message-passing neural networks. Similar attempts are already progressing in the application of such networkas a dense graph and then learning perturbations to our atomic representations to allow for an end-to-end systematically improvable descriptor. 


## Example Use
To run the code install the necessary dependencies and then simply run `python train.py`. The default task is on the experimental bandgap data set referenced in the paper. The code has been set up to allow most things to be controlled using argparse flags, the flags are listed under `roost/data.py`. 

Note that if no validation set is given the model will evaluate the test set performance after each epoch, **do not use this metric for early stopping**.

## Cite This Work

If you use this code please cite our work for which this model was built:

[Predicting materials properties without crystal structure: Deep representation learning from stoichiometry](https://arxiv.org/abs/1910.00617)

## Disclaimer

This is research code shared without support or any guarantee on its quality. However, if you do find an error please submit a pull request or raise an issue and I will try my best to solve it.

