import magpy
import numpy as np
import torch
from scipy.special import gamma


def magpy_featurise(compositions, features):
    '''
    Use Magpy package to featurise the input compositions

    Parameters
    ----------
    compositions: list of strings shape N0
        list of compositions of interest
    features: list of Strings of length 
        list of features to use as atom descriptors
    Returns
    -------
    df
    '''
    elements, weights = magpy.parse_input(compositions)
    df_list = magpy.look_up(elements, weights, features=features)

    return df_list

def crystal_lists(df_list):
    '''
    Parameters
    ----------
    df: list of pandas.DataFrame of length N0
        DataFrame containing the features for different atoms
    Returns
    -------
    nbr_list: list of Integers of length N
        returns a list of which other atoms in the dataset 
        are related to a given atom.
    '''
    nbr_list = []
    cry_list = []
    max_nbrs = 0
    crystal_start = 0
    for df in df_list:
        cry_list.append(list(range(crystal_start,crystal_start+len(df))))
        crystal_elements = set(range(crystal_start,crystal_start+len(df)))
        for j in range(df.shape[0]):
            nbrs = list(crystal_elements.difference(set([crystal_start+j])))
            nbr_list.append(nbrs)
        crystal_start += df.shape[0]
        if len(df)-1 > max_nbrs:
            max_nbrs = len(df)-1
    for nbr in nbr_list:
        nbr += [0]*(max_nbrs-len(nbr))
    # nbr_list = torch.cat(nbr_list, dim=0)
    return cry_list, nbr_list

def edge_embeddings():
    '''
    Use the idea of van Arkel-Ketelaar triangle and then 
    gamma distribution to define a number of electrons
    and the expected type of bond between two nodes.

    if a bond is ionic we will only count it if 

    Parameters
    ----------
    weights: Variable(numpy.array) shape (M,)  
        composition weights of different elements
    valence: (N,) vector giving the valence of different elements

    Returns
    ---------
    edges: Variable(numpy.array) shape (N, M, len_fea)
        return

    '''

    eff_prob = gamma(5)/gamma(6)

    return eff_prob

comp = ['LaCu2O4', 'K2MgO4', 'La1.85Sr0.1Y0.05Cu0.9Ca0.1O4', 'NaCl', 'YLaNaB2Cu3O6.7F0.3']
feat = ['CovalentRadius', 'Polarizability', 'Electronegativity', 'ElectronAffinity', 'FirstIonizationEnergy']


def main():
    data = magpy_featurise(comp, feat)

    
    print(crystal_lists(data))
    print(edge_embeddings())

main()

