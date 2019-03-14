import magpy
import numpy as np
import torch
from scipy.special import gamma


def magpy_featurise(compositions, features):
    '''
    Use Magpy package to featurise the input compositions

    Parameters
    ----------
    orig_atom_fea_len: 

    Inputs
    ----------
    compositions: list of strings shape C
        list of compositions of interest
    features: list of strings of length orig_atom_fea_len 
        list of features to use as atom descriptors
    Returns
    ---------
    df: pandas.DataFrame of shape (C,orig_atom_fea_len)
        dataframe containing the features having been looked up for each
    '''
    elements, weights = magpy.parse_input(compositions)
    df_list = magpy.look_up(elements, weights, features=features)

    return df_list

def crystal_lists(df_list):
    '''
    Parameters
    ----------
    df: list of pandas.DataFrame of length C
        DataFrame containing the features for different atoms
    Returns
    -------
    nbr_list: list of Integers of length N
        returns a list of which other atoms in the dataset 
        are related to a given atom.
    '''
    self_list = []
    nbr_list = []
    crystal_atom_idx = []
    atom_bond_idx = []
    crystal_id = 0
    crystal_start = 0
    for df in df_list:
        crystal_elements = set(range(crystal_start,crystal_start+len(df)))
        for j in range(df.shape[0]):
            nbrs = list(crystal_elements.difference(set([crystal_start+j])))
            nbr_list += nbrs
            self_list += [crystal_start+j]*len(nbrs)
            atom_bond_idx.append()
        crystal_start += df.shape[0]

    return cry_list, nbr_list, self_list

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

    cluster, nbr, slf = crystal_lists(data)
    
    print(cluster)
    print(nbr)
    print(slf)

main()

