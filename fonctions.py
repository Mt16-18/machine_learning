import os #use to open the file "data"
import numpy as np # use for matrix
import random #use to plot label a random place
import matplotlib . pyplot as plt #use to plot result
from scipy import linalg #use for eigenvalues and ... decomposition
import csv #use to open the file data
import networkx as nx #use to create graph
from mlxtend.plotting import plot_pca_correlation_graph #only use for create correlation circles



## connect sub graph, use to connect 2 sub_graph  (heuristic question 6)

def connect_sub_graph(list,D):
    TAILLE = len(D)
    min = [0,0,10000]
    not_list = []
    for k in range(TAILLE):
        if(k not in list):
            not_list.append(k)
    for k in list:
        for not_k in not_list:
            if(D[k,not_k]<min[2]):
                min = [k,not_k,D[k,not_k]]
    return min

## k-voisins, use to find the k-1 neighbors of a point on the graph

def k_voisin(point,D,k):
    ind_min = []
    liste_distance = list(D[point])
    for i in range(k):
        ind_min.append(liste_distance.index(min(liste_distance)))
        liste_distance[ind_min[-1]]=100000
    return ind_min
## distance matrix, use to find D

def distance(matrix):
    shape_matrix = matrix.shape
    distance_matrix = np.zeros((shape_matrix[0],shape_matrix[0]))
    for point1 in range(shape_matrix[0]):
        for point2 in range(shape_matrix[0]):
            distance_matrix[point1][point2] = np.linalg.norm(matrix[point1]-matrix[point2])
    return distance_matrix



## random

def randomization(matrix):
    m,n = matrix.shape
    for i in range(m):
        for j in range(n):
            matrix[i][j] = matrix[i][j]+(random.random()-0.5)/100000000
    return matrix


## convertion :
def conversion_int(matrix):
    matrix_int = []
    for a in matrix:
        matrix_int.append(int(a))
    return matrix_int


## normalisation
def normalization(matrice):
    # matrice = randomization(matrice)
    taille = matrice.shape
    mean_variable = np.mean(matrice,0)
    var_variable = np.var(matrice,0)*(taille[0]/(taille[0]-1))
    Y_Normalize  = (matrice-mean_variable)/np.sqrt(var_variable)
    return Y_Normalize
