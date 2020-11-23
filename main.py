import os #use to open the file "data"
import numpy as np # use for matrix
import random #use to plot label a random place
import matplotlib.pyplot as plt #use to plot result
from scipy import linalg #use for eigenvalues and ... decomposition
import csv #use to open the file data
import networkx as nx #use to create graph
from mlxtend.plotting import plot_pca_correlation_graph #only use for create correlation circles

from fonctions import * # here we create lot of fonctions



## open test
csvfile_data = open("zoo.data","r", encoding="utf8")
read_data = csv.reader(csvfile_data, delimiter=',')
Nom = []
Type = []
Data = []
for line in read_data:
        Nom.append(line[0])
        Type.append(int(line[-1]))
        Data.append(conversion_int(line[1:-1]))

Data = np.array(Data)
Y_norm =normalization(Data) # we normalize variables
TAILLE = Y_norm.shape[0] ## number of animal
DIMENTION =  Y_norm.shape[1] #number of variables
p = 2 ## number of principales components we care about
Yh_norm = Y_norm.T
feature_names = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes',' venomous','fins','legs','tail','domestic','catsize']

##choice TODO


TAILLE = Y_norm.shape[0] ## number of animal
DIMENTION =  Y_norm.shape[1] #number of variables
p = 2 ## number of principales components we care about


# choice = 3 # PCA, MDS (special metric) and isomap (ALL)
#choice = 1 # PCA (with normalize metric)
#choice = 2 # MDS with normalize metric
#choice = 3 # MDS with special metric (to have better interspace between type)
choice = 4 # non linear, isomap







##PCA
if(choice==0 or choice ==1):
    U, s, Vh = linalg.svd(Y_norm)
    I = np.zeros((DIMENTION,p)) ## create I
    for k in range(0,p):
        I[k,k] = 1
    V = Vh.T
    X = np.dot(np.dot(Y_norm,V),I)
    Xh = X.T
    ## print PCA
    A = []
    B = []
    C = []
    D = []
    E = []
    F = []
    G = []
    for k in range(0,len(X)):
        if(Type[k]==1):
            A.append(X[k])
        if(Type[k]==2):
            B.append(X[k])
        if(Type[k]==3):
            C.append(X[k])
        if(Type[k]==4):
            D.append(X[k])
        if(Type[k]==5):
            E.append(X[k])
        if(Type[k]==6):
            F.append(X[k])
        if(Type[k]==7):
            G.append(X[k])


    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)
    E = np.array(E)
    F = np.array(F)
    G = np.array(G)
    Ah = A.T
    Bh = B.T
    Ch = C.T
    Dh = D.T
    Eh = E.T
    Fh = F.T
    Gh = G.T
    fig, ax = plt.subplots()
    for i, txt in enumerate(Nom):
        ax.annotate(txt, (X[i][0], X[i][1]+ (random.random()-0.5)/5))
    ax.scatter(Ah[0],Ah[1],s=40)
    ax.scatter(Bh[0],Bh[1],s=80)
    ax.scatter(Ch[0],Ch[1],s=120)
    ax.scatter(Dh[0],Dh[1],s=160)
    ax.scatter(Eh[0],Eh[1],s=200)
    ax.scatter(Fh[0],Fh[1],s=240)
    ax.scatter(Gh[0],Gh[1],s=270)
    ax.legend( [('class ' + str(k+1)) for k in range(7)])
    plt.show()

    plt.title('Projection by PCA of the zoo on the first 2 principal components')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    ## PCA circle (we just use a librairy to do this part)

    figure, correlation_matrix = plot_pca_correlation_graph(Y_norm,
                                                            feature_names,
                                                            dimensions=(1, 2),
                                                            figure_axis_size=10)
    plt.show()

## MDS
if(choice == 2 or choice == 3 or choice == 0):
    Yh_norm = Y_norm.T
    if(choice == 3 or choice == 0):
        metric = []
        for k in range(DIMENTION):
            metric.append((abs(np.corrcoef(Yh_norm[k], Type)[0][1])))## find correlation betweeen type and initial variables

        Y_norm = Y_norm*metric #change the metric (that mean that's we encourage interspace between types.



    S=np.dot(Y_norm,Y_norm.T)
    Lambda,U = linalg.eig(S) ## eignes decompositions
    Lambda = np.real(Lambda)
    U = np.real(U)
    I = np.zeros((len(U),p)) ## create I
    for k in range(0,p):
        I[k,k] = 1
    Lambda_diag =np.zeros((len(U),len(U)))
    for k in range(0,p):
        Lambda_diag[k,k] = Lambda[k]
    X=np.dot(np.dot(U,np.sqrt(Lambda_diag)),I)

    ## plot MDS
    Xh = X.T
    A = []
    B = []
    C = []
    D = []
    E = []
    F = []
    G = []
    for k in range(0,len(X)):
        if(Type[k]==1):
            A.append(X[k])
        if(Type[k]==2):
            B.append(X[k])
        if(Type[k]==3):
            C.append(X[k])
        if(Type[k]==4):
            D.append(X[k])
        if(Type[k]==5):
            E.append(X[k])
        if(Type[k]==6):
            F.append(X[k])
        if(Type[k]==7):
            G.append(X[k])


    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)
    E = np.array(E)
    F = np.array(F)
    G = np.array(G)
    Ah = A.T
    Bh = B.T
    Ch = C.T
    Dh = D.T
    Eh = E.T
    Fh = F.T
    Gh = G.T
    fig, ax = plt.subplots()
    for i, txt in enumerate(Nom):
        ax.annotate(txt, (X[i][0], X[i][1]+ (random.random()-0.5)/5))
    ax.scatter(Ah[0],Ah[1],s=40)
    ax.scatter(Bh[0],Bh[1],s=80)
    ax.scatter(Ch[0],Ch[1],s=120)
    ax.scatter(Dh[0],Dh[1],s=160)
    ax.scatter(Eh[0],Eh[1],s=200)
    ax.scatter(Fh[0],Fh[1],s=240)
    ax.scatter(Gh[0],Gh[1],s=270)
    ax.legend( [('class ' + str(k+1)) for k in range(7)])
    plt.show()

    plt.title('Projection by MDS of the zoo on the first 2 principal components')
    plt.xlabel('PC1')
    plt.ylabel('PC2')


    ## PCA correlation circle with metric base on correlation with type
    figure, correlation_matrix = plot_pca_correlation_graph(Y_norm,
                                                            feature_names,
                                                            dimensions=(1, 2),
                                                            figure_axis_size=10)
    plt.show()
##No linear
if(choice == 4 or choice == 0):
    nb_voisin = 5
    Y_norm = Yh_norm.T
    D= distance(Y_norm)
    G = nx.Graph()
    nb_point = [k for k in range(TAILLE)]
    G.add_nodes_from(nb_point)
    for point in range(TAILLE):
        voisins  = k_voisin(point,D,nb_voisin) #we find k-1 neibors of "point"
        for v in voisins:
            G.add_edge(point, v, weight=D[point,v])
    nb_connect_graph = len([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]) # find the number of unconnected graphs
    while(nb_connect_graph>1): # while we have a disconnected graph (see Question 6)
        connect_graph = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for k in range(nb_connect_graph):
            list_element = list(connect_graph[k].nodes) # list of element of one of the subgraph
            minimum_link = connect_sub_graph(list_element,D) # We find the closest neighbors between 2 unconnected subgraphs
            G.add_edge(minimum_link[0], minimum_link[1], weight=minimum_link[2]) #we create the link
        nb_connect_graph = len([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]) # find the number of unconnected graphs
    D_nn_lin = np.zeros((TAILLE,TAILLE))
    for point1 in range(TAILLE):
        for point2 in range(TAILLE):
            length, path = nx.single_source_dijkstra(G, point1, point2)
            D_nn_lin[point1,point2]=length**2 #we create D^2 with the graph

    # classic MDS but we use D__nn_lin instead of D
    One = np.array([1 for k in range(TAILLE)])
    One = One.reshape(-1,1)
    Ones = np.dot(One,One.T)
    S = -0.5*(D_nn_lin-(1/TAILLE)*np.dot(D_nn_lin,Ones)-(1/TAILLE)*np.dot(Ones,D)+(1/(TAILLE**2))*np.dot(np.dot(Ones,D),Ones)) # Gram matrix

    Lambda,U = linalg.eig(S)
    Lambda = np.real(Lambda)
    U = np.real(U)
    I = np.zeros((len(U),p))
    for k in range(0,p):
        I[k,k] = 1
    Lambda_diag =np.zeros((len(U),len(U)))
    for k in range(0,p):
        Lambda_diag[k,k] = Lambda[k]
    X=np.dot(np.dot(U,np.sqrt(Lambda_diag)),I)

    ##Plot isomap
    Xh = X.T
    A = []
    B = []
    C = []
    D = []
    E = []
    F = []
    G = []
    for k in range(0,len(X)):
        if(Type[k]==1):
            A.append(X[k])
        if(Type[k]==2):
            B.append(X[k])
        if(Type[k]==3):
            C.append(X[k])
        if(Type[k]==4):
            D.append(X[k])
        if(Type[k]==5):
            E.append(X[k])
        if(Type[k]==6):
            F.append(X[k])
        if(Type[k]==7):
            G.append(X[k])


    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)
    E = np.array(E)
    F = np.array(F)
    G = np.array(G)
    Ah = A.T
    Bh = B.T
    Ch = C.T
    Dh = D.T
    Eh = E.T
    Fh = F.T
    Gh = G.T
    fig, ax = plt.subplots()
    for i, txt in enumerate(Nom):
        ax.annotate(txt, (X[i][0], X[i][1]+ (random.random()-0.5)/5))
    ax.scatter(Ah[0],Ah[1],s=40)
    ax.scatter(Bh[0],Bh[1],s=80)
    ax.scatter(Ch[0],Ch[1],s=120)
    ax.scatter(Dh[0],Dh[1],s=160)
    ax.scatter(Eh[0],Eh[1],s=200)
    ax.scatter(Fh[0],Fh[1],s=240)
    ax.scatter(Gh[0],Gh[1],s=270)
    ax.legend( [('class ' + str(k+1)) for k in range(7)])
    plt.show()

    plt.title('Isomap projection')
    plt.xlabel('x')
    plt.ylabel('y')



