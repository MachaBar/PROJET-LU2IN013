import os, glob
import re
import numpy as np
import math 
import matplotlib.pyplot as plt
import matplotlib.tri as tri 
from tempfile import TemporaryFile
from scipy.spatial import distance

from ouvertureFichiers import lecture_fichier
from preprocessing import preprocessing, BoW, document_frequency, matriceX_Dico_processed, inverse_doc_frequency, remove_keys, matriceX_Dico_processed_inverse, mots_recurrents
from creationMatricesDistances import matrice_DistancesEucl, matrice_DistancesEucl_version, matrice_DistancesCos_version, matrice_DistancesCos
from utilitaire import split

import sklearn
from collections import Counter
from sklearn.model_selection import train_test_split


def kppv( M ,  X_train, Y_train, X_test, k): 
    Y_chapeau = []
    #print(M_test)
    for i in range(len(X_test)):
        listeDistances = [] #liste des k distances les plus petites de l'episode ep
        l = [] #liste des episodes se situant aux distances les plus petites de ep (les k plus proches episodes voisins)
        listeDistances = M[i].argsort()
        listeDistances = listeDistances[:k]
        listeSerie = [Y_train[s][2] for s in listeDistances]
        maxi,num_most_common = Counter(listeSerie).most_common(1)[0]
        Y_chapeau.append(maxi)
    #print(Y_chapeau)
    return Y_chapeau #doit aussi renvoyer Y_train_chapeau
   
   
def performance(Y_test, Y_chapeau):
    count = 0
    for i in range(len(Y_chapeau)):
        if (Y_chapeau[i] == Y_test[i][2]):
            count = count + 1
    return (count*100)/len(Y_chapeau)
    

def placement_ep( M, numS, Y):
    listePlacementEpisodes = []
   
    for i in range(len(Y)):
        if (Y[i][2] == numS ) :
            listePlacementEpisodes.append(i)
    return listePlacementEpisodes

def kppvRec1( numS, MkppvRec1, listePlacementEpisodes, Y, k): 
    Y_chapeau = []

    for p in listePlacementEpisodes:
        listeDistances = [] #liste des episodes ayant les k distances les plus petites de l'episode a l'indice i 
        listeDistances = MkppvRec1[p].argsort()
        listeDistances = [l for l in listeDistances if (Y[l][2] != numS)]
        listeDistances = listeDistances[:k]
        listeSerie = [Y[s][2] for s in listeDistances]
        maxi,num_most_common = Counter(listeSerie).most_common(1)[0]
        Y_chapeau.append(maxi)
    return Y_chapeau 

def fseriesRecom( Y_chapeau, Y, f ):   
    listefseries = []

    for i in range(f):
        a,b = Counter(Y_chapeau).most_common(1)[0]
        listefseries.append(a)
        Y_chapeau = [j for j in Y_chapeau if (j!=a) ]
    
    return listefseries



#tests :

# On choisi ici la serie 0 et on pose f = 5 et k = 5:
T,Y = lecture_fichier("/home/baranova/Bureau/L2/LU2IN013/series_5")
X, dico = BoW(preprocessing(T))
X, dico = matriceX_Dico_processed_inverse(X, dico, document_frequency(X))
M = matrice_DistancesCos(X)
listePlacementEpisodes = placement_ep(M,0,Y)      
Y_chapeau = kppvRec1( 0, M, listePlacementEpisodes, Y, 5)     
print(fseriesRecom( Y_chapeau, Y, 3))


"""    
#main 1
T,Y = lecture_fichier("/home/baranova/Bureau/L2/LU2IN013/series")
X, dico = BoN(preprocessing(T))
X, dico = matriceX_Dico_processed_inverse(X, dico, document_frequency(X))
M = matrice_DistancesCos(X)
#M = scipy.spatial.distance.pdist(X, 'cosine')
m_train, m_test, y_train, y_test = split( M,Y) 
#print(y_train)
#print(performance(y_test,kppv(m_train, y_train, m_test, y_test,100)))
"""


"""T,Y = lecture_fichier("/home/baranova/Bureau/L2/LU2IN013/series")
X, dico = BoW(preprocessing(T))
np.savez("matriceXY.npz", name1 = X, name2 = Y)
data = np.load("matriceXY.npz")
X_train, X_test, Y_train, Y_test = split(data['name1'],data['name2'])"""


"""
data = np.load("sauvegarde_multi.npz")
X_train, X_test, Y_train, Y_test = data['name3'],data['name4'],data['name5'],data['name6']   
M = matrice_DistancesCos_version (X_test , X_train )
print(data['name1'])
print(kppv(M, X_train, Y_train  , X_test ,10))
print(performance(Y_test,kppv(M, X_train, Y_train  , X_test ,10)))
#plt.imshow(matrice_DistancesEucl(X))
"""

