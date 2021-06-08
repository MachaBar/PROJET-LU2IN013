import os, glob
import re
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from tempfile import TemporaryFile
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import sklearn
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from collections import Counter
import random as Random

from utilitaire import split
from perceptron import ajout_colonne_de1, labels, epoque2, cout_L, perceptron_epoque, perceptron, graphe, z_fonction, retrouveSerie, performance

def labels_multiclasse(numSerie, Y):
    K = np.zeros((len(Y),1))
    for j in range(len(Y)):
        if Y[j][2] == numSerie:
            K[j][0] = 1
        else :
            K[j][0] = -1
    return K
    
                                             
def multiclasse_epoque(X_train, Y_train, W, epsilon, epoque): #renvoie la matrice W, dont les colonnes sont les vecteurs directeurs w pour chaque serie
    for serie in range(0,Y_train[len(Y_train)-1][2]-Y_train[0][2]): #on choisit la serie qui sera labelise 1 (toutes les autres seront labelisees -1) 
        labels = labels_multiclasse(serie, Y_train)
        w = np.zeros(len(X_train[0]))
        for i in epoque:
            if (np.dot(w,X_train[i])*labels[i]) < 0 :
                w = np.add(w , epsilon*labels[i]*X_train[i])
        np.append(w, W)
    return W
    

def multiclasse_perceptron(X_train, Y_train,  nbEpoques):
    epsilon = 1 #nous avons teste avec epsilon 1, 0.75, 1.5
    print(Y_train[len(Y_train)-1][2]-Y_train[0][2])
    W = np.ones((Y_train[len(Y_train)-1][2]-Y_train[0][2],len(X_train[0])))
    #W[0][0] = 0.75
    for i in range(0,nbEpoques):
        epoquee = epoque2(X_train)
        W = multiclasse_epoque(X_train, Y_train, W, epsilon, epoquee)
    return W
    

def recherche_serie_multi(X_test, W, nbEpoques):
    #W = multiclasse_perceptron(X_train, Y_train, nbEpoques)
    K = W.dot(np.transpose(X_test))
    return np.argmax(K, axis=0)

    
def performance_multi(X_test, W , Y_test, nbEpoques): #evaluer la precision
    K = recherche_serie_multi(X_test, W, nbEpoques )
    L = labels(Y_test)
    L = np.where(K-L==0)
    return np.mean(L,axis=0)
    

data = np.load("sauvegarde.npz")
X_train, X_test, Y_train, Y_test = data['name3'],data['name4'],data['name5'],data['name6'] 
  
W = multiclasse_perceptron(ajout_colonne_de1(X_train), Y_train, 10)
print(performance_multi(ajout_colonne_de1(X_test), W, Y_test,10))
