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

from utilitaire import split , nb_serie
from perceptron import ajout_colonne_de1, labels, epoque2, cout_L, perceptron_epoque, perceptron, graphe, z_fonction, retrouveSerie, performance

def labels_multiclasse(numSerie, Y):
    K = Y[:,2]
    K = np.where(K==numSerie,1,-1)
    return K
    
                                             
def multiclasse_epoque(X_train, Y_train, W, epsilon, epoque , n_serie): #renvoie la matrice W, dont les colonnes sont les vecteurs directeurs w pour chaque serie
    for serie in range(n_serie): #on choisit la serie qui sera labelise 1 (toutes les autres seront labelisees -1)
        labels = labels_multiclasse(serie, Y_train)
        w = np.zeros(len(X_train[0]))
        w[0] = 5
        for i in epoque:
            if (np.dot(w,X_train[i])*labels[i]) < 0 :
                w = np.add(w , epsilon*labels[i]*X_train[i])
        W[serie]=w
    return W
    
   

def multiclasse_perceptron(X_train, Y_train, epsilon ,   nbEpoques , n_serie):
    #epsilon = 1 #nous avons teste avec epsilon 1, 0.75, 1.5
    W = np.ones((n_serie,len(X_train[0])))
    #W[0][0] = 0.75
    for i in range(0,nbEpoques):
        epoquee = epoque2(X_train)
        W = multiclasse_epoque(X_train, Y_train, W, epsilon, epoquee, n_serie)
    return W
    
    
def recherche_serie_multi(X_test, W):
    K = W.dot(np.transpose(X_test))
    return np.argmax(K, axis=0)

def matrice_confusion(X_test, Y_test, W, nbSerie):
    K = np.zeros((nbSerie,nbSerie))
    Y_prediction = recherche_serie_multi(X_test, W)
    print(Y_prediction)
    for i in range(0,len(Y_test)):
        print(Y_test[i][2])
        print(Y_prediction[i])
        K[Y_test[i][2]][Y_prediction[i]] = K[Y_test[i][2]][Y_prediction[i]] + 1
    return K
    
    
def performance_multi(X_test, W , Y_test): #evaluer la precision
    K = recherche_serie_multi(X_test, W)
    L = Y_test[:,2].T
    c = L-K
    return (c == 0).sum() *100. / len(c)
    
def plot_perf_eps(X_test, Y_test, nbEps,X_train, Y_train , nbEpoques , n_serie):
    L = []
    for eps in range(0,nbEps):
        W = multiclasse_perceptron(X_train, Y_train, eps ,   nbEpoques , n_serie)
        L.append(performance_multi(X_test,W, Y_test))
    plt.plot(range(nbEps),L) #construction du graphe du cout en fonction de l'epoque
    plt.xlabel("epsilon")
    plt.ylabel("performance")
    plt.show()
    return L

        
data = np.load("sauvegarde_multi.npz")
X_train, X_test, Y_train, Y_test = data['name3'],data['name4'],data['name5'],data['name6']   
W = multiclasse_perceptron(ajout_colonne_de1(X_train), Y_train, 1, 10,10)
print(matrice_confusion(ajout_colonne_de1(X_test), Y_test, W, 10))
print(performance_multi(ajout_colonne_de1(X_test), W, Y_test))
