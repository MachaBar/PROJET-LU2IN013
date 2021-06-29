import os, glob
import re
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from tempfile import TemporaryFile
import pickle
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import sklearn
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from collections import Counter
import random as Random

from utilitaire import split, split_agreg
from preprocessing import preprocessing, BoW, document_frequency, matriceX_Dico_processed, inverse_doc_frequency, remove_keys, matriceX_Dico_processed_inverse, mots_recurrents,matrice_agreg


def ajout_colonne_de1( X ):
    b = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype) , X))
    return b
   
def labels(Y):
    K = Y[:,2]
    K = np.where(K==0,1,-1)
    return K
       
   
def epoque2(X_train): #genere une epoque de facon aleatoire
    tab = [ i for i in range(len(X_train))] #fonction permute
    Random.shuffle(tab)
    return tab
   
def cout_L ( w , labels, X_train ):
    a = X_train.dot(w.T)
    a = -a*labels
    L = np.where( a>0 , a , 0.)
    return np.mean(L)

     
def perceptron_epoque(X_train,labels,w, epsilon, epoque): #calcule la position du vecteur directeur w pour une epoque
    for i in epoque:
        if (np.dot(w,X_train[i])*labels[i]) < 0 :
            w = np.add(w , epsilon*labels[i]*X_train[i])
    return w
   
def perceptron(X_train,labels, nbEpoques): #graphe du cout en fonction de l'epoque (met a jour w pour nbEpoques differentes generees aleatoirement
    epsilon = 1 #nous avons teste avec epsilon 1, 0.75, 1.5
    w = np.zeros(len(X_train[0]))
    w[0] = 0.75
    L = []
    for i in range(0,nbEpoques):
        epoquee = epoque2(X_train)
        w = perceptron_epoque(X_train,labels, w, epsilon, epoquee)
        L.append(cout_L(w, labels, X_train))
    return w, L 
    
def graphe(L):
    #j = [i for i in range(0,len(L))]
    plt.plot(range(len(L)),L) #construction du graphe du cout en fonction de l'epoque
    plt.xlabel("epoque")
    plt.ylabel("L")
    plt.show()
    
    
def retrouveSerie(X_test, w ):
    a = X_test.dot(np.transpose(w))
    return np.where(np.where(a < 0,1,-1))
   
def performance(X_test, w , Y_test): #evaluer la precision
    K = retrouveSerie(X_test, w )
    L = labels(Y_test)
    L = np.where(K-L==0)
    return np.mean(L,axis=0)
    
def mots_moins_influents(w , dico):
    print("Mots les moins influents")
    k = np.argsort(w)
    #plus influents
    l1 = k[:40]
    keys = [key for key in dico]
    return [keys[i] for i in l1]

def mots_plus_influents(w , dico):
    print("Mots les plus influents")
    k = np.argsort(w)
    #plus influents
    l1 = k[-40:]
    keys = [key for key in dico]
    return [keys[i] for i in l1] 

def epsilon_perceptron(X_train,labels , nbepsilon , nbEpoques ):
    w = np.zeros(len(X_train[0]))
    w[0] = 0.75
    L = []
    epoquee = epoque2(X_train)
    for e in range(0,nbepsilon):
        w = perceptron_epoque(X_train,labels, w, e, epoquee)
        L.append(cout_L(w, labels, X_train))
    return L 
 
#je suis sur qu'on peut le faire sans boucles, mais je suis con   
def labels_reco(Xa): #les series de numero paire sont les series aimes (label 1), les autres ne sont pas aimes (-1)
    L = []
    for i in range(0, len(Xa)):
        if (i%2==0):
            L.append(1)
        else:
            L.append(-1)
    K = np.array(L)
    return K
    
def score_series(X, w):
    return X.dot(w.T)
    
def reco_serie(etiquettes, scores):
    scores_ind = np.sort(scores)
    scores_ind = np.where(scores_ind)
    reco = []
    for i in scores_ind :
        reco.append(etiquettes[i])
    return reco
    
"""def reco_serie_score(etiquettes, scores):
    scores_ind = np.sort(scores)
    scores_ind = np.where(scores_ind)
    reco = []
    for i in scores_ind :
        reco.append(etiquettes[i])
    return reco"""
        
    
   
"""
#classif    
data = np.load("sauvegarde_serie_similaires.npz")
X_train, X_test, Y_train, Y_test = data['name3'],data['name4'],data['name5'],data['name6']
with open("myDictionary.pkl", "rb") as tf:
    dico = pickle.load(tf)
w, L = perceptron(ajout_colonne_de1(X_train),labels(Y_train),10)
#graphe(L)
print(performance(ajout_colonne_de1(X_test),w, Y_test))

print(mots_moins_influents(w, dico))
print(mots_plus_influents(w , dico))
"""

#reco
data = np.load("sauvegarde_multi_10.npz")
X = data['name1']
Y = data['name2']
Xa = matrice_agreg(X,Y,10)
Xa_train, Xa_test  = split_agreg(Xa)
print(Xa_train)
w, L = perceptron(Xa_train,labels_reco(Xa_train), 5)
with open("myEtiquette.pkl", "rb") as tf:
    etiquettes = pickle.load(tf)
print(score_series(Xa_test,w))

print(reco_serie(etiquettes,score_series(Xa_test,w) ))


    
    
    
        
        
        
        
    
