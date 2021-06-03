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


def ajout_colonne_de1( X ):
    b = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype) , X))
    return b
   

def mise_a_jour_Y ( Y ):
    K = np.zeros((len(Y),1))
    for j in range(len(Y)):
        if Y[j][2] == 0:
            K[j][0] = 1
        else :
            K[j][0] = -1
    return np.hstack((Y,K))
   

"""
def epoque(X_train): #renvoie un tableau
    tabBool = [False for i in range(0,len(X_train))]
    res = []
    cpt = 0
    while cpt < len(X_train):
        r = Random.randint(0,len(X_train)-1)
        if (tabBool[r] != True):
            res.append(r)
            cpt= cpt+1
            tabBool[r] = True
    return res"""
   
def epoque2(X_train): #genere une epoque de facon aleatoire
    tab = [ i for i in range(len(X_train))]
    Random.shuffle(tab)
    return tab
   
def cout_L ( w , Y_train , X_train , epoque):
    """L = [max(0.0,-np.dot(w,X_train[epoque[i]])*Y_train[epoque[i]]) for i in range(len(epoque))]
    return sum(L)*1./len(epoque)"""
    L=[]
    for i in range(len(epoque)):
        if ( -np.dot(w,X_train[epoque[i]])*Y_train[epoque[i]][3] > 0 ):
            L.append(-np.dot(w,X_train[epoque[i]])*Y_train[epoque[i]][3])
        else :
            L.append(0.0)
    return sum(L)*1./len(epoque)

     
def perceptron_epoque(X_train,Y_train_labelise,w, epsilon, epoque): #calcule la position du vecteur directeur w pour une epoque
    for i in range(0,len(epoque)):
        if (np.dot(w,X_train[epoque[i]])*Y_train_labelise[epoque[i]][3]) < 0 :
            w = np.add(w , epsilon*Y_train_labelise[epoque[i]][3]*X_train[epoque[i]])
    return w
   
def perceptron(X_train, Y_train_labelise, nbEpoques): #graphe du cout en fonction de l'epoque (met a jour w pour nbEpoques differentes generees aleatoirement
    epsilon = 1 #nous avons teste avec epsilon 1, 0.75, 1.5
    w = np.zeros(len(X_train[0]))
    w[0] = 0.75
    L = []
    for i in range(0,nbEpoques):
        epoquee = epoque2(X_train)
        w = perceptron_epoque(X_train,Y_train_labelise, w, epsilon, epoquee)
        L.append(cout_L(w, Y_train_labelise, X_train, epoquee))
    j = [i for i in range(0,nbEpoques)]
    return w, j, L
    
def graphe(j,L):
    plt.plot(j,L) #construction du graphe du cout en fonction de l'epoque
    plt.xlabel("epoque")
    plt.ylabel("L")
    plt.show()
    
def retrouveSaison(X_test, X_train, Y_train_labelise, nbEpoques, numeroEp): 
    w, j, L = perceptron(X_train, Y_train_labelise, nbEpoques)
    if (np.dot(w, X_test[numeroEp]) < 0):
        return 1
    return 0
    
def performance(X_test, X_train, Y_train_labelise, nbEpoques, Y_test):
    count = 0
    for i in range(0,len(X_test)):
        if (retrouveSaison(X_test, X_train, Y_train_labelise, nbEpoques, i) == Y_test[i][2]):
            count = count + 1
    return (count*100)/len(Y_test)
    


data = np.load("sauvegarde.npz")
X_train, X_test, Y_train, Y_test = data['name3'],data['name4'],data['name5'],data['name6']
w, j, L = perceptron(ajout_colonne_de1(X_train),mise_a_jour_Y(Y_train),10)
#graphe(j, L)
print("test1")
print(performance(ajout_colonne_de1(X_test), ajout_colonne_de1(X_train), mise_a_jour_Y(Y_train), 10, Y_test))
print("test2")

    
    
    
        
        
        
        
    
