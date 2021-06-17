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
    
def z_fonction(w, point):
    return np.dot(w, point)

"""def probabilite(z): #probabilite que l'episode numeroEp appartienne a la serie labelise 1, utilise la fonction sigmoide
    return 1/(1 + math.exp(z))
    
def fonction_activation(X_test, w):
    a = []
    z_liste = []
    for i in range(0,len(X_test)):
        z = z_fonction(w,X_test[i])
        print(z)
        z_liste.append(z)
        a.append(probabilite(z))
    plt.plot(z_liste, a)
    plt.xlabel("z")
    plt.ylabel("a(z)")
    plt.show()"""
        
            
"""def retrouveSerie(X_test, X_train, labels, nbEpoques, numeroEp): 
    w, L = perceptron(X_train, labels, nbEpoques)
    if (z(w, X_test[numeroEp]) < 0):
        return 1
    return -1
    
def performance(X_test, X_train, labels, nbEpoques, Y_test): #evaluer la precision
    count = 0
    for i in range(0,len(X_test)): #pas de boucle
        if (retrouveSaison(X_test, X_train, labels, nbEpoques, i) == Y_test[i][2]):
            count = count + 1
    return (count*100)/len(Y_test)"""
    
def retrouveSerie(X_test, w ):
    a = X_test.dot(np.transpose(w))
    return np.where(np.where(a < 0,1,-1))
   
def performance(X_test, w , Y_test): #evaluer la precision
    K = retrouveSerie(X_test, w )
    L = labels(Y_test)
    L = np.where(K-L==0)
    return np.mean(L,axis=0)
    

    
#moyenne d'erreures
#regarder les valeurs de w (quelles sont les plus elevees)    
#histogramme des valeurs de w (classer les valeurs de w, obtenir une courbe)
#interpreter ce qui sort du classifieur 



data = np.load("sauvegarde.npz")
X_train, X_test, Y_train, Y_test = data['name3'],data['name4'],data['name5'],data['name6']
w, L = perceptron(ajout_colonne_de1(X_train),labels(Y_train),10)
#graphe(L)
#print(performance(ajout_colonne_de1(X_test), ajout_colonne_de1(X_train), labels(Y_train), 10, Y_test))

#fonction_activation(ajout_colonne_de1(X_test),w)
    
    
    
        
        
        
        
    
