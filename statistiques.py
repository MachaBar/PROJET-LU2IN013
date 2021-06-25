import os, glob
import re
import numpy as np
import math 
import matplotlib.pyplot as plt
import matplotlib.tri as tri 
import pickle
from tempfile import TemporaryFile
from wordcloud import WordCloud, STOPWORDS
from PIL import Image


from ouvertureFichiers import lecture_fichier
from preprocessing import preprocessing, BoW, document_frequency, matriceX_Dico_processed, inverse_doc_frequency, remove_keys, matriceX_Dico_processed_inverse, mots_recurrents
from creationMatricesDistances import matrice_DistancesEucl, matrice_DistancesEucl_version, matrice_DistancesCos_version, matrice_DistancesCos
from utilitaire import split
from perceptron import ajout_colonne_de1, labels, epoque2, cout_L, perceptron_epoque, perceptron, graphe, retrouveSerie, performance


def histo(M):
    plt.hist(np.reshape(M,len(M)*len(M),order='F'),bins=10,edgecolor = 'blue')
    plt.title('Histogramme des distances par rapport aux episodes')
    plt.xlabel('Distances')
    plt.ylabel('Nombre')
    plt.show()

def find_key(v,Dico):
    for k,val in Dico.items() :  
        if v == val:
            return k

"""def wordcloud(folder_path):
    
    mask = np.array(Image.open("cloud.png"))
    mask[mask == 1] = 255

    #text = open(episode, mode='r').read()
    liste_text, Y = lecture_fichier(folder_path)
    text = ""
    for c in liste_text:
        text = text + c
    wordcloud = WordCloud(background_color = 'white', stopwords = [] , max_words = 50, mask = mask).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()"""
    
def wordcloudLecFic(folder_path):
    #text = open(episode, mode='r').read()
    """mask = np.array(Image.open("cloud.png"))
    mask[mask == 1] = 255"""

    liste_text, Y = lecture_fichier(folder_path)
    text = ""
    for c in liste_text:
        text = text + c
    wordcloud = WordCloud(background_color = 'white', stopwords = [] , max_words = 50).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

def wordcloudPreproc(L):
    """mask = np.array(Image.open("cloud.png"))
    mask[mask == 1] = 255"""

    R = [ e for s in L for e in s ]
    text = " ".join(R)
    wordcloud = WordCloud(background_color = 'white', stopwords = [] , max_words = 50).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

"""def wordcloudDocFreq(X,D,docFreq,L):
    ListeIndiceToDelete = [ i for i in range(0,len(docFreq)) if (docFreq[i] > 0.9) or (docFreq[i] < 3/len(X))]
    ListeMotsToDelete = [ cle for cle, valeur in D.items() if (valeur in ListeIndiceToDelete) ]
    R = [ e for s in L for e in s if (e not in ListeMotsToDelete)]
    text = " ".join(R)
    wordcloud = WordCloud(background_color = 'white', stopwords = [] , max_words = 50).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show() """
    
def wordcloudInvDocFreq(X,D,invDocFreq):
    R = mots_recurrents(X,D,invDocFreq)
    text = " ".join(R)
    wordcloud = WordCloud(background_color = 'white', stopwords = [] , max_words = 50).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show() 
    
    
    
    
T,Y = lecture_fichier("/home/baranova/Bureau/L2/LU2IN013/series_3")

data = np.load("series_3.npz")
X = data['name1']
with open("myDictionary_3series.pkl", "rb") as tf:
    dico = pickle.load(tf)
#print( wordcloudLecFic("/home/baranova/Bureau/L2/LU2IN013/series_3"))
#print(wordcloudPreproc(preprocessing(T)))

print(wordcloudInvDocFreq(X, dico, inverse_doc_frequency(X)))

