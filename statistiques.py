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
    
def wordcloudLecFic(folder_path):
    liste_text, Y = lecture_fichier(folder_path)
    text = ""
    for c in liste_text:
        text = text + c
    wordcloud = WordCloud(background_color = 'white', stopwords = [] , max_words = 50).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

def wordcloudPreproc(L):
    R = [ e for s in L for e in s ]
    text = " ".join(R)
    wordcloud = WordCloud(background_color = 'white', stopwords = [] , max_words = 50).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

    
def wordcloudInvDocFreq(X,D,invDocFreq):
    R = mots_recurrents(X,D,invDocFreq)
    text = " ".join(R)
    wordcloud = WordCloud(background_color = 'white', stopwords = [] , max_words = 50).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show() 
   
