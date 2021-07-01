import os, glob
import re
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from tempfile import TemporaryFile
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import sklearn
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from collections import Counter

from ouvertureFichiers import lecture_fichier
from preprocessing import preprocessing, BoW, document_frequency, matriceX_Dico_processed, inverse_doc_frequency, remove_keys, matriceX_Dico_processed_inverse, mots_recurrents,matrice_agreg

def matrice_DistancesEucl( X ):
    return distance.squareform(distance.pdist(X, 'euclidean'))
   
# X et Y la sont mtrain et mtest
def matrice_DistancesEucl_version( X , Y ):
    return distance.cdist(X, Y, 'euclidean')
   
# X et Y la sont mtrain et mtest
def matrice_DistancesCos_version(X , Y):
    return distance.cdist(X, Y, 'cosine')

def matrice_DistancesCos(X):
    return distance.squareform(distance.pdist(X, 'cosine'))
    
def split ( M, Y):
    m_train, m_test, y_train, y_test = train_test_split(M, Y, test_size=0.25,shuffle=False)
    return m_train, m_test, y_train, y_test
   
