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

from ouvertureFichiers import lecture_fichier
from preprocessing import preprocessing, BoW, document_frequency, matriceX_Dico_processed
from creationMatricesDistances import matrice_DistancesEucl, matrice_DistancesEucl_version, matrice_DistancesCos_version, matrice_DistancesCos

def split ( M, Y ):
    m_train, m_test, y_train, y_test = train_test_split(M, Y, test_size=0.25,shuffle=True)
    return m_train, m_test, y_train, y_test
    
T,Y = lecture_fichier("/home/baranova/Bureau/L2/LU2IN013/series")
X, dico = BoW(preprocessing(T))
X,dico = matriceX_Dico_processed(X,dico,document_frequency(X))
X_train, X_test, Y_train, Y_test = split(X,Y)
np.savez("sauvegarde.npz", name1 = X, name2 = Y, name3 = X_train, name4 = X_test, name5 = Y_train, name6 = Y_test)

