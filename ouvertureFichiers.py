import os, glob
import re
import numpy as np
import math 
import matplotlib.pyplot as plt
import matplotlib.tri as tri 


from tempfile import TemporaryFile

""" Lecture des sous-titres à partir des fichiers fournis"""

def lecture_fichier(folder_path): 
    """ Renvoie:
        - une liste de chaînes de caractères T, chaque chaîne est remplie du contenu d'un fichier
        - la matrice Y, associant les numéros de chaque épisode dans matrice à leur numéro de saison et de série
        - une liste etiquette de noms de toutes les séries contenues dans le dossier (les noms des sous-dossiers)
    """
    T = []
    Y = []
    etiquette =[]
    i = 0
    for serie,dossname in enumerate(glob.glob(os.path.join(folder_path, '*'))):
        folder_path1 = dossname
        s = str(folder_path1)
        s = s.split('___')[1]
        etiquette.append(s)
        for sais,filename in enumerate(glob.glob(os.path.join(folder_path1, '*'))):
            sais = sais +1
            for ep,episode in enumerate(glob.glob(os.path.join(filename, '*.txt'))):
                Y.append([ep,sais,serie])
                with open(episode, 'r' ) as f:
                    T.append(f.read())
    Y = np.array(Y)        
    return T, Y , etiquette

