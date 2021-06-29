import os, glob
import re
import numpy as np
import math 
import matplotlib.pyplot as plt
import matplotlib.tri as tri 


from tempfile import TemporaryFile
"""
def lecture_fichier(folder_path): #Renvoie T et Y
    T = []
    Y = []
    i = 0
    for serie,dossname in enumerate(glob.glob(os.path.join(folder_path, '*'))):
        folder_path1 = dossname
        for sais,filename in enumerate(glob.glob(os.path.join(folder_path1, '*'))):
            sais = sais +1
            for ep,episode in enumerate(glob.glob(os.path.join(filename, '*.txt'))):
                Y.append([ep,sais,serie])
                with open(episode, 'r' ) as f:
                    T.append(f.read())
    Y = np.array(Y)        
    return T, Y"""
   
def lecture_fichier(folder_path): #Renvoie T et Y
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

