import os, glob
import re
import numpy as np
import math 
import matplotlib.pyplot as plt
import matplotlib.tri as tri 


def lecture_fichier(folder_path):
    s = []
    for filename in glob.glob(os.path.join(folder_path, '*')):
        for episode in glob.glob(os.path.join(filename, '*.txt')):
            with open(episode, 'r') as f:
                s.append(f.read())
                
    return s

def compterFichier(folder_path): #compte le nombre de fichiers
    countFic = 0
    for filename in glob.glob(os.path.join(folder_path, '*')):
        for episode in glob.glob(os.path.join(filename, '*.txt')):
            with open(episode, 'r') as f:
                countFic = countFic + 1
                
    return countFic
    
def filtrageParMots(s):
    pattern = r'[a-zA-Z]+'
    listeMots = re.findall(pattern, s)
    return listeMots
    
def creationDico(m):
    Dico = dict()
    for s in m:
        listeMots = filtrageParMots(s)
        i = 0
        for mot in listeMots:
            if mot not in Dico:
                if len(mot)>=2:
                    Dico[mot] = i
                    i = i+1 
    return Dico
    
    
def creationMatrice_X(m):
    Dico = creationDico(m)
    X = np.zeros((compterFichier("/home/baranova/Bureau/L2/LU2IN013/addic7ed/1___Lost"), len(Dico)))
    
    ind = 0
    for s in m:
        listeMots = filtrageParMots(s)
        for mot in listeMots:
            if len(mot)>=2:
               
                X[ind, Dico[mot]] = X[ind, Dico[mot]] + 1 
        ind = ind + 1  
    return X

def matrice_DistancesEucl():
    X = creationMatrice_X(lecture_fichier("/home/baranova/Bureau/L2/LU2IN013/addic7ed/1___Lost"))
    M = np.zeros((len(X),len(X)))
    dist = 0
    for i in range(0,len(X)):
        for j in range(0,len(X)):
            M[i][j] = np.linalg.norm(X[i]- X[j])
    return M
    

def matrice_DistancesCos():
    X = creationMatrice_X(lecture_fichier("/home/baranova/Bureau/L2/LU2IN013/addic7ed/1___Lost"))
    M = np.zeros((len(X),len(X)))
    prodScalaire = 0
    for i in range(0,len(X)):
        for j in range(0,len(X)):
            M[i][j]= np.dot(X[i],X[j])/(np.dot(X[i],X[i])*np.dot(X[j],X[j]))
    return M  
    
def matriceTriangulaire(M):
    for i in range(0,len(M)):
        for j in range(0,i):
            M[i][j]=0
    return M
        
def histo(M):
    plt.hist(matriceTriangulaire(M),bins=[0,10,20,30,40],edgecolor = 'blue')
    plt.title('Histogramme des distances par rapport aux episodes')
    plt.xlabel('Distances')
    plt.ylabel('Nombre')
    plt.show()

  
print(histo(matrice_DistancesEucl()))


"""def carteChaleur(M):
    x = []
    y = []
    for k in range(0,compterFichier("/home/baranova/Bureau/L2/LU2IN013/addic7ed/1___Lost")+1):
        x.append(k)
        y.append(k)
    z = []
    for i in range(0,len(M)):
        for j in range(0,len(M)):
            z.append(M[i][j])
    
    levels = [0.7, 0.75, 0.8, 0.85, 0.9] 
    
    plt.figure() 
    ax = plt.gca() 
    ax.set_aspect('equal') 
    CS = ax.tricontourf(x, y, z, levels, cmap=plt.get_cmap('jet')) 
    print(CS)
    cbar = plt.colorbar(CS, ticks=np.sort(np.array(levels)),ax=ax, orientation='horizontal', shrink=.75, pad=.09, aspect=40,fraction=0.05) 
    cbar.ax.set_xticklabels(list(map(str,np.sort(np.array(levels))))) # horizontal colorbar 
    cbar.ax.tick_params(labelsize=8) 
    plt.title('Heat Map') 
    plt.xlabel('Episodes') 
    plt.ylabel('Episodes') 

    plt.show() 
    
print(carteChaleur(matrice_DistancesEucl()))"""
