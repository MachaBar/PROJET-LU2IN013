import os, glob
import re
import numpy as np
import math 
import matplotlib.pyplot as plt

"""folder_path = "/home/baranova/Bureau/L2/LU2IN013/addic7ed/1___Lost"

pattern = r'[a-zA-Z]+'
Dico = dict()
countFic = 0
i = 0"""
#s = "Hello baby, what's up? We have a lot of hope! Kises's"
#listeMots = re.findall(pattern, s)
#print(listeMots)

"""for root, dirnames, filenames in os.walk("/home/baranova/Bureau/L2/LU2IN013/addic7ed/1___Lost/"):
    for dirname in dirnames:
        for filename in filenames:           
           f1 = os.path.join(root,filename)
           f = open(f1,"r")    
           s = f.read()
                
           listeMots = re.findall(pattern, s)
           print(listeMots)   
           for mot in listeMots:
              if mot not in Dico:
                if len(mot)>=2:
                    Dico[mot] = i
                    i = i+1 
              countFic = countFic + 1
 
print(Dico)"""

"""for filename in glob.glob(os.path.join(folder_path, '*')):
    
    for episode in glob.glob(os.path.join(filename, '*.txt')):
        
        with open(episode, 'r') as f:
            
            s = f.read()
            listeMots = re.findall(pattern, s)
            
            for mot in listeMots:
                if mot not in Dico:
                   if len(mot)>=2:
                    Dico[mot] = i
                    i = i+1 
            countFic = countFic + 1
 
    print(Dico)"""

"""def creationMatrice_X():
    folder_path = "/home/baranova/Bureau/L2/LU2IN013/addic7ed/1___Lost"

    pattern = r'[a-zA-Z]+'
    Dico = dict()
    countFic = 0
    for filename in glob.glob(os.path.join(folder_path, '*')):
        for episode in glob.glob(os.path.join(filename, '*.txt')):
            with open(episode, 'r') as f:
                s = f.read()
                listeMots = re.findall(pattern, s)
                i = 0
                for mot in listeMots:
                    if mot not in Dico:
                       if len(mot)>=2:
                            Dico[mot] = i
                            i = i+1 
                countFic = countFic + 1

    X = np.zeros((countFic, len(Dico)))
    ind = 0
    for filename in glob.glob(os.path.join(folder_path, '*')):
        for episode in glob.glob(os.path.join(filename, '*.txt')):
            with open(episode, 'r') as f:
        
                s = f.read()
                listeMots = re.findall(pattern, s)
        
                for mot in listeMots:
                    if len(mot)>=2:
                        X[ind, Dico[mot]] = X[ind, Dico[mot]] + 1 
                ind = ind + 1    
    print(Dico)    
    return X"""
    
def creationMatrice_X():
    folder_path = "/home/baranova/Bureau/L2/LU2IN013/addic7ed/1___Lost/01"

    pattern = r'[a-zA-Z]+'
    Dico = dict()
    countFic = 0
    for filename in glob.glob(os.path.join(folder_path, '*.txt')):
        with open(filename, 'r') as f:
            s = f.read()
            listeMots = re.findall(pattern, s)
            i = 0
            for mot in listeMots:
                if mot not in Dico:
                   if len(mot)>=2:
                        Dico[mot] = i
                        i = i+1 
            countFic = countFic + 1

    X = np.zeros((countFic, len(Dico)))
    ind = 0
    for filename in glob.glob(os.path.join(folder_path, '*.txt')):
       
        with open(filename, 'r') as f:
        
            s = f.read()
            listeMots = re.findall(pattern, s)
        
            for mot in listeMots:
                if len(mot)>=2:
                    X[ind, Dico[mot]] = X[ind, Dico[mot]] + 1 
            ind = ind + 1    
    #print(Dico)    
    return X
#creationMatrice_X()

def matrice_DistancesEucl():
    X = creationMatrice_X()
    M = np.zeros((len(X),len(X)))
    dist = 0
    for i in range(0,len(X)):
        for j in range(0,len(X)):
            dist = 0
            for k in range(0,len(X[0])): 
                dist = dist +(X[i][k]-X[j][k])*(X[i][k]-X[j][k])
            M[i][j]=math.sqrt(dist)
    return M 

def matrice_DistancesCos():
    X = creationMatrice_X()
    M = np.zeros((len(X),len(X)))
    prodScalaire = 0
    for i in range(0,len(X)):
        mod1 = 0
        for j in range(0,len(X)):
            prodScalaire = 0
            mod2 = 0
            for k in range(0,len(X[0])): 
                prodScalaire = prodScalaire + X[i][k]*X[j][k]
                mod1 = mod1 + X[i][k]*X[i][k]
                mod2 = mod2 + X[j][k]*X[j][k]
            M[i][j]= prodScalaire/(mod1*mod2)
    return M
    
def histo(M):
    """for i in range(0,len(M)):
        plt.hist(M[i],len(M),edgecolor = 'blue')"""
    plt.hist(M,len(M),edgecolor = 'blue')
    plt.title('Histogramme des distances par rapport aux episodes')
    plt.xlabel('Distances')
    plt.ylabel('Episodes')
    plt.show()

histo(matrice_DistancesEucl())

#print(matrice_DistancesCos())        
            
            
    
    

        





