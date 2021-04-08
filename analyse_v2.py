import os, glob
import re
import numpy as np
import math 
import matplotlib.pyplot as plt
import matplotlib.tri as tri 
from tempfile import TemporaryFile


def lecture_fichier(folder_path):
    s = []
    for dossname in glob.glob(os.path.join(folder_path, '*')):
        folder_path1 = dossname
        for filename in glob.glob(os.path.join(folder_path1, '*')):
            for episode in glob.glob(os.path.join(filename, '*.txt')):
                with open(episode, 'r' ) as f:
                    s.append(f.read())       
    return s

def compterFichier(folder_path): #compte le nombre de fichiers
    countFic = 0
    for dossname in glob.glob(os.path.join(folder_path, '*')):
        folder_path1 = dossname
        for filename in glob.glob(os.path.join(folder_path1, '*')):
            for episode in glob.glob(os.path.join(filename, '*.txt')):
                with open(episode, 'r' ) as f:
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
    #X = np.zeros((compterFichier("1__Lost"), len(Dico)))
    X = np.zeros((compterFichier("/home/baranova/Bureau/L2/LU2IN013/series"), len(Dico)))
    ind = 0
    for s in m:
        listeMots = filtrageParMots(s)
        for mot in listeMots:
            if len(mot)>=2:
                #print(mot)
                X[ind, Dico[mot]] = X[ind, Dico[mot]] + 1 
        ind = ind + 1  
    return X
    
"""def sauvegarde_X():    
    outfile = TemporaryFile()
    M = creationMatrice_X(lecture_fichier("/home/baranova/Bureau/L2/LU2IN013/series"))
    np.save(outfile, M, True, False)
    return outfile"""
    
def matriceY():
    folder_path = "/home/baranova/Bureau/L2/LU2IN013/series"
    Y = np.zeros((compterFichier(folder_path),3))
    i = 0
    ep = 0
    sais = 0
    serie = 0
    for dossname in glob.glob(os.path.join(folder_path, '*')):
        serie = serie + 1
        folder_path1 = dossname
        for filename in glob.glob(os.path.join(folder_path1, '*')):
            sais = sais +1 
            #print(filename)
            for episode in glob.glob(os.path.join(filename, '*.txt')):
                ep = ep + 1
                Y[i][0]= ep
                Y[i][1] = sais
                Y[i][2] = serie
                i = i+1
                #print(episode)
            ep = 0
        sais = 0   
    return Y 
    
"""def matriceY():
    folder_path = "/home/baranova/Bureau/L2/LU2IN013/series"
    Y = np.zeros((compterFichier(folder_path),3))
    pattern1 = r'[__a-zA-Z/]*\d{2}*[__]'
    pattern1_bis = r'\d{2}'
    #pattern2 = 
    i = 0
    for dossname in glob.glob(os.path.join(folder_path, '*')):
        folder_path1 = dossname
        for filename in glob.glob(os.path.join(folder_path1, '*')):
            for episode in glob.glob(os.path.join(filename, '*.txt')):
                epListe = re.findall(pattern1, episode)
                for p in epListe:
                    ep = re.findall(pattern1_bis, p) 
                print(ep)
                #sais = re.findall(pattern2, episode)
                #Y[i][0]= ep
                #Y[i][1] = sais
                #Y[i][2] = serie
                i = i+1
    return Y"""
                     
#print(matriceY())
#matriceY()

def matrice_DistancesEucl():
    #X = creationMatrice_X(lecture_fichier("1__Lost"))
    X = creationMatrice_X(lecture_fichier("/home/baranova/Bureau/L2/LU2IN013/series"))
    M = np.zeros((len(X),len(X)))
    dist = 0
    for i in range(0,len(X)):
        for j in range(0,len(X)):
            M[i][j] = np.linalg.norm(X[i]- X[j])
    return M
#print(matrice_DistancesEucl())
    

def matrice_DistancesCos():
    #X = creationMatrice_X(lecture_fichier("1__Lost"))
    X = creationMatrice_X(lecture_fichier("/home/baranova/Bureau/L2/LU2IN013/series"))
    M = np.zeros((len(X),len(X)))
    prodScalaire = 0
    for i in range(0,len(X)):
        for j in range(0,len(X)):
            M[i][j]= np.dot(X[i],X[j])/(np.dot(X[i],X[i])*np.dot(X[j],X[j]))
    return M 
 
# fonction qui sauvegarde les quatres matrices dans un ficher 
def sauvegarde_matrices():
    outfile = TemporaryFile
    X = creationMatrice_X(lecture_fichier("/home/baranova/Bureau/L2/LU2IN013/series"))
    Y = matriceY()
    DistEucl = matrice_DistancesEucl()
    DistCos = matrice_DistancesCos()
    np.savez(outfile, X=X, Y=Y, DistEucl=DistEucl, DistCos=DistCos)
    return outfile

#npzfile = np.load(sauvegarde_matrices())
#print(npzfile['X'])
    
def classification(k,ep,M): #ep correspond a la ligne/episode dans la matrice X (et M)
   listeDistances = [] #liste des k distances les plus petites de l'episode ep
   l = [] #liste des episodes se situant aux distances les plus petites de ep (les k plus proches episodes voisins)
   listeDistances = sorted(M[ep])
   listeDistances = listeDistances[:k]
   for i in listeDistances:
        for j in range(0,len(M[ep])):
            if (M[ep][j] == i):
                l.append(j) 
   return l
   
#print(classification(3,2,matrice_DistancesEucl()))

#renvoie le numero de la serie qui correspondrai a l'episode recherche   
def rechercheSerie(Y,k,ep,M):#Y est la matrice Y indexee, M est la matrice des distances
    listeEp = classification(k,ep,M)
    listeSerie = []
    for i in listeEp:
        listeSerie.append(Y[i][2]) #dans la matrice Y le numero de la serie est situe dans la 3eme colonne  
    maxiOcc = 0 
    for k in listeSerie:
        if ( listeSerie.count(k) >= maxiOcc) :
            maxiOcc = listeSerie.count(k)
            maxi = k
    return maxi
#print(rechercheSerie(matriceY(),3,2,matrice_DistancesEucl()))
    
def performance(ep,k,Y,M):
    serieHypothese = rechercheSerie(Y,k,ep,M)
    serieReel = Y[ep][2]
    
    if (serieHypothese == serieReel):
        return 1
    return 0
#print(performance(2,3,matriceY(),matrice_DistancesEucl()))

def matriceTriangulaire(M):
    for i in range(0,len(M)):
        for j in range(0,i):
            M[i][j]=0
    return M
    


def histo(M):
    plt.hist(np.reshape(M,len(M)*len(M),order='F'),bins=10,edgecolor = 'blue')
    plt.title('Histogramme des distances par rapport aux episodes')
    plt.xlabel('Distances')
    plt.ylabel('Nombre')
    plt.show()

#print(histo(matrice_DistancesEucl()))
#plt.imshow(matrice_DistancesEucl())


#tests creation et sauvegarde des matrices (faire des prints)
"""
print(creationMatrice_X(m))
print(matriceY())
print(matrice_DistancesEucl())

npzfile = np.load(sauvegarde_matrices())
print(npzfile['X']) #pour acceder a la matrice X
print(npzfile['DistEucl'])
"""

#tests classification
"""
print(classification(3,2,matrice_DistancesEucl()))
print(rechercheSerie(matriceY(),3,2,matrice_DistancesEucl()))
print(performance(2,3,matriceY(),matrice_DistancesEucl()))
"""

#tests statistiques
"""
print(histo(matrice_DistancesEucl()))
plt.imshow(matrice_DistancesEucl())
"""  


   
