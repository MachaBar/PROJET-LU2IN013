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

stop_words_list = stopwords.words('english')

"""stop_words_list = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]"""


 
def preprocessing(T):#T est une liste de chaines de caracteres (chaque chaine de caractere contient tout le text d'un fichier)
    pattern = r'[a-zA-Z]+'
    L = [re.findall(pattern,c) for c in T]
    listeMotsProcessed = [ [mot.lower() for mot in liste if len(mot)>2 and (mot.lower() not in stop_words_list)] for liste in L]
    return listeMotsProcessed
 

def BoW(m): #Renvoie X et Dico
    Dico = dict()
    ind = 0
    i = 0
    for s in m:
        for mot in s:
            if mot not in Dico:
                Dico[mot] = i
                i = i+1
    X = np.zeros((len(m), len(Dico)))
    for s in m:
        for mot in s:
            X[ind, Dico[mot]] = X[ind, Dico[mot]] + 1
        ind = ind + 1  
    return X, Dico
    

def document_frequency(X):
    listeDocFreq = [] #vecteur des frequences documentaires pour chaque mot du dictionnaire
    liste = [sum(x) for x in zip(*X)]
    listeDocFreq = [(1.0*count)/len(X) for count in liste]
    return listeDocFreq
       
def inverse_doc_frequency(X):
    liste = [sum(x) for x in zip(*X)]
    listeDocFreq = document_frequency(X)
    liste_log = [math.log(len(X)/count) for count in liste ]
    listeInvDocFreq = [x*y for x,y in zip(listeDocFreq,liste_log)]
    return listeInvDocFreq
    
     
def remove_keys(d, keys):
    to_remove = set(keys)
    filtered_keys = set(d.keys()) - to_remove
    filtered_values = map(d.get, filtered_keys)
    return dict(zip(filtered_keys, filtered_values))

def matriceX_Dico_processed(X,D,docFreq): #seuil en parametre
    #prend en argument la matrice X et la liste des, et enleve les colonnes (mots) qui ne sont pas pertinants grace a l'analyse de word frequency
    L = np.array(docFreq)
    L1 = np.where(L>0.9)[0]
    L2 = np.where(L < 3/len(X))[0]
    L = list(set(L1) | set(L2))
    X = np.delete(X, L, axis=1)
    keys = [key for key in D if D[key] in L]
    #k = [keys[i] for i in L]
    D = remove_keys(D,keys)    
    return X,D

  
def matriceX_Dico_processed_inverse(X,D,invDocFreq): 
    #prend en argument la matrice X et la liste des, et enleve les colonnes (mots) qui ne sont pas pertinants grace a l'analyse de word frequency
    vect = np.array(invDocFreq)
    ind = np.argsort(vect)
    ind = ind[:int(len(ind)*0.1)]
    X = np.delete(X, ind, axis = 1) # on supprime la ieme colonne de X
    keys = [key for key in D if D[key] in ind]
    #k = [keys[i] for i in L]
    D = remove_keys(D,keys)    
    return X,D
    
def mots_recurrents(X,D,invDocFreq): 
    vect = np.array(invDocFreq)
    ind = np.argsort(vect)
    ind = ind[:int(len(ind)*0.1)]
    keys = [key for key in D]
    return [keys[i] for i in ind] 
    
    
def matrice_agreg(X,Y,nbserie):
    Xa = np.zeros((nbserie,len(X[0])))
    for s in range(nbserie):
        y = Y[:,2]
        L = np.where(y==s)[0]
        cpt=0
        p=[]
        for j in range(0,len(X[0])):
            for i in L:
                cpt = cpt + X[i,j]
            p.append(cpt*1./len(L))
            cpt=0
        #f = np.array([np.mean(X[L[0]:L[len(L)-1],i]) for i in range(len(X[0]))])
        Xa[s]=p
    return Xa
    
    


