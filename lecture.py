import os, glob
import re
import numpy as np

folder_path = "/home/baranova/Bureau/L2/LU2IN013/addic7ed/1___Lost/02"

pattern = r'[a-zA-Z]+'
Dico = dict()
countFic = 0
i = 0


for filename in glob.glob(os.path.join(folder_path, '*.txt')):
    with open(filename, 'r') as f:
        
        s = f.read()
        listeMots = re.findall(pattern, s)
        
        for mot in listeMots:
            if mot not in Dico:
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
            X[ind, Dico[mot]] = X[ind, Dico[mot]] + 1 
        ind = ind + 1    
         
print(X)

        





