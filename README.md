# PROJET-LU2IN013

Ce dépot Git conserve l'ensemble du code réalisé dans le cadre d'un projet universitaire de trois étudiantes en double-majeure mathématiques-informatique de Sorbonne Université encadrées par Nicolas Baskiotis et Vincent Guigue pendant l'année 2020-2021. 

Ce projet nommé "Recommandation et analyse de sous-titres" a pour objectif de présenter des algorithmes utilisés pour la recommandation des séries télévisées. Grâce au travail réalisé nous avons pu découvrir le monde passionnant de la Data Science et du Machine Learning en nous familiarisant avec ces principaux concepts. Ce domaine, inconnu auparavant nous a fortement passionné. En effet, il se trouve à l'intersection des mathématiques et de l'informatique, ce qui se marie parfaitement à notre formation. De plus, ce projet illustre parfaitement les possibilités d'application des principes appris dans des domaines différents, notamment le cinéma et l'industrie en général. 

Dans une première partie, nous avons traîtés les fichiers textuels fournis, en utilisant les outils de prétraitement connus tels que les listes des Stop Words,  les Regular Expressions ou encore les fréquences TF/IDF. Cette partie est présentée dans les fichiers ouvertureFichiers.py et preprocessing.py.

Nous avons par la suite formé notre Data Set, puis les matrices de distances et de similarité. (Fichiers preprocessing.py, creationMatricesDistances.py)
Cette partie nous a paru spécialement passionnante, car la représentation de données textuelles abstraites n'est en effet pas une chose triviale. Elle nous a également permis d'apprécier les avantages de notre background mathématique pouvant sembler parfois trop théorique.

Dans un deuxième temps, nous nous sommes intéressées aux algorithmes d'apprentissage supervisé afin de réaliser une classification de nos données. Nous avons implémenté les algorithmes classiques: des k plus proches voisins (fichier classification.py), perceptron (fichier perceptron.py) et percreptron multiclasse (fichier multiclasse.py). Nous avons également fait attention à respecter tout au long de notre travail la règle fondamentale de l'apprentissage supérvisé: la subdivision de notre corpus en un ensemble d'apprentissage et de test. (Fichier utilitaire.py).

Finalement, nous avions adaptés nos algorithmes de classification pour présenter des algos de recommandation. Ainsi, le perceptron était un exemple d'algorithmes de recommandation basés sur le contenu et le filtrage, alors que le kPPV représente les algos basés uniquement sur le contenu.

Tout au long du travail nous avons utilisés des outils statistiques afin d'analyser les résultats obtenues (diverses graphiques, histogrammes, word clouds), nous avons ainsi réussi à obtenir une performance assez élevé de nos algorithmes. 

Un rapport bien plus détaillé de ce projet est donné dans le fichier Rapport Pima.

Ce projet a été codé en Python, en utilisant principalement la bibiothèque Numpy. Cependant, le projet était principalement axé autour de la compréhension du fonctionnement des algorithmes ainsi que des principes de bases du Machine Learning, ainsi le code pur n'était pas la seule priorité.

