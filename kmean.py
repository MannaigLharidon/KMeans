# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:38:01 2018

@author: Mannaig L'Haridon
"""

from skimage import io
import numpy as np
import matplotlib.pyplot as plt


"""
#############################################################################
#                                                                           #
#                      TP : ALGORITHME DE LA K-MOYENNE                      #
#                                                                           #
#############################################################################
"""

        
def distCentroide(coulPixel,coulCtd):
    """   
    Entree : 
        - coulPixel : couleurs rvb d'un pixel
        - coulCtd : couleurs rvb d'un centroide
             
    Sortie : 
        - dist : distance euclidienne des couleurs d'un pixel de l'image à celles d'un centroide
    """
    dist = 0
    for rvb in range(3):
        dist += (coulPixel[rvb]-coulCtd[rvb])**2
    dist = np.sqrt(dist)
    return dist



def classePx(img,coulCtd,K):
    """
    Entrée :
        - img : image étudiée
        - coulCtd : couleurs rvb des centroides
        
    Sortie :
        - classe : nouvelle estimation de la classe d'appartenance des pixels
        - coulCtd : nouvelle estimation des positions des centroides
    """
    #Attribution d'une classe pour chaque pixel
    classe = np.zeros((img.shape[0],img.shape[1]))
    for l in range(img.shape[0]):
        for c in range(img.shape[1]):
            #Recuperation de la couleur RVB du pixel
            pixel = img[l][c]
            #Calcul des distances euclidiennes des pixels aux centroides
            distPxCtd = []
            for k in range(K):
                distPxCtd.append(distCentroide(pixel,coulCtd[k]))
            #Attribution de chaque pixel a un cluster
            attr = np.argmin(distPxCtd)
            classe[l][c] = attr
        
    #Réestimation des positions des centroides
    for k in range(K):
        ctdRVB = [0.,0.,0.]
        gpe = np.where(classe==k)
        for n in range(len(gpe[0])):
            for c in range(3):
                ctdRVB[c] += img[gpe[0][n]][gpe[1][n]][c]
        coulCtd[k] = np.divide(ctdRVB,len(gpe[0]))
        
    return classe,coulCtd
    

            
def Kmean(img,K):
    """
    Algorithme de la K-moyenne
    
    Entree :
        - img : image étudiée
        - K : nombre de centroides souhaités
        
    Sortie : 
        - classe1 : matrice de même taille que img, indiquant les classes d'appartenance
        - cpt : nombre d'itérations effectuées
    """

    # Definition des centroides selon K clusters
    coulCtd = np.zeros((K,3))
    for k in range(K):
        y = int(np.random.randint(0,img.shape[0]-1))
        x = int(np.random.randint(0,img.shape[1]-1))
        coulCtd[k] = img[y][x]

    #Attribution d'une première classe pour chaque pixel et réestimation des positions des centroides
    cpt = 0
    print("--> Itération : ",cpt)
    classe0 = np.zeros((img.shape[0],img.shape[1]))
    classe1, coulCtd = classePx(img,coulCtd,K)
    cpt +=1
    print("--> Itération : ",cpt)
    
    #Itérations jusqu'à ce que les objets arrêtent de changer de groupes
    while(np.any(classe0 != classe1)):
        #Attribution d'une nouvelle classe pour chaque pixel et réestimation des positions des centroides
        classe0 = classe1
        classe1, coulCtd = classePx(img,coulCtd,K)
        cpt += 1
        print("--> Itération : ",cpt)
            
    return classe1,cpt
    
    
    
    

if __name__ == "__main__" :
    
    # Chargement de l'image
    image = io.imread('dalleRVB.tif')
    plt.figure()
    plt.imshow(image)
    
    # Nombre K de centroides
    K = 3

    #Matrice des clusters
    classes,cpt = Kmean(image,K)
    print(cpt," itérations effectuées")
    plt.figure()
    plt.imshow(classes)
 