# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:46:58 2021

@author: Yanis Zatout
"""

        
import cv2 as cv
import glob
import numpy as np
from numpy.linalg import svd
import numpy.linalg as npl
import matplotlib.pyplot as plt

path1='FG1/'
path2='FG2/'




pgms    = glob.glob(path1+'*.pgm')

pgms2   = glob.glob(path2+'*.pgm')
def affiche_image(images,numero_image=None,label=None):
    if label is None:
        label="J'ai la trique"
    if numero_image is None:
        cv.imshow(label,images)
        cv.waitKey(0)
    else:
        cv.imshow(label,images[numero_image])
        cv.waitKey(0)
    
def read_n_first_images(path,n=None):
    if n is None:
        images=[]
        for paths in path:
            images.append(cv.imread(paths,0))
        
        return images
    images=[]
    for paths in path[:n]:      
        images.append(cv.imread(paths,0))
    return images

def reshape_images(images,shape=(1,4096)):
    reshaped=images
    for i in range(len(images)):
        reshaped[i]=reshaped[i].reshape(shape)
    return reshaped


def create_M(FlattenedImages):
    meanImages=np.mean(FlattenedImages,axis=0).reshape(1,-1)
    stdImages=np.std(FlattenedImages,axis=0).reshape(1,-1)

    return np.concatenate((meanImages,stdImages),axis=0)

def affiche(S,k):
    pct_repr=np.cumsum(S)/np.sum(S)
    plt.plot(pct_repr[:k])
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Pourcentage de représentation")
    plt.title("Pourcentage de représentation de la variance")
    print('Avec {k_} directions principales, on a {rep} % de représentation des données'.format(k_=k,rep=pct_repr[k]))
    
def learn(path, number_of_images=100,k=50, plot=False):
    images = read_n_first_images(path,number_of_images)
    X=reshape_images(images)
    X=np.concatenate(X)
    M=create_M(X)
    Y=(X - M[0,:])/M[1,:]
    _,S,Vt=svd(Y)
    P=Vt.T
    Pk=P[:,:k]
    Z=Y@Pk
    if plot == True:
        affiche(S,k)
    return M,Pk,Z

def affiche_pairs(args,distances,n,train,test,tol=60):
    i = 0
    for arg,dist in zip(args[:n],distances[:n]):
        if dist <= tol:
            resized1 = cv.resize(train[i], (200,200), interpolation = cv.INTER_AREA)
            resized2 = cv.resize(test[arg], (200,200), interpolation = cv.INTER_AREA)
            
            cv.imshow("Image 1",resized1)
            cv.moveWindow("Image 1",400,600)
            cv.imshow("Image 2",resized2)
            cv.moveWindow("Image 2",600,600)
            cv.waitKey(0)      
            i+=1
    
def compute_Z(image,M,Pk):
    X=np.reshape(image,(len(image),-1))
    Y=(X - M[0])/M[1]
    Z=Y@Pk
    return Z


def ajout(image, M,Pk,Z):
    Z2=compute_Z(image,M,Pk)
    Z_appended=np.append(Z,Z2,axis=0)
    return Z_appended

def get_dist_images(Z_appended,z_test):
    di=[npl.norm(z_test-Z_appended[i,:],axis=1)for i in range(len(z_test[:,0]))]
    distances=[di[i].min() for i in range(len(di))]
    args=[di[i].argmin() for i in range(len(di))]
    return args,distances

def create_test_Z(path,M):
    test_set = read_n_first_images(pgms2)
    test_set = reshape_images(test_set)
    test_set = np.concatenate(test_set)
    normalised=(test_set - M[0])/M[1]
    zp=normalised@Pk
    
    return zp


M,Pk,Z=learn(pgms,1680,100)
train=read_n_first_images(pgms,1680)
    

test=read_n_first_images(pgms2)

zp = create_test_Z(pgms2,M)

args,distances=get_dist_images(Z,zp)

#affiche_pairs(args,distances,100,train,test,40)
