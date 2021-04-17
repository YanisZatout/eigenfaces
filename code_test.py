# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:38:48 2021

@author: Yanis Zatout
"""

from . import eigenfaces
from . import show
import numpy as np
import matplotlib.pyplot as plt
import glob
path1='FG1/'
path2='FG2/'

pgms    = glob.glob(path1+'*.pgm')

pgms2   = glob.glob(path2+'*.pgm')

def read_images(path,n=None):
    """
    Algorithm that reads n first images from given path
    path: Given paths thanks to library glob.
    n: number of images to take. Default: None for all images.
    """
    if n is None:
        images=[]
        for paths in path:
            images.append(plt.imread(paths))
        
        return images
    images=[]
    for paths in path[:n]:      
        images.append(plt.imread(paths))
    return images


images=read_images(pgms)
images2=read_images(pgms2)

eig=eigenfaces()
eig.fit(images)

training_reconstruct = eig.reconstruct(eig.images[0])
training_reconstruct  = training_reconstruct.reshape(eig.image_shape)
show(np.concatenate((training_reconstruct, images[0]), axis=1 ))
plt.title('Reconstruction on the left vs real image on the right (training set) for {} eigenfaces'.format(eig.n_components))

plt.figure()

testing_reconstruct = eig.reconstruct(images2[0])
testing_reconstruct = testing_reconstruct.reshape(eig.image_shape)
show(np.concatenate((testing_reconstruct, images2[0]), axis=1 ))
plt.title('Reconstruction on the left vs real image on the right (testing set) for {} eigenfaces'.format(eig.n_components))

#I highly advise you to add to the number of eigenfaces if you want a descent accuracy. With the provided image set, about 700 eigenfaces should be enough