# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 05:23:19 2021

@author: Yanis Zatout
"""

        
import cv2 as cv
import glob
import numpy as np
from numpy.linalg import svd
#import numpy.linalg as npl
import matplotlib.pyplot as plt

def show(image, cmap="gray"):
    """
    Simple image representation function:
    image: Input image having a good shape
    """
    plt.clf()
    plt.imshow(image,cmap=cmap)

class eigenfaces():
    #TODO: Implement eigenfaces based on representation percentage
    def __init__(self,n_components = 100, pct_repr=.9):
        self.centered_images=None      
        self.eigenfaces = None
        self.images=None
        self.n_components=n_components
        self.pct_repr=pct_repr
        
    def __str__(self):
        if self.images is None:
            return('Eigenfaces algorithm \n'            
                   'Number of eigenfaces: {}'.format(self.n_components))
        else:
            return ('Eigenfaces algorithm \n'            
                    'Number of eigenfaces: {} \n'
                    'Number of images:     {} \n'
                    'Expected precision:   {}'
                    .format(self.n_components,self.n_images,round(self.explained_variance_ratio_,4)))
            
        

        
    def __repr__(self):
        return self.__str__()
        
    def prepare_images(self,images):
        self.images=images
        self.total_pixels=np.size(self.images[0])
        self.image_shape=self.images[0].shape
        self.n_images=len(self.images)
        self.reshaped_images=np.reshape(images, (self.n_images,-1))
    def compute_mean_and_std(self):
        self.mean_image=np.mean(self.reshaped_images,axis=0)
        self.std_image=np.std(self.reshaped_images,axis=0)


        
    def center(self):
        if self.centered_images is None:
            self.compute_mean_and_std()
            self.centered_images=self.reshaped_images-self.mean_image.reshape(1,-1)
            return

        #self.normalised_images=self.centered_images/self.std_image.reshape(self.total_pixels,1)

    def view(self,image):
        show(image.reshape(self.image_shape))        
        
    def fit(self,images):        
        self.prepare_images(images)
        self.center()
        _,self.S,self.Vr=svd(self.centered_images)
        self.Vr=self.Vr.T
        self.explained_variance_ratio_=(np.cumsum(self.S)/np.sum(self.S))[self.n_components-1]
        #Matrix to keep
        self.eigenfaces=self.Vr[:,:self.n_components]
        
    def T(self):
        try:
            return self.eigenfaces.T[key]
        except TypeError:
            print('You might want to fit on your images before trying to access eigenfaces')
            return

    def __getitem__(self,key):
        try:
            return self.eigenfaces[key]
        except TypeError:
            print('You might want to fit on your images before trying to access eigenfaces')
            return
    def reconstruct(self,image):
        return image.reshape(1,-1)@eig.eigenfaces@eig.eigenfaces.T+eig.mean_image
