#!/usr/bin/env python
# coding: utf-8

# # Raphaël extended edition - Director's Cut
# 
# Because there's never enough Raphaël.

# In[1]:


import glob

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import   ImageDataGenerator

EXT_DATA = "../../data/external"


# In[2]:


img = plt.imread(f"{EXT_DATA}/1.jpg", format="jpeg")


# In[3]:


plt.imshow(img)


# ## Square and center the Raph

# In[4]:


img.shape


# In[5]:


size = min(img.shape[0:2])
print(f"Chosen min size: {size}")


# In[6]:


margin = (max(img.shape[0:2]) - size) // 2
print(f"Margin: {margin}")


# In[7]:


if np.array(img.shape).argmax() == 0:
    cropped = img[margin:margin+size,:,:]
else:
    cropped = img[:,margin:margin+size,:]
plt.imshow(cropped)   
print(f"Shape: {cropped.shape}")


# ## Randomize the Raph

# In[8]:


augmenter = ImageDataGenerator(
    featurewise_center=True, 
    featurewise_std_normalization=True, 
    rotation_range=15, 
    brightness_range=(0.1, 0.1), 
    zoom_range=(-0.1, 0.1), 
    fill_mode='reflect', 
    horizontal_flip=True, 
    dtype='float32')


# In[ ]:




