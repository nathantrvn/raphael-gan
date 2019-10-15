#!/usr/bin/env python
# coding: utf-8

# # Raphaël extended edition - Director's Cut
# 
# Because there's never enough Raphaël.

# In[45]:


import glob

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

EXT_DATA = "../../data/external"


# In[46]:


img = plt.imread(f"{EXT_DATA}/1.jpg", format="jpeg")


# In[47]:


plt.imshow(img)


# ## Square and center the Raph

# In[39]:


img.shape


# In[40]:


size = min(img.shape[0:2])
print(f"Chosen min size: {size}")


# In[41]:


margin = (max(img.shape[0:2]) - size) // 2
print(f"Margin: {margin}")


# In[42]:


if np.array(img.shape).argmax() == 0:
    cropped = img[margin:margin+size,:,:]
else:
    cropped = img[:,margin:margin+size,:]
plt.imshow(cropped)   
print(f"Shape: {cropped.shape}")


# ## Randomize the Raph

# In[34]:


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




