#!/usr/bin/env python
# coding: utf-8

# # Raphaël extended edition - Director's Cut
# 
# Because there's never enough Raphaël, we will now data augment him.

# In[370]:


import glob
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import PIL
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EXT_DATA = "../../data/external"


# ## Load the Raph

# In[26]:


filenames = pd.read_csv("../../data/rawdata.csv")


# In[27]:


display(filenames.head())


# In[48]:


imgs = [plt.imread(f["train_example"], format="jpeg") for _, f in filenames.iterrows()]


# In[49]:


plt.imshow(imgs[0])


# ## Square and center the Raph

# In[51]:


imgs[0].shape


# In[52]:


size = min(imgs[0].shape[0:2])
print(f"Chosen min size: {size}")


# In[53]:


margin = (max(imgs[0].shape[0:2]) - size) // 2
print(f"Margin: {margin}")


# In[54]:


if np.array(img.shape).argmax() == 0:
    cropped = imgs[0][margin:margin+size,:,:]
else:
    cropped = imgs[0][:,margin:margin+size,:]
plt.imshow(cropped)   
print(f"Shape: {cropped.shape}")


# In[158]:


cropped = []
for img in imgs:
    size = min(img.shape[0:2])
    margin = (max(img.shape[0:2]) - size) // 2
    if np.array(img.shape).argmax() == 0:
        cropped.append(Image.fromarray(img[margin:margin+size,:,:]).resize((128, 128), Image.ANTIALIAS))
    else:
        cropped.append(Image.fromarray(img[:,margin:margin+size,:]).resize((128, 128), Image.ANTIALIAS))


# In[160]:


plt.figure(figsize=(10, 5))
for i in range(1, 11):
    plt.subplot(2, 5, i)
    plt.imshow(cropped[i-1])


# In[208]:


for i, crop in enumerate(cropped):
    crop.convert("RGB").save(f"../../data/interim/{i}.png")


# In[209]:


filenames = pd.DataFrame([[f"../../data/interim/{i}.png", 1] for i in range(len(cropped))], columns=["train_example", "label"])


# In[210]:


display(filenames.head())
filenames.to_csv("../../data/interimdata.csv")


# ## Randomize the Raph

# In[364]:


augmenter = ImageDataGenerator(
    brightness_range=[0.8, 1.5],
    rotation_range=20, 
    zoom_range=0.1, 
    fill_mode='reflect', 
    horizontal_flip=True, 
    dtype='float64')


# In[271]:


filenames = pd.read_csv("../../data/interimdata.csv")
imgs = np.array([plt.imread(f["train_example"], format="jpeg") for _, f in filenames.iterrows()])


# In[272]:


plt.imshow(imgs[0])


# In[369]:


plt.imshow(augmenter.random_transform(imgs[0]).astype("uint8"))


# In[374]:


for i in range(120):
    plt.imsave(f"../../data/processed/{i}.png", augmenter.random_transform(imgs[random.randint(0, len(imgs) - 1)]).astype("uint8"))
for i, img in enumerate(imgs):
    plt.imsave(f"../../data/processed/{119 + i}.png", img)

