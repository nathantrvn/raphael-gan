#!/usr/bin/env python
# coding: utf-8

# # Genesis of Raphaël
# 
# Unleashing the power of DCNN to create random Raphaëls.

# In[52]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, Input, Dense, Flatten, Reshape, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model


# In[ ]:


def build_generator(latent_size):
    
    generator = Sequential([
        # (100)
        Input(shape=(100,), name="LatentNoise"),
        Dense(16*16*384, activation="relu"),
        # (16, 16, 384)
        Reshape((16, 16, 384)),
        # (32, 32, 192)
        Conv2DTranspose(192, 5, strides=2, padding="same", activation="relu"),
        BatchNormalization(),
        # (64, 64, 96)
        Conv2DTranspose(96, 5, strides=2, padding="same", activation="relu"),
        BatchNormalization(),
        # (128, 128, 3)
        Conv2DTranspose(3, 5, strides=2, padding="same", activation="tanh")
    ])
    
    return generator

def build_discriminator():
    
    discriminator = Sequential([
        # (128, 128, 3)
        Input(shape=(128, 128, 3), name="InputImages"),
        Conv2D(32, 3, padding="same", strides=1),
        LeakyReLu(0.2),
        SpatialDropout2D(0.2),
        
    ])


# In[11]:


import math as m


# In[31]:


128 * 128 * 96 / (64 * 64)


# In[32]:


64 * 64 * 192 / (32 * 32)


# In[34]:


7 * 7 * 384 / (3 * 3)


# In[40]:


32 * 32 * 384 / (14 * 14)


# In[47]:


128 * 3


# In[44]:


96 * 2


# In[45]:


192 * 2


# In[46]:


96 /2


# In[ ]:




