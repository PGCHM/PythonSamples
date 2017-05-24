#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 00:12:04 2017

ref: 
    
@author: JaniePG
"""
import numpy as np
import skimage
from skimage import io,filters

def channel_adjust(channel,values):
    orig_size = channel.shape
    flat_channel = channel.flatten()
    adjusted = np.interp(flat_channel, np.linspace(0, 1, len(values)),values)
    return adjusted.reshape(orig_size)
    
def Gotham_filter(orig_image):
    r = orig_image[:,:,0]
    g = orig_image[:,:,1]
    r_boost_lower = channel_adjust(r, [
        0, 0.05, 0.1, 0.2, 0.3, 
        0.5, 0.7, 0.8, 0.9,
        0.95, 1.0])
    g_more = np.clip(g+0.03, 0, 1.0)
    merged = np.stack([r_boost_lower, g_more, orig_image[:,:,2]], axis=2)
    blurred = filters.gaussian(merged, sigma = 10, multichannel = True)
    final = np.clip(merged*1.3 - blurred*0.3, 0, 1.0)
    g = final[:,:,1]
    g_adjusted = channel_adjust(g, [
        0, 0.047, 0.118, 0.251, 0.381,
        0.392, 0.42, 0.439, 0.475,
        0.561, 0.58, 0.627, 0.671,
        0.733, 0.847, 0.925, 1])
    final[:,:,1] = g_adjusted
    return final
    
original_image = skimage.img_as_float(io.imread("skyline.jpg"))
new_image = Gotham_filter(original_image)
io.imsave("skyline_filtered.jpg", new_image)
