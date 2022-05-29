# coding=<'utf-8'>

import sys

from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import numpy.ma as ma
import argparse

from pathlib import Path
import json
from PIL import Image
from PIL.TiffTags import TAGS
import importlib
import matplotlib.pyplot as plt


from pathlib import Path
from scipy.spatial.distance import cdist
from typing import List, Tuple, Any
from tqdm import tqdm
from skimage.feature import greycomatrix, greycoprops, canny
from skimage.filters import rank
from skimage.morphology import dilation, disk
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from matplotlib import patches as Patch

from skimage.measure import label
from skimage import morphology
import skfmm


def centroidFromMask(label_image, sizeCutoff=20):
    """
    From Henry and Samir's code
    Identify centroids of cells

    Arguments:
        label_image: image with regions labeled
        sizeCutoff: the size of salient regions, which eliminates false positives
            that are smaller than cutoff

    Return: list of centroids
    """
    p_lst = list(np.unique(label_image))
    p_lst.pop(0)

    centroids=[]
    for i in p_lst:
        if np.sum(label_image==i) < sizeCutoff:
            continue
        posr,posc=np.where(label_image==i)
        centroidr=sum(posr)//len(posr)
        centroidc=sum(posc)//len(posc)
        centroids.append([centroidr,centroidc])
    centroids=np.array(centroids)
    return centroids



def filter_size(label_image, thres=15):
    '''
    filtering based on the length and width
    label_iamge: input_image
    thres: threshold for size filtering
    '''
    new_img = label_image
    for i in range(1,np.max(label_image)+1):
        x, y = np.where(label_image == i)
        if max(x) - min(x) < thres or max(y) - min(y) < thres:
            new_img[new_img==i] = 0

    return new_img

def fmm_split_images(label_img, thres=40):
    '''
    given a label image, split the image based on fmm distance and return a new label image
    label_img: input image
    thres: threshold for fmm distance
    '''
    # get the labels
    p_lst, counts = np.unique(label_img, return_counts=True)
    background = p_lst[np.argmax(counts)]
    # n_p: new label to assign
    n_p = 100
    for i in p_lst:
        if i==background: continue
        x,y = np.where(label_img==i)
        mask_img, min_x, min_y, mask_x, mask_y = crop_image(label_img, x,y)
        label_img, n_p, p_lst = fmm_hard_thres(mask_img, min_x, min_y, thres,  label_img,mask_x,mask_y, n_p, p_lst)

    return label_img

def crop_image(label_img, x,y):
    '''
    crop the image to a smaller size to avoid unnessary calculations
    and mask other region
    label_img: input image
    x, y: coordinate of region to be masked
    '''

    # cropping image
    min_x, max_x, min_y, max_y = min(x), max(x), min(y), max(y)
    crop_img = np.copy(label_img[min_x:(max_x+1), min_y:(max_y+1)])

    # mask the region of input
    mask = np.ones((crop_img.shape))
    mask[x-min(x),y-min(y)] = 0
    mask_img = ma.masked_array(crop_img, mask = mask)

    return mask_img, min_x,min_y, x-min(x), y-min(y)

def fmm_hard_thres(mask_img,min_x, min_y, thres,  label_img, point_x, point_y, n_p, p_lst):
    '''
    split the component based on the hard threshold for fmm distance, iteratively starts from lowest y
    mask_img: the mask
    min_x, min_y: the corner location of the mask in whole image
    thres: threshold for distance
    label_img: whole image
    point_x, point_y: location of masked points
    n_p: label to be assigned
    p_lst: previous labels
    '''
    # get all masks locations
    temp_mask = np.zeros(mask_img.shape)
    temp_mask[point_x,point_y] = 1
    temp_label = label(temp_mask, connectivity=1)
    # for each component
    for j in np.unique(temp_label):
        if j==0: continue
        x,y = np.where(temp_label==j)
        # assign the min y to be starting point to calculate dist
        ind = np.argmin(y)
        min_xx = x[ind]
        min_yy = y[ind]
        temp_val = mask_img[min_xx,min_yy]
        mask_img[min_xx,min_yy] = 0
        # calculat dist, change the zero back
        dis = skfmm.distance(mask_img)
        mask_img[min_xx, min_yy] = temp_val

        # if no points are farther than dist
        if np.sum(dis>thres)==0:
            while n_p in p_lst:
                n_p +=10
            label_img[x+min_x,y+min_y] = n_p
            n_p+=10
        else:
            # if there are points are farther than dist, change the label for points within dist
            # and iterative split those points that are farther than dist
            re_x, re_y = np.where(np.logical_and(dis.data<=thres, dis.data!=0))
            while n_p in p_lst:
                n_p +=10
            label_img[re_x+min_x, re_y+min_y] = n_p
            n_p+=10
            new_crop_x, new_crop_y = np.where(dis.data>thres)

            new_mask_img, crop_min_x, crop_min_y, mask_x, mask_y = crop_image(label_img, new_crop_x+min_x, new_crop_y+min_y)
            label_img, n_p, p_lst = fmm_hard_thres(new_mask_img,crop_min_x, crop_min_y, thres,  label_img,mask_x, mask_y, n_p, p_lst)

    return label_img, n_p, p_lst






parser = argparse.ArgumentParser(description="Count Cells")
parser.add_argument('input_image', type=str)
# parser.add_argument('output_name', type=str)
parser.add_argument('--lam',dest='lam',  help='lambda for combining red and green channel',default=0.7, type=float)
parser.add_argument('--fthres',dest='fthres',  help='threshold for filtering small size region',default=8, type=int)
parser.add_argument('--sthres',dest='sthres',  help='threshold for splitting a region', default=25,type=int)
parser.add_argument('--winsize',dest='winsize',  help='window size for adaptive threshold', default=39,type=int)
parser.add_argument('--C',dest='C',  help='constant to substract for adaptive threshold', default=-8,type=int)
parser.add_argument('--dil',dest='dil',  help='size of dilation kernel', default=4,type=int)
parser.add_argument('--cutsize',dest='cutsize',  help='size for removing small region', default=20,type=int)
parser.add_argument('--plot',dest='plot',  help='determine whether to plot the figure', default=False,type=bool)
parser.add_argument('--plot_name',dest='plot_name',  help='if plot==True, the name of the plot',type=str)
parser.add_argument('--cutradius', dest='cutradius', help ='number of px from edge to exclude', default=0, type=int)

args = parser.parse_args()


output_count = []

# read image
img = cv2.imread(args.input_image, cv2.IMREAD_UNCHANGED)
inputheight, inputwidth, inputchannels = img.shape

# combine red and green channel
new_image = args.lam*img[:,:,1] + (1-args.lam)*img[:,:,2] #+ 0.3*img[:,:,0]
cv2.imwrite(args.input_image[0:len(args.input_image)-4] + '_processed.jpg', new_image)
new_img = cv2.imread(args.input_image[0:len(args.input_image)-4] + '_processed.jpg')
new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

# Adaptive thresholding
imgg1 = cv2.adaptiveThreshold(new_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,args.winsize,args.C)

# label image
label_image = label(imgg1)

label_image = filter_size(label_image, thres=args.fthres)
label_image[label_image>=1] = 255
label_image = np.uint8(label_image)

# dilate image
kernel_dilate = np.ones((args.dil,args.dil),np.uint8)

dilation = cv2.dilate(label_image,kernel_dilate,iterations = 1)


# reverse the label
label_image = np.zeros(dilation.shape)
label_image[dilation==0] = 255
label_image[dilation!=0] = 0

label_image = label(label_image)

# splitting by fmm
label_image = fmm_split_images(label_image, thres=args.sthres)

centroids = centroidFromMask(label_image, sizeCutoff=args.cutsize)


# moving points that fit radius condition to new ndarray
radius = args.cutradius

output_points_raw = []
# for row in centroids:  # Radius for all sides
#     if (radius < row[0] < (inputheight - radius)) and (radius < row[1] < (inputwidth - radius)):  # y-values, x-values
#         output_points_raw.append(row)
# output_points = np.asarray(output_points_raw)

for row in centroids:  # Radius excluding left and bottom only
    if (row[0] < (inputheight - radius)) and (radius < row[1]):  # y-values, x-values
        output_points_raw.append(row)
output_points = np.asarray(output_points_raw)

if args.plot:
    fig, axes = plt.subplots(ncols=2, figsize=(16,8))
    ax1, ax2 = axes.flatten()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax2.imshow(label_image, cmap='flag')

    for n in range(output_points.shape[0]):
        ax1.scatter(output_points[n][1], output_points[n][0], c='red')
        ax2.scatter(output_points[n][1], output_points[n][0], c='black')

    plt.savefig(args.plot_name)


# fw = open(args.output_name, 'w')

for cen in output_points:
    print(str(cen[1]) +','+ str(cen[0]))  # output in x,y (flip from output_points) 

#     fw.write(str(cen[1]) +','+ str(cen[0]) + '\n')
# fw.close()
