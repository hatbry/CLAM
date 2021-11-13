from matplotlib.pyplot import sca
import openslide as osh
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from tqdm import tqdm, trange
from PIL import Image
import sys

vis_level = 2
_patch_size = 256
start = time.time()
slide_path = '/Users/brycehatfield/Documents/Programming/Slides/Root Dir/slides/9 3412 2-H-1 HCC.svs'

slide = osh.OpenSlide(slide_path)
downsample = slide.level_downsamples[vis_level]
dim_w, dim_h = slide.level_dimensions[0]


print('w, h: ', dim_w, dim_h)


def get_all_coords():
    coords = []
    print('dim_w: ', dim_w)
    print('dim_h', dim_h)
    print('patch: ', _patch_size)
    num_w = dim_w // _patch_size
    num_h = dim_h // _patch_size
    print(num_w, num_h, _patch_size)
    for y in range(num_h):
        for x in range(num_w):
            coords.append([x * _patch_size, y * _patch_size])
    return np.array(coords)


def get_scores(coord_len):
    scores = []
    for _ in range(coord_len):
        a = random.random()
        scores.append(a)
    return scores

def assertLevelDownsamples():
    level_downsamples = []
    dim_0 = slide.level_dimensions[0]
    
    for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):
        print('Assert Down: ', downsample)
        print('Assert Dim: ', dim)
        estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
        print('Estimated Downsample: ', estimated_downsample)
        level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))
    
    return level_downsamples




scale = [1/downsample, 1/downsample]

region_size = slide.level_dimensions[vis_level]

w, h = region_size
patch_size = np.ceil(np.array(_patch_size) * np.array(scale)).astype(int)
coords = get_all_coords()
scores = get_scores(len(coords))
print('Original coords shape: ', coords.shape, coords[:5])
coords = np.ceil(coords * np.array(scale)).astype(int)
overlay = np.full(np.flip(region_size), 0).astype(float)
counter = np.full(np.flip(region_size), 0).astype(np.uint16)

for idx in trange(len(coords)):
    score = scores[idx]
    coord = coords[idx]

    if score >= 0.5:
        pass
    else:
        score = 0.0

    overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score

overlay = cv2.GaussianBlur(overlay, tuple((patch_size).astype(int) * 2 + 1), 0)

img = np.array(slide.read_region((0, 0), vis_level, region_size).convert('RGB'))

cmap = plt.get_cmap('coolwarm')



for idx in trange(len(coords)):
    score = scores[idx]
    coord = coords[idx]

    raw_block = overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]
    img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()
    color_block = (cmap(raw_block) * 255)[:,:,:3].astype(np.uint8)

    img_block = color_block

    img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = img_block.copy()

del overlay

fresh_img = np.array(slide.read_region((0, 0), vis_level, region_size).convert('RGB'))
img = cv2.addWeighted(img, 0.4, fresh_img, 0.6, 0, fresh_img)
img = Image.fromarray(img)
img.show()

print('Done in {} seconds'.format(round(time.time() - start, 5)))
