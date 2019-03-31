import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2


years = [0, 1, 2, 3, 4]
size = 32
i = [np.load('allimg_2014.npy'),
     np.load('allimg_2015.npy'),
     np.load('allimg_2016.npy'),
     np.load('allimg_2017.npy'),
     np.load('allimg_2018.npy')]

dataset = []

sinn = np.zeros((18, 875, 515, 3))
for y in tqdm(years):
    for n, im in enumerate(i[y]):
        sinn[n] = cv2.resize(im, (515, 875))

    sout = plt.imread('map'+str(2014+y)+'.png')[:, :, 0]
    for x in range(0, sout.shape[0], size//2):
        for y in range(0, sout.shape[1], size//2):
            if ~(sout[x, y] == 0.):
                target = np.mean(sout[(x - size):(x + size),
                                      (y - size):(y + size)])

                inputs = sinn[:, (x - size):(x + size),
                                 (y - size):(y + size)]

                dataset.append(np.asarray([inputs, target]))


np.save('dataset.npy', dataset)
