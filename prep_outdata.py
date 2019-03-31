import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


years = ['2013', '2015', '2017', '2019']
years_num = [0, 1, 2, 3]
regions = ['2', '3', '4', '5']
regions_num = [0, 1, 2, 3]

protein_region_year = np.zeros([4, 4])
for y in years_num:
    d = pd.read_csv(years[y]+'Silage.csv')
    protein_variety = d['CP'][1:].values.astype(float)/100
    protein_variety *= d['CP'][0]

    for r in regions_num:
        yield_variety = d[regions[r]][1:].values.astype(float)/100
        protein_region = np.multiply(protein_variety, yield_variety)
        protein_region = np.mean(protein_region)

        protein_region_year[y, r] = protein_region

print(protein_region_year)


canvas = np.zeros([4, 875, 515])
for y in years_num:
    for r in regions_num:
        mask = plt.imread('region'+regions[r]+'.png')[:, :, -1]
        mask *= protein_region_year[y, r]

        canvas[y] += mask


canvas_interp = np.zeros([7, 875, 515])

years_num_new = [0, 1, 2, 3, 4, 5, 6]
for y in years_num_new:
    if y in [0, 2, 4, 6]:
        canvas_interp[y] = canvas[y//2]
    else:
        canvas_interp[y] = (canvas[y//2] + canvas[1 + y//2])/2

    img = canvas_interp[y]
    plt.imsave('map' + str(2013 + y) + '.png', img,  cmap='gray')

plt.imsave('template.png', canvas_interp[-1]>0, cmap='gray')

