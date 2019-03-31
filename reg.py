import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import SimpleITK

static = plt.imread('MapMaskHighRes.png')[:, :, 0]
moving = plt.imread('alberta.png')[:, :, 0]

print(moving.shape)
print(static.shape)

moving = cv2.resize(moving, (static.shape[1], static.shape[0]))
print(static.shape)


# Read image
im_in = moving*255/np.amax(moving)
# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.

th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)

# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255)

# Invert floodfilled image
moving = cv2.bitwise_not(im_floodfill)
plt.figure()
plt.subplot(131)
plt.imshow(static)
plt.subplot(132)
plt.imshow(moving)



resultImage = SimpleITK.Elastix(static, moving)
plt.subplot(133)
plt.imshow(resultImage)
plt.show()


print(moving.shape)
print(static.shape)
mse = 1
num_pixels = static.shape[0]*static.shape[1]
for scale_x in tqdm(np.arange(0, 2, 0.1)):
    for dis_x in np.arange(-2, 2, 0.1):
        for dis_y in np.arange(-2, 2, 0.1):
            T = cv2.UMat(np.array([[scale_x, 0, dis_x],
                                    [0, 1, dis_y]],
                                    dtype=np.float32))

            h, w = moving.shape
            transformed = cv2.warpAffine(moving, T, (w, h)).get()

            mse_new = np.square(
                transformed - static)/num_pixels
            mse_new = np.mean(mse_new)
            if mse_new < mse:
                mse = mse_new
                best_T = T.get()
                print(mse)

print(best_T)
plt.figure()
plt.subplot(131)
plt.imshow(static)
plt.subplot(132)
plt.imshow(moving)

plt.imsave('transformed.png', transformed)
plt.subplot(133)
plt.imshow(transformed)
plt.show()
