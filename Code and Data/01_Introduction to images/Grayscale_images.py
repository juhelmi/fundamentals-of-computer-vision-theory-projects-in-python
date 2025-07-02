import numpy as np
import matplotlib.pyplot as plt

import cv2

im = np.arange(256)
im = im[np.newaxis,:]
im = np.repeat(im,100,axis=0)
plt.imshow(im,cmap='gray')

im = plt.imread(r'albert-einstein_gray.jpg')
im.setflags(write=1)
#im = cv2.imread(r'albert-einstein_gray.jpg',cv2.IMREAD_GRAYSCALE)

print(f"Image flags {im.flags}")
print(type(im))
print(im.shape)

plt.imshow(im,cmap='gray')
cv2.imshow("First image", im)

# if cv2.waitKey(0) == ord('q'):
#     quit()

im[23:100,40:100] = 255
im[23,100] = 200
im2 = im.copy()
im2[23,100] = 200
im2[23:100,40:100] = 255
plt.imshow(im2,cmap='gray')

cv2.imshow("2", im2)

im2[300:400,40:100] = 0
plt.imshow(im2,cmap='gray')

if cv2.waitKey(0) == ord('q'):
    quit()