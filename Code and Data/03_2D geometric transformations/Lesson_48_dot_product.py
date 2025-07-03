import cv2
import numpy as np
import matplotlib.pyplot as plt

grayImage = r'../01_Introduction to images/albert-einstein_gray.jpg'
colourImage = r'../01_Introduction to images/tulips.jpg'

I_gray = cv2.imread(grayImage,cv2.IMREAD_GRAYSCALE)
I_BGR = cv2.imread(colourImage)

#plt.imshow(I_gray,cmap='gray')
#plt.imshow(I_BGR[:,:,::-1])
#plt.show()

S = np.array([[2,0],[0,2]])

numRows = I_gray.shape[0]
numCols = I_gray.shape[1]

I2 = np.ones((2*numRows,2*numCols),dtype='uint8')*255

for i in range(numRows):
    for j in range(numCols):
        P = np.array([i,j])
        P_dash = S.dot(P)
        new_i , new_j = P_dash[0] , P_dash[1]
        I2[new_i,new_j] = I_gray[i,j]

#cv2.imshow("I2", I2)
print(I2.shape)

import matplotlib as mpl
def displayImageInActualSize(I):
    dpi = mpl.rcParams['figure.dpi']
    H,W = I.shape
    figSize = W/float(dpi) , H/float(dpi)
    fig = plt.figure(figsize = figSize)
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off')
    ax.imshow(I,cmap='gray')
    print(f"Time to show {I.shape}")
    plt.show()

#displayImageInActualSize(I2)

print("Next in double size")

S = np.array([[2,0],[0,2]])
I2 = np.zeros((2*numRows,2*numCols),dtype='uint8')
Tinv = np.linalg.inv(S)
#T_1 = np.dot(Tinv, S)
for new_i in range(I2.shape[0]):
    for new_j in range(I2.shape[1]):
        P_dash = np.array([new_i,new_j])
        P = Tinv.dot(P_dash)
        P = np.int16(np.round(P))
        i , j = P[0] , P[1]
        if i < 0 or i>=numRows or j<0 or j>=numCols:
            pass
        else:
            I2[new_i,new_j] = I_gray[i,j]

displayImageInActualSize(I2)

print("Wait end")
cv2.waitKey(0)
