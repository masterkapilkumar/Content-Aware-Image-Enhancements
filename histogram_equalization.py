import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1],0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('res.png',res)