import cv2
import sys
import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

def check_neighbor(mask):
	neighbor = np.ones([4,4], dtype='float')
	return correlate2d(mask.astype('float')/255.0, neighbor, mode='same', boundary='wrap') >= 1

def detect_skin(img):
	
	h,w = img.shape[0:2]
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(h/16,w/16))

	LAB_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype('float')
	skin = 255*np.ones(img.shape, img.dtype)
	
	#ellipse test
	ellipse1 = ((LAB_image[...,1]-143)/6.5)**2 + ((LAB_image[...,2]-148)/12.0)**2
	e1 = ellipse1 < (1.0/0.6)     #bound/prob
	
	#hsv threshold
	HSV_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float')
	h = (HSV_image[...,1]>=0.25*255) & (HSV_image[...,1]<=0.75*255) & (HSV_image[...,0]<0.095*180)

	skinMask = np.zeros(img.shape[0:2], img.dtype)
	skinMask[e1 & h] = 255
	
	ellipse2 = ((LAB_image[...,1]-143)/6.5)**2 + ((LAB_image[...,2]-148)/12.0)**2
	e2 = ellipse2 < (1.25/0.6)     #bound/prob
	
	skinMask[(skinMask == 0) & e2 & check_neighbor(skinMask)] = 255
	skinMask_closed = cv2.morphologyEx(skinMask,cv2.MORPH_CLOSE,kernel)
	
	skin = cv2.bitwise_and(img, img, mask=skinMask_closed)
	return skin, (skinMask_closed/255).astype(bool)

def main():
	img = cv2.imread(sys.argv[1])
	skin, _ = detect_skin( img )
	plt.imshow( cv2.cvtColor(np.hstack([img, skin]), cv2.COLOR_BGR2RGB) )
	plt.show()
	return 0

if __name__ == '__main__':
	main()
