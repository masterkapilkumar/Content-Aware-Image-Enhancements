import cv2
import sys
import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

def display(img, name='', mode='bgr'):
    """
    display image using matplotlib
    ARGS:
    img: bgr mode
    name: string, displayed as title
    """
    if mode == 'bgr':
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif mode == 'rgb':
        plt.imshow(img)
    elif mode == 'gray':
        plt.imshow(img, 'gray')
    elif mode == 'rainbow': # for 0-1 img
        plt.imshow(img, cmap='rainbow')
    else:
        raise ValueError('CAPE display: unkown mode')
    plt.title(name)
    plt.show()

def HSV_threshold(H, S):
	return (S>=0.25*255) & (S<=0.75*255) & (H<0.095*180)

def ellipse_test(A, B, bound=1.0, prob=1.0, return_prob=False):
	elpse = (1.0*(A-143)/6.5)**2 + (1.0*(B-148)/12.0)**2
	if not return_prob:
		return elpse < (1.0*bound/prob)
	else:
		return np.minimum(1.0/(elpse+1e-6), 1.0)

def check_neighbor(mask):
	neighbor = np.ones([4,4], dtype='float')
	return correlate2d(mask.astype('float')/255.0, neighbor, mode='same', boundary='wrap') >= 1

def detect_skin(img):
	
	skinMask = np.zeros(img.shape[0:2], img.dtype)
	img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype('float')
	img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float')
	
	e1 = ellipse_test(img_LAB[...,1], img_LAB[...,2], bound=1.0, prob=0.6)
	h = HSV_threshold(img_HSV[...,0], img_HSV[...,1]) 

	skinMask[e1 & h] = 255
	
	e2 = ellipse_test(img_LAB[...,1], img_LAB[...,2], bound=1.25, prob=0.6)
	
	skinMask[(skinMask == 0) & e2 & check_neighbor(skinMask)] = 255
	_h,_w = img.shape[0:2]
	_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(_h/16,_w/16))
	skinMask_closed = cv2.morphologyEx(skinMask,cv2.MORPH_CLOSE,_kernel)
	
	skin = 255*np.ones(img.shape, img.dtype)
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
