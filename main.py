import cv2
import sys

import face_detector as fd

def show_image(img):
	img  = cv2.resize(img, (960, 540))
	cv2.imshow('Image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__=='__main__':
	img = cv2.imread(sys.argv[1])		
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	faces = fd.detect_face(gray_img)
	#for (x,y,w,h) in faces:
	#	print("face detected")
	#	face = cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0),2)
	#show_image(img);
	
	
