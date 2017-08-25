import cv2
import sys

def show_image(img):
	img  = cv2.resize(img, (960, 540))
	cv2.imshow('Image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def detect_face(gray_img):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

<<<<<<< Updated upstream
	faces = face_cascade.detectMultiScale(gray_img, 1.05,5)
=======
	faces = face_cascade.detectMultiScale(gray_img, 1.05, 5)
	return faces
	
	
if __name__=='__main__':
	img = cv2.imread(sys.argv[1])		
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = detect_face(gray_img)
>>>>>>> Stashed changes
	for (x,y,w,h) in faces:
		print("face detected")
		face = cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0),2)
	show_image(img);

