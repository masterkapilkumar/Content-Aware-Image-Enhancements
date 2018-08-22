import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags

import wls_filter
import face_detector as fd
import skin_detector as sd

use_skin = False

def EACP(G, I, lambda_=0.2):
    
	print "Doing EACP"
	
	g, w, s = G.flatten(1), (np.ones(G.shape)).flatten(1), I.shape

	k = np.prod(s)
	dy = np.diff(I, 1, 0)
	dx = np.diff(I, 1, 1)
	
	dy = -lambda_ / (np.absolute(dy) ** 0.3 + 0.0001)
	dx = -lambda_ / (np.absolute(dx) ** 0.3 + 0.0001)
	
	dy = (np.vstack((dy, np.zeros(s[1], )))).flatten(1)
	dx = (np.hstack((dx, np.zeros(s[0], )[:, np.newaxis]))).flatten(1)
	
	d = w - (dx + np.roll(dx, s[0]) + dy + np.roll(dy, 1))
	a = spdiags(np.vstack((dx, dy)), [-s[0], -1], k, k)
	a = spdiags(d, 0, k, k) + a + a.T
	f = spsolve(a, w*g).reshape(s[::-1])
	
	return np.rollaxis(f,1)

def detect_bimodal(H):
	n = len(H)
	D,M,B = np.zeros(n),np.zeros(n),np.zeros(n)
	is_bimodal = np.zeros(n ,dtype=bool)
	maximas_Fs = [ argrelextrema(h, np.greater, order=10)[0] for h in H ]
	minimas_Fs = [ argrelextrema(h, np.less, order=10)[0] for h in H ]
	
	for i in range(n):
		hist = H[i]
		face_sum = np.sum(hist)
		if(len(maximas_Fs[i]) == 2 and len(minimas_Fs[i]) == 1):
			d,b,m = maximas_Fs[i][0], maximas_Fs[i][1], minimas_Fs[i][0]
			B[i], M[i], D[i] = b, m, d
			
			if(hist[m] <= 0.8*hist[d] and hist[m] <= 0.8*hist[b] and hist[d] >= 0.003*face_sum and hist[b] >= 0.003*face_sum):
				is_bimodal[i] = True
		elif(len(maximas_Fs[i]) > 2):
			print 'More than two maximas found in face', (i+1)
			# print(maximas_Fs[i])
			# print(minimas_Fs[i])
			d,b,m = maximas_Fs[i][0], maximas_Fs[i][-1], minimas_Fs[i][1]
			print(d,b,m)
			B[i], M[i], D[i] = b, m, d
			is_bimodal[i] = True
		elif(len(minimas_Fs[i]) > 1):
			print 'More than one minima found in face', (i+1)
			is_bimodal[i] = None
		else:
			print('face '+str(i+1)+' is not bimodal')
		plt.plot(hist)
		plt.xlim([1,256])
		plt.show()
	return is_bimodal, D, M, B

def sidelight_correction(orig_img, base_img, H, S, faces_rect):
	
	print "Starting sidelight correction..."
	n=len(H)
	is_bimodal, D, M, B = detect_bimodal(H)
	
	base_img_scaled = base_img*255
	A = np.ones(base_img_scaled.shape)
	
	for i in range(n):
		if(is_bimodal[i]):
			print "face "+str(i+1)+" bimodal"
			x, y, w, h = faces_rect[i]
			b,d,m = B[i], D[i], M[i]
			f = (b - d)/(m - d)
			A[y:y + h, x:x + w][(S[i]*255 <= m) & (S[i]*255 > 0)] = f
	
	if(not (A == 1).all()):
		A = EACP(A, orig_img, lambda_=120)
	else:
		print 'unit adjustment map'
	
	print "Sidelight correction done"
	print
	out = (A * base_img_scaled)
	return out


def exposure_correction(base_img, base_img_sidlit_corrected, skin_masks, faces_rect):
	
	print "Starting exposure correction..."
	n = len(faces_rect)
	enhanced_img = base_img_sidlit_corrected.copy()
	base_img_scaled = base_img*255
	A = np.ones(base_img_scaled.shape)
	info = np.iinfo(np.uint8)
	
	for i in range(n):
		x, y, w, h = faces_rect[i]
		skin_face = base_img_sidlit_corrected[y:y + h, x:x + w].copy()
		
		if use_skin:
			skin_face[ ~skin_masks[i] ] = 0
		 
		trimmed_skin = np.rint(skin_face).clip(info.min, info.max).astype(np.uint8)
		
		#calculate 75th percentile
		cumsum = cv2.calcHist([trimmed_skin], [0], None, [255], [1, 256]).T.ravel().cumsum()
		p = np.searchsorted(cumsum, cumsum[-1] * 0.75)
		
		if(p < 120):   #underexposed face
			
			print "face "+str(i+1)+" is underexposed!"
			
			f = (120+p)/(2*p + 0.000001)
			A[y:y + h, x:x + w][skin_face > 0] = f
			
	A = EACP(A, base_img_scaled, lambda_=120)
	enhanced_img = A * base_img_sidlit_corrected
	
	print "Exposure correction done"
	
	return enhanced_img

def face_enhance(img):
	LAB_Image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	
	print "Applying WLS Filter..."
	base, detail = wls_filter.wlsfilter_layer(LAB_Image[..., 0])
	print "WLS Filter done!"
	print
	
	print "Detecting faces..."
	faces_rect = fd.detect_face(img)
	if(faces_rect == []):
		print "No face detected"
		return img
	else:
		print str(len(faces_rect))+" faces detected"
	print
	
	faces=[]
	skin_masks = []
	base_faces = []
	S=[]
	H=[]
	info = np.iinfo(np.uint8)
	
	for (x,y,w,h) in faces_rect:
		faces.append(img[y:y + h, x:x + w].copy())
	
	print "Detecting skin..."
	for face in faces:
		skin_masks.append(sd.detect_skin(face)[1])
	
	skin_detected = False
	for mask in skin_masks:
		if mask.any():
			skin_detected = True
			break
	if not skin_detected:
		print 'No skin detected'
		return img
	print "Skin detection done..."
	print
	
	for (x,y,w,h) in faces_rect:
		base_faces.append(base[y:y + h, x:x + w].copy())
	
	for i in range(len(base_faces)):
		if use_skin:
			base_faces[i][~skin_masks[i]] = 0
		S.append(base_faces[i])

	print "Calculating smoothed histograms..."
	for skin in S:
		#magnify
		s = np.rint(skin*255)
		ss = s.clip(info.min, info.max).astype(np.uint8)
		
		# unsmoothed hist
		unsmoothed_h = cv2.calcHist([ss],[0],None,[255],[1,256]).T.ravel()
		# smoothed hist
		smoothed_h = np.correlate(unsmoothed_h, cv2.getGaussianKernel(30,10).ravel(), 'same')
		H.append(smoothed_h)
	print "Smoothed histograms calculated!"
	print
	
	out_temp = sidelight_correction(LAB_Image[..., 0], base, H, S, faces_rect)
	
	out_base = exposure_correction(base, out_temp, skin_masks, faces_rect)

	LAB_Image[..., 0] = np.rint(out_base + 255.0 * detail).clip(info.min, info.max).astype(np.uint8)
	
	outt = cv2.cvtColor(LAB_Image, cv2.COLOR_LAB2BGR)
	
	return outt



if __name__=='__main__':
	I_origin = cv2.imread(sys.argv[1])
	I_res = face_enhance(I_origin)
	cv2.imwrite("11.jpg",I_res)
	print "\nShowing enhanced image..."
	plt.imshow(cv2.cvtColor(np.hstack([I_origin, I_res]), cv2.COLOR_BGR2RGB))
	plt.title("Enhanced image")
	plt.show()

