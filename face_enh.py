import cv2
import sys

import wls_filter

def face_enhance(img, eacp_lambda=0.2):
	_I_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	I = _I_LAB[..., 0]
	Base, Detail = wls_filter.wlsfilter(I)
	I_out = Base  # float [0,1]

	I_aindane = aindane.aindane(I_origin)
	faces_xywh = vj_face.face_detect(I_aindane)
	faces = [I_origin[y:y + h, x:x + w] for (x, y, w, h) in faces_xywh]
	skin_masks = [apa_skin.skin_detect(face) for face in faces]
	_any_skin = False
	for _mask in skin_masks:
	_any_skin |= ((_mask[1]).any())
	if (faces_xywh == []) or (not _any_skin):  # face not detected
	print 'face or skin not detected!'
	return I_origin
	_I_out_faces = [I_out[y:y + h, x:x + w] for (x, y, w, h) in faces_xywh]  #float[0,1]
	S = [ cape_util.mask_skin(_I_out_faces[i], skin_masks[i][1]) \
	  for i in range(len(_I_out_faces)) ]  # float [0,1]

	# to visualize detected skin and it's (unsmoothed) hist
	for idx, s in enumerate(S):
	cape_util.display(cape_util.mag(s),
			  name='detected skin of L channel',
			  mode='gray')
	# plot original hist(rectangles form, of S). don't include 0(masked)
	plt.hist(cape_util.mag(s).ravel(), 255, [1, 256])
	plt.xlim([1, 256])
	plt.show()
	## save image data for R silverman analysis
	import os
	DATA_DIR = os.path.expanduser('~/Dropbox/dropcode/r/')
	# save skin pixels (value neq 0.0)
	np.savetxt(DATA_DIR+str(idx)+'.csv', s[s!=0.0].ravel(), delimiter=',')

	H = [cape_util.get_smoothed_hist(cape_util.mag(s)) for s in S]
	# visualize smoothed hist
	for h in H:
	plt.plot(h)
	plt.xlim([1, 256])
	plt.show()

	I_out_side_crr = sidelight_correction(_I_LAB[..., 0], I_out, H, S,
					  faces_xywh, _eacp_lambda_)
	I_out_expo_crr = exposure_correction(_I_LAB[..., 0], I_out, I_out_side_crr,
					 skin_masks, faces_xywh, _eacp_lambda_)

	_I_LAB[..., 0] = cape_util.mag((I_out_expo_crr + 255.0 * Detail), 'trim')
	I_res = cv2.cvtColor(_I_LAB, cv2.COLOR_LAB2BGR)
	return I_res	

