

import cv2
import numpy as np

def image_masking(filepath):
	
	BLUR = 21
	CANNY_THRESH_1 = 10
	CANNY_THRESH_2 = 200
	MASK_DILATE_ITER = 10
	MASK_ERODE_ITER = 10
	MASK_COLOR = (0.0,0.0,0.0) # In BGR format
	
	img = cv2.imread(filepath)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#cv2.imshow("img", img)
	#cv2.waitKey()

	#edges
	#cv2.destroyAllWindows()
	edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
	edges = cv2.dilate(edges, None)
	edges = cv2.erode(edges, None)
	#cv2.imshow("edges", edges)
	#cv2.waitKey()

	contour_info = []
	contours, __ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	
	for c in contours:
	    contour_info.append((c, cv2.isContourConvex(c), cv2.contourArea(c),))
	contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

	max_contour = contour_info[0]
	mask = np.zeros(edges.shape)
	cv2.fillConvexPoly(mask, max_contour[0], (255))

	mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
	mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
	mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
	
	mask_stack = np.dstack([mask]*3)
	mask_stack  = mask_stack.astype('float32') / 255.0
	img = img.astype('float32') / 255.0
	cv2.imshow("mask", mask)
	cv2.waitKey()
	masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)
	masked = (masked * 255).astype('uint8')
	cv2.imshow("masked", masked)
	cv2.waitKey()

	fileName, fileExtension = filepath.split('.')
	fileName += '-masked.'
	filepath = fileName + fileExtension
	print (filepath)

	cv2.imwrite(filepath, masked)

if __name__ == '__main__':
	filepath = input("Enter Image File Name:\n")
	image_masking(filepath)
