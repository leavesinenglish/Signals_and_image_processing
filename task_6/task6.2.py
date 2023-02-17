import cv2 as cv
import numpy as np


img = cv.imread('planet3.png')


def scaleImg(image, scale: float):
	_width, _height = image.shape[1], image.shape[0]
	return cv.resize(image, (int(_width * scale), int(_height * scale)))


img = scaleImg(img, 0.8)

width = img.shape[1]
height = img.shape[0]
total_area = width * height

font = cv.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 255, 0)
thickness = 1


def filter_contours(contours, tresh=0.002):
	filtered = []
	for contour in contours:
		contour_hull = cv.convexHull(contour)
		area = cv.contourArea(contour_hull)
		if area / total_area > tresh:
			filtered.append(contour)
	return filtered


def classify_area(_hull, _picture_width, _picture_height, grades=None):
	if grades is None:
		grades = [0.7, 0.3]
	_, _, rect_width, rect_height = cv.boundingRect(_hull)
	rect_area = rect_height * rect_width
	ratio = rect_area / total_area
	if ratio >= grades[0]:
		return ratio, 'large'
	elif ratio >= grades[1]:
		return ratio, 'medium'
	else:
		return ratio, 'small'


def getContours(image):
	edges_canny = cv.Canny(image, 1, 250)

	contours, hierarchy = cv.findContours(edges_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	filtered_contours = filter_contours(contours)

	total_contour = np.vstack(filtered_contours)
	total_hull = cv.convexHull(total_contour)

	return total_hull, total_contour


overlay = np.zeros(img.shape, dtype=np.uint8)
overlay[:, :, :] = 0
while True:

	hull, total_contour = getContours(img)

	area_ratio, size_label = classify_area(hull, width, height)

	x, y, w, h = cv.boundingRect(hull)
	cv.putText(overlay, 'area: ' + "{:.1f}".format(area_ratio * 100) + '%',
			   (np.int32(x + w / 2 - 40), np.int32(y + h / 2)), font, 0.6, fontColor, thickness)
	cv.putText(overlay, size_label,
			   (np.int32(x + w / 2 - 40), np.int32(y + h / 2) + 15), font, 0.6, fontColor, thickness)

	cv.drawContours(overlay, [hull], 0, (255, 10, 10))

	cv.imshow('task 6.2', cv.addWeighted(img, 0.4, overlay, 0.6, 0.0))

	k = cv.waitKey(1)
	if k == 27:  # ESC key
		break

cv.destroyAllWindows()
exit()
