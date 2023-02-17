import cv2 as cv
import numpy as np


# TODO: взять реальные примеры форм, например планеты


def scaleImg(image, scale: float):
	width, height = image.shape[1], image.shape[0]
	return cv.resize(image, (int(width * scale), int(height * scale)))


x_pos = 0
y_pos = 0
scale = 0.4
orig_img = cv.imread('planet3.png')
img = np.copy(orig_img)
img = scaleImg(img, scale)

cv.namedWindow('Original')
cv.imshow('Original', scaleImg(orig_img, scale))


def draw_bounding_boxes(_canvas, contours_to_draw):
	for _contour in contours_to_draw:
		draw_contour_bounds(_contour, _canvas)


def draw_contour_bounds(_contour, _canvas, color=(0, 255, 0), contour_thickness=1):
	x, y, w, h = cv.boundingRect(_contour)
	cv.rectangle(_canvas, (x, y), (x + w, y + h), color, contour_thickness)


def onMouse(event, x, y, flags, param):
	global x_pos, y_pos
	x = np.clip(x, 0, img.shape[1] - 1)
	y = np.clip(y, 0, img.shape[0] - 1)
	x_pos = x
	y_pos = y


def filter_contours(contours, tresh=100):
	filtered = []
	approximated = []
	for c in contours:
		_hull = cv.convexHull(c)
		approx = approximate_contour(c)
		if cv.contourArea(_hull) > tresh:
			filtered.append(c)
			approximated.append(approx)
	return filtered, approximated


def approximate_contour(cntr):
	perimeter = cv.arcLength(cntr, True)
	approx = cv.approxPolyDP(cntr, 0.02 * perimeter, True)
	return approx


def define_shape(approximated_shape):
	if len(approximated_shape) == 3:
		shape = "triangle"
	elif len(approximated_shape) == 4:
		x, y, w, h = cv.boundingRect(approximated_shape)
		shape = "square" if np.abs(w - h) <= 10 else "rectangle"
	elif len(approximated_shape) == 5:
		shape = "pentagon"
	elif len(approximated_shape) == 6:
		shape = "hexagon"
	else:
		x, y, w, h = cv.boundingRect(approximated_shape)
		shape = "circle" if np.abs(w - h) <= 10 else "ellipse"

	return shape


cv.namedWindow('Types')
cv.setMouseCallback('Types', onMouse)
figure_types_layer = np.zeros(img.shape, dtype=np.uint8)

while True:
	edges_canny = cv.Canny(img, 10, 250)
	contours, _ = cv.findContours(edges_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	filtered_contours, approximated_contours = filter_contours(contours, 10000)
	figure_types_layer[:, :, :] = 0  # Reset the layer

	draw_bounding_boxes(figure_types_layer, filtered_contours)

	for idx, contour in enumerate(filtered_contours):
		x, y, w, h = cv.boundingRect(contour)
		point_is_in_bounds = x <= x_pos <= x + w and y <= y_pos <= y + h
		if point_is_in_bounds:
			draw_contour_bounds(contour, figure_types_layer, (0, 0, 255), 2)  # To outline current object with another color
			shape_type = define_shape(approximated_contours[idx])
			cv.putText(figure_types_layer, 'Type: ' + shape_type, (np.int32(x + w / 8), np.int32(y + h / 2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (210, 255, 100), 1)

	figure_types_layer = cv.add(cv.cvtColor(edges_canny, cv.COLOR_GRAY2BGR), figure_types_layer)  # Real edges
	cv.drawContours(figure_types_layer, approximated_contours, -1, (0, 200, 255), 1)  # Approximated edges
	cv.imshow('Types', figure_types_layer)

	k = cv.waitKey(1) & 0xFF
	if k == 27:
		break
cv.destroyAllWindows()
exit()
