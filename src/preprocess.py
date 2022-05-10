import cv2 as cv
import numpy as np


def show_image(image, window_name):

    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def resize_image(image, scale_ratio):
    image_height, image_width = image.shape[:2]

    
    width = int(image_height*scale_ratio)
    height = int(image_width*scale_ratio)

    dimension = (width, height)

    return cv.resize(image, dimension, interpolation=cv.INTER_AREA)


def apply_canny_edge_detection(image):
    low_threshold = 100
    ratio = 5

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # To remove undesirable edges
    blur_image = cv.GaussianBlur(gray_image, ksize=(3, 3), sigmaX=0)

    canny_edge_detected_image = cv.Canny(
        blur_image, threshold1=low_threshold, threshold2=low_threshold*ratio)

    return canny_edge_detected_image


def remove_horizontal_line(image):
    image = image.copy()

    # Specify size on horizontal axis
    width = image.shape[1]
    horizontal_size = width//70

    # Create structure element for extracting horizonatal lines through morphology operations
    horizontal_structure = cv.getStructuringElement(
        cv.MORPH_RECT, (horizontal_size, 1))

    # Erosion followed by dilation
    horizontal_edge_detected_image = cv.morphologyEx(
        image, cv.MORPH_OPEN, horizontal_structure)

    # Inverse horizontal edge detected image
    horizontal_edge_detected_image = cv.bitwise_not(
        horizontal_edge_detected_image)
    # show_image(horizontal_edge_detected_image, "image")

    final_image = cv.bitwise_and(
        image, image, mask=horizontal_edge_detected_image)
    return final_image


def remove_vertical_line(image):
    image = image.copy()

    # Specify size on vertical axis
    height = image.shape[0]
    vertical_size = height//70

    # Create structure element for extracting vertical lines through morphology operations
    vertical_structure = cv.getStructuringElement(
        cv.MORPH_RECT, (1, vertical_size))

    # Erosion followed by dilation
    vertical_edge_detected_image = cv.morphologyEx(
        image, cv.MORPH_OPEN, vertical_structure)

    # Inverse vertical edge detected image
    vertical_edge_detected_image = cv.bitwise_not(vertical_edge_detected_image)

    final_image = cv.bitwise_and(
        image, image, mask=vertical_edge_detected_image)
    return final_image


def apply_adaptive_threshold(image):
    # gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur_image = cv.GaussianBlur(image, ksize=(3, 3), sigmaX=0)
    thresholded_image = cv.threshold(
        blur_image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    return thresholded_image


def apply_dilation(image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 2))
    dilated_image = cv.dilate(image, kernel, iterations=2)
    return dilated_image
