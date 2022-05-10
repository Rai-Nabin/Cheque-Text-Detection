import cv2 as cv
import numpy as np
from src.sort_contours import sort_contours


def show_image(image, window_name):

    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_image_size(image):

    height = image.shape[0]
    width = image.shape[1]

    return (height, width)


def apply_filter(gray_image):
    kernel = np.ones((5, 5), np.float32)/10
    filtered_2D_image = cv.filter2D(gray_image, ddepth=-1, kernel=kernel)
    # show_image(filtered_2D_image, "Filtered Image")
    return filtered_2D_image


def apply_threshold(filtered_2D_image):

    # Removes noise while keeping edges sharp
    filtered_image = cv.bilateralFilter(filtered_2D_image, 75, 75, 75)
    # filtered_image = cv.GaussianBlur(filtered_2D_image, (5, 5), 0)

    thresholded_image = cv.threshold(
        filtered_image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    # show_image(thresholded_image, "Thresholded image")

    return thresholded_image


def find_contours(thresholded_image):
    # Contours are the boundaries of a shape with same intenstity
    contours, _ = cv.findContours(
        thresholded_image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

    return contours


def draw_contours(sorted_contours, dimension):

    canvas = np.zeros(dimension, np.uint8)
    # show_image(canvas, "Empty canvas")

    cv.drawContours(canvas, contours=sorted_contours, contourIdx=-1,
                    color=(255, 255, 255), thickness=1, lineType=cv.LINE_AA)
    # show_image(canvas, "Canvas")

    return canvas


def detect_corners_from_contours(canvas, sorted_contours):
    approximate_curve_coordinates = []
    for contour in sorted_contours:
        contour_perimeter = cv.arcLength(contour, closed=True)

        approximation_accuracy = 0.01*contour_perimeter
        approximate_curve_coordinates.append(cv.approxPolyDP(
            contour, epsilon=approximation_accuracy, closed=True))
    
    # To inspect the approximate curve
    cv.drawContours(canvas, contours=approximate_curve_coordinates, contourIdx=-1,
                    color=(255, 255, 255), thickness=2, lineType=1)

    # show_image(canvas, "Canvas corner")

    return approximate_curve_coordinates


def return_four_corners(image, approximate_curve_coordinates):
    image = image.copy()

    cv.drawContours(image, contours=approximate_curve_coordinates, contourIdx=-1,
                    color=(255, 255, 255), thickness=2, lineType=1)

    # Convert nested numpy list to python list
    approx_corners = np.concatenate(
        approximate_curve_coordinates).tolist()
    for index, corner in enumerate(approx_corners):

        x, y = corner[0]

        cv.putText(img=image, text=str(index), org=(x, y), fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
                   fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv.LINE_AA)
        # print(index, x, y)

    show_image(image, "Canvas")

    # By inspection
    indices_list = [3, 4, 5, 10]
    four_corners = [approx_corners[i] for i in indices_list]

    # print(four_corners)

    return four_corners


def get_destination_corners(four_corners):

    # Flatten nested list
    four_corners = sum(four_corners, [])
    four_corners = sum(four_corners, [])
    
    x1, y1 = four_corners[:2]
    x2, y2 = four_corners[2:4]
    x3, y3 = four_corners[4:6]
    x4, y4 = four_corners[6:8]

    # Compute the width of the new image, which will be the maximum distance between x-coordinates
    w1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    w2 = np.sqrt((x3-x1)**2 + (y3-y1)**2)

    max_width = max(int(w1), int(w2))

    # Compute the height of the new image, which will be the maximum distance between y-coordinates
    h1 = np.sqrt((x4-x1)**2 + (y4-y1)**2)
    h2 = np.sqrt((x3-x2)**2 + (y3-y2)**2)

    max_height = max(int(h1), int(h2))

    # destination_corners = np.array([[0, 0], [
    #     max_width-1, 0], [max_width-1, max_height-1], [0, max_height-1]], dtype="float32")
    destination_corners = np.array([[0, max_height-1],[max_width-1, max_height-1], [max_width-1, 0], [0, 0]], dtype="float32")
    return destination_corners, max_width, max_height


def unwarp_image(image, four_corners, destination_corners):
    height, width = get_image_size(image)

    # print(four_corners)
    point1 = np.float32(sum(four_corners, []))
    point2 = np.float32(destination_corners)
    
    H, _ = cv.findHomography(
        point1, point2, method=cv.RANSAC, ransacReprojThreshold=1.0)
    unwarpped_image = cv.warpPerspective(
        image, H, (width, height), flags=cv.INTER_LINEAR)
    # show_image(unwarpped_image, "Final Image")
    return unwarpped_image


def crop_image(image, max_width, max_height):

    cropped_image = image[0:max_height, 0:max_width]
    cv.imwrite("output/output1.jpg", cropped_image)
    show_image(cropped_image, "Cropped_image")



def main():
    image = cv.imread('images/cheque1.jpg')
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # show_image(gray_image, "Gray Image")

    # Apply 2D filter to nicely blur texts
    filtered_2D_image = apply_filter(gray_image)

    thresholded_image = apply_threshold(filtered_2D_image)

    # Median filter clears small details
    # blurred_image = cv.medianBlur(thresholded_image, 5)
    # show_image(blurred_image, "Blur_image")

    # Add black border
    # bordered_image = cv.copyMakeBorder(
    #     blurred_image, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=[0, 0, 0])
    # # show_image(bordered_image, "Bodered Image")

    contours = find_contours(thresholded_image)

    sorted_contours, _ = sort_contours(contours)
    
    dimension = get_image_size(image)

    contours_canvas = draw_contours(sorted_contours, dimension)
    # show_image(contours_canvas, "Canvas")


    # Detect corners from contours
    approximate_curve_coordinates = detect_corners_from_contours(
        contours_canvas, sorted_contours)
    
    # Draw corners on image and by inspection returns four points
    four_corners = return_four_corners(
        image, approximate_curve_coordinates)
    # print(four_corners)
    
    destination_corners, max_width, max_height = get_destination_corners(
        four_corners)
    unwarpped_image = unwarp_image(image, four_corners, destination_corners)
    crop_image(unwarpped_image, max_width, max_height)


if __name__ == "__main__":
    main()
