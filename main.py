import cv2 as cv
import sys

from src.preprocess import (apply_adaptive_threshold,
                            apply_canny_edge_detection, apply_dilation,
                            remove_horizontal_line, remove_vertical_line, resize_image,
                            show_image)
from src.sort_contours import sort_contours


def find_contours(image):
    # Contours coordinates
    contours, _ = cv.findContours(
        image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    # print(contours)
    return contours


def draw_rect(image, contours):
    image = image.copy()
    
    for index, box in enumerate(contours):
        x, y, w, h = cv.boundingRect(box)
        if 10 < h < 80  and w > 30:
            cv.rectangle(image, (x, y), (x+w, y+h), (36, 255, 12), 2)
            # cv.putText(image, str(index+1), (x+w//4, y+h//4),
            #            cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,
            #            (0, 0, 0), 1, cv.LINE_AA)

    cv.imwrite("output/output2.jpg", image)
    show_image(image, "Bounded image")


def main():
    INPUT_IMAGE_PATH = sys.argv[1]
    image = cv.imread(INPUT_IMAGE_PATH)
    # show_image(image, "input_image")

    # resized_image = resize_image(image, 1)
    # show_image(resized_image, "resized_image")

    canny_edge_detected_image = apply_canny_edge_detection(image)
    # show_image(canny_edge_detected_image, "Canny edge")

    # Remove horizontal and vertical edges
    horizontal_line_removed_image = remove_horizontal_line(
        canny_edge_detected_image)
    vertical_line_removed_image = remove_vertical_line(
        horizontal_line_removed_image)
    # show_image(image, "image")

    thresholded_image = apply_adaptive_threshold(vertical_line_removed_image)

    dilated_image = apply_dilation(thresholded_image)

    contours = find_contours(dilated_image)
    sorted_contours, _ = sort_contours(contours)

    draw_rect(image, sorted_contours)


if __name__ == "__main__":
    main()
