import cv2 as cv


def sort_contours(contours):
    ''' 
    Sorts contours from left -> right, top -> bottom.

    '''
    bounding_boxes = [cv.boundingRect(c) for c in contours]

    # Sorting on x-axis
    sorted_by_x = zip(*sorted(zip(contours, bounding_boxes),
                              key=lambda b: b[1][0], reverse=False))

    # Sorting on y-axis
    (contours, bounding_boxes) = zip(*sorted(zip(*sorted_by_x),
                                             key=lambda b: b[1][1],
                                             reverse=False))
    # Returns the list of sorted contours and bounding boxes
    return (contours, bounding_boxes)
