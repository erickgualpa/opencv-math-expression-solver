from imutils.object_detection import non_max_suppression
from utils import *
from TextBoxes import TextBoxes
import numpy as np
import cv2

## CONSTANTS DECLARATION #####################################################
EAST_FILENAME = "./classifiers/frozen_east_text_detection.pb"

NEW_WIDTH = 320
NEW_HEIGHT = 320

FROM_PIXEL_TO_LEFT = 3
FROM_PIXEL_TO_RIGHT = 1
FROM_PIXEL_TO_UP = 0
FROM_PIXEL_TO_DOWN = 2
BOX_ANGLE = 4
##############################################################################

def detect_text_on_image(img, min_confidence):
    original = img.copy()   # Keep a copy of the original image

    # STEP 1: Resize image for making its dimensions 32 multiples
    height, width = img.shape[:2]
    img = cv2.resize(img, (NEW_WIDTH, NEW_HEIGHT))
    #show_image(img)

    rW = width / float(NEW_WIDTH)   # Width change ratio
    rH = height / float(NEW_HEIGHT) # Height change ratio

    # STEP 2: Load the pre-trained EAST text detector
    east = cv2.dnn.readNet(EAST_FILENAME)

    # STEP 3: Define the model output layers we are interested in
    layerNames = [
                    "feature_fusion/Conv_7/Sigmoid",    # - Probability of a region about containing text or not
                    "feature_fusion/concat_3"           # - Geometry to derive the bounding box coordinates of the text
                                                        # in the input image
    ]

    # STEP 4: Image pre-processing
    scale_factor = 1.0  # Keep the current scale factor
    size = (img.shape[1], img.shape[0]) # (width, height)
    mean_subtraction = (123.68, 116.78, 103.94) # Reduce illumination changes

    blob = cv2.dnn.blobFromImage(img, scale_factor, size, mean_subtraction, swapRB=True, crop=False)

    # STEP 5: Set the image into the net and perform the model
    east.setInput(blob)
    scores, geometry = east.forward(layerNames)

    # STEP 6: Loop over the results for the data extraction
    scores = scores[0, 0, :, :]
    geometry = geometry[0, :, :, :]
    n_rows, n_cols = scores.shape[:2]
    bounding_rects_list = []
    confidences_list = []

    for row in range(n_rows):
        # Scores data grouped by pixel row -> 1-item per 1-pixel
        scores_data_row = scores[row]

        # Geometrical data grouped by pixel row -> 1-item per 1-pixel
        pixel_to_up_side_row = geometry[FROM_PIXEL_TO_UP, row]          # Distances from pixels to its bounding box up side
        pixel_to_down_side_row = geometry[FROM_PIXEL_TO_DOWN, row]      # Distances from pixels to its bounding box down side
        pixel_to_left_side_row = geometry[FROM_PIXEL_TO_LEFT, row]      # Distances from pixels to its bounding box left side
        pixel_to_right_side_row = geometry[FROM_PIXEL_TO_RIGHT, row]    # Distances from pixels to its bounding box right side
        angles_row = geometry[BOX_ANGLE, row]

        for col in range(n_cols): # Loop over each pixel data
            # Check if pixel score reaches the minimum confidence
            if scores_data_row[col] < min_confidence:
                continue

            # Feature map is 4x smaller than the input image -> Compute the (row, col) coord in the original image
            offset_x = col * 4.0
            offset_y = row * 4.0

            # Get bounding box shape
            box_height = pixel_to_up_side_row[col] + pixel_to_down_side_row[col]
            box_width = pixel_to_left_side_row[col] + pixel_to_right_side_row[col]

            # Get bounding box by angle
            angle = angles_row[col]
            cos = np.cos(angle)
            sin = np.sin(angle)

            end_x = int( offset_x + (cos * pixel_to_right_side_row[col]) + (sin * pixel_to_down_side_row[col]) )
            end_y = int( offset_y - (sin * pixel_to_right_side_row[col]) + (cos * pixel_to_down_side_row[col]) )

            start_x = int(end_x - box_width)
            start_y = int(end_y - box_height)

            # Save bounding rectangle in 'bounding_rects_list'
            bounding_rects_list.append((start_x, start_y, end_x, end_y))

            # Save pixel confidence in 'confidences_list'
            confidences_list.append(scores_data_row[col])

    # Remove all superfluous detections
    boxes = non_max_suppression(np.array(bounding_rects_list), probs=confidences_list)

    # Re-scale the boxes and draw them over the original image
    offset_scale_x = 0.5  # increasing scale of x-dim of the bounding rectangle size
    offset_scale_y = 1.1  # increasing scale of y-dim of the bounding rectangle size

    original_height = np.array(original).shape[0]
    original_width = np.array(original).shape[1]

    final_boxes = []    # list containing the boxes for the original size
    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * rW)
        start_y = int(start_y * rH)
        end_x = int(end_x * rW)
        end_y = int(end_y * rH)

        box_offset_x, box_offset_y = get_offset_for_all_directions(offset_scale_x, offset_scale_y, (end_x - start_x), (end_y - start_y))

        start_x = (start_x - box_offset_x) if (start_x - box_offset_x) > 0 else 0
        start_y = (start_y - box_offset_y) if (start_y - box_offset_y) > 0 else 0
        end_x = (end_x + box_offset_x) if (end_x + box_offset_x) < original_width else original_width
        end_y = (end_y + box_offset_y) if (end_y + box_offset_y) < original_height else original_height

        final_boxes.append((start_x, start_y, end_x, end_y))

    if len(final_boxes) > 1:
        # Sort 'boxes' by x-axis
        final_boxes = np.array(final_boxes)
        final_boxes = final_boxes[final_boxes[:, 0].argsort()]
        tb = TextBoxes(final_boxes, min_dist_allowed=125)
        final_boxes = tb.get_final_boxes()

    for box in final_boxes:
        cv2.rectangle(original, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 5)

    return final_boxes, original

