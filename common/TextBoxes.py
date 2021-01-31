import numpy as np
import matplotlib.path as mplPath

START_X = 0
START_Y = 1
END_X = 2
END_Y = 3

class TextBoxes:
    def __init__(self, boxes, min_dist_allowed):
        self.__boxes = boxes
        self.__boxes_hierarchy = {}
        self.__min_dist_allowed = min_dist_allowed
        self.__final_boxes = []
        self.__build_boxes_hierarchy()

    def __check_overlapping_between_two_boxes(self, box, neighbor):   # 'box' and 'neighbor' have this form:
        # (top_left, top_right, bottom_left, bottom_right)
        overlapping = False

        ## Neighbor points #################################################
        neighbor_top_left = (neighbor[0][0], neighbor[0][1])
        neighbor_bottom_left = (neighbor[2][0], neighbor[2][1])
        neighbor_middle_left = [neighbor[2][0], neighbor[0][1] + int(neighbor[2][1] - neighbor[0][1])//2]
        ####################################################################

        ## Main box vertices ###############################################
        box_top_left = [box[0][0], box[0][1]]
        box_top_right = [box[1][0], box[1][1]]
        box_bottom_left = [box[2][0], box[2][1]]
        box_bottom_right = [box[3][0], box[3][1]]
        ####################################################################

        ## Build bounding box ##############################################
        boundingBox = mplPath.Path(np.array([box_top_left, box_top_right,box_bottom_left, box_bottom_right]))
        ####################################################################

        if boundingBox.contains_point(neighbor_top_left) or boundingBox.contains_point(neighbor_bottom_left) or boundingBox.contains_point(neighbor_middle_left):
            overlapping = True
        return overlapping

    def __euclidean_distance_between_two_boxes(self, box, neighbor):
        # Working with the box1 right-bottom vertex and the box2 left-bottom vertex

        ## Neighbor bottom left vertex ######################################
        neighbor_point = np.array([neighbor[2][0], neighbor[2][1]])
        ####################################################################

        ## Main box bottom right vertex ####################################
        main_box_point = np.array([box[3][0], box[3][1]])
        ####################################################################

        ## Calculate Euclidean distance ####################################
        dist = (main_box_point - neighbor_point)**2
        dist = np.sum(dist)
        return np.sqrt(dist)
        ####################################################################

    def __build_boxes_hierarchy(self):
        box_index = 0
        for box in self.__boxes: # Add each box to the hierarchy based on its coords
            # Compute box vertices
            top_left = (box[START_X], box[START_Y])     # (startX, startY)
            top_right = (box[END_X], box[START_Y])      # (endX, startY)
            bottom_left = (box[START_X], box[END_Y])    # (startX, endY)
            bottom_right = (box[END_X], box[END_Y])     # (endX, endY)

            # Keep the original (startX, startY, endX, endY) coords and save the computed vertices
            self.__boxes_hierarchy['box' + str(box_index)] = {}
            self.__boxes_hierarchy['box' + str(box_index)]['original'] = (box[START_X], box[START_Y], box[END_X], box[END_Y])
            self.__boxes_hierarchy['box' + str(box_index)]['vertices'] = (top_left, top_right, bottom_left, bottom_right)
            box_index += 1

        for box_id in self.__boxes_hierarchy:   # Check the box-neighbors relation for merging later
            self.__boxes_hierarchy[box_id]['merge'] = []
            for neighbor_box_id in self.__boxes_hierarchy:
                if neighbor_box_id != box_id:
                    current_box = self.__boxes_hierarchy[box_id]['vertices']
                    neighbor_box = self.__boxes_hierarchy[neighbor_box_id]['vertices']

                    # Check overlapping
                    if self.__check_overlapping_between_two_boxes(current_box, neighbor_box):
                        self.__boxes_hierarchy[box_id]['merge'].append(neighbor_box_id)

                    # Check distance between boxes
                    elif self.__euclidean_distance_between_two_boxes(current_box, neighbor_box) < self.__min_dist_allowed:
                        self.__boxes_hierarchy[box_id]['merge'].append(neighbor_box_id)

        all_boxes = [] # This will store the boxes where each item is a unique box or a set of boxes with a relation
        boxes_ids = list(self.__boxes_hierarchy.keys())
        current_box = [boxes_ids[0]]
        prev_box = current_box.copy()
        for box_id in self.__boxes_hierarchy: # Build the new boxes based on the computed hierarchy
            if box_id not in prev_box: # Check if the current box has any relation with the previous ones
                all_boxes.append(current_box)   # If not, restart the current box
                current_box = [box_id]          # and put the current box-id as the first member of the new box-set

            for neighbor_box_id in self.__boxes_hierarchy[box_id]['merge']: # Check the neighbors with a relation with the
                # current box, and add them to the current box-set
                current_box.append(neighbor_box_id)

            prev_box = current_box.copy()
        all_boxes.append(current_box) # Add the last box-set computed to the 'all_boxes' list

        for box in all_boxes:   # This will process the boxes, the unique ones and the sets
            if len(box) > 1:    # If it is a box-set, compute the bounding rectangle for it
                boxes_coords = []
                for box_id in box:
                    boxes_coords.append(self.__boxes_hierarchy[box_id]['vertices'])

                boxes_coords = np.array(boxes_coords)
                # Compute the bounding rectangle as the original (startX, startY, endX, endY) coords
                startX = np.min(boxes_coords[:, 0, 0])
                startY = np.min(boxes_coords[:, 0, 1])
                endX = np.max(boxes_coords[:, 3, 0])
                endY = np.max(boxes_coords[:, 3, 1])

                # Add the bounding rectangle as a new box
                self.__final_boxes.append((startX, startY, endX, endY))
            else: # If not, just get back the original (startX, startY, endX, endY) coords
                self.__final_boxes.append(self.__boxes_hierarchy[box[0]]['original'])

    def get_final_boxes(self):
        return self.__final_boxes