import cv2
import numpy as np

def load_image(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    return img

def show_image(img):
    if type(img) == str:
        img = cv2.imread(img, cv2.IMREAD_COLOR)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_image(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def resize_image_by_dim(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def get_offset_for_all_directions(scale_x, scale_y, width, height):
    offset_x = int(width * scale_x) // 2
    offset_y = int(height * scale_y) // 2
    return offset_x, offset_y

def get_boxes_as_images(boxes, img): # 'boxes' item:(start_x, start_y, end_x, end_y)
    detected_texts = []

    if len(boxes) > 1:
        # Sort 'boxes' by x-axis
        boxes = np.array(boxes)
        boxes = boxes[boxes[:, 0].argsort()]

    for box in boxes:   # box:(start_x, start_y, end_x, end_y)
        start_x = box[0]
        start_y = box[1]
        end_x = box[2]
        end_y = box[3]

        detected_texts.append(img[start_y:end_y, start_x:end_x, :])

    return np.array(detected_texts)

def write_message_on_img(img, message, position):
    RED = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 2
    font_color = RED
    font_thickness = 3

    img_text = cv2.putText(img, message, position, font, font_size, font_color, font_thickness, cv2.LINE_AA)

    return img_text

def clean_factor_from_expression(factor):
    if len(factor) > 1 and factor[0] == '0':
        return factor[1:]
    else:
        return factor

def add_square_canvas_to_image(img):
    h, w = img.shape[:2]

    if w > h:
        canvas = np.zeros((w, w))
        y_axis_start = (w-h)//2
        canvas[y_axis_start:y_axis_start + h, :w] = img
    else:
        canvas = np.zeros((h, h))
        x_axis_start = (h-w)//2
        canvas[:h, x_axis_start:x_axis_start + w] = img

    return canvas

def add_borders_to_image(img, border_scale, color=(255, 255, 255)):
    top = int(border_scale * img.shape[0])  # shape[0] = rows
    bottom = top
    left = int(border_scale * img.shape[1])  # shape[1] = cols
    right = left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, color)
