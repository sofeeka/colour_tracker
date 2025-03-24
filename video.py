# %%
import cv2
import sys
import numpy as np
from matplotlib import pyplot


# %%
def create_red_mask(img_hsv):
    # Lower mask (0-10)
    sat = 150
    val = 120

    lower_red = np.array([0, sat, val])
    upper_red = np.array([10, 255, 255])
    # Find the colors within the boundaries
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # Upper mask (170-180)
    lower_red = np.array([170, sat, val])
    upper_red = np.array([180, 255, 255])
    # Find the colors within the boundaries
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # Join masks
    mask = mask0 + mask1

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# %%
def create_green_mask(img_hsv):
    sat = 100
    val = 50

    lower_green = np.array([40, sat, val])
    upper_green = np.array([75, 255, 255])
    # Find the colors within the boundaries
    mask = cv2.inRange(img_hsv, lower_green, upper_green)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# %%
def create_blue_mask(img_hsv):
    sat = 100
    val = 50

    lower_blue = np.array([100, sat, val])
    upper_blue = np.array([125, 255, 255])
    # Find the colors within the boundaries
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# %%
def create_yellow_mask(img_hsv):
    sat = 50
    val = 25

    lower_yellow = np.array([20, sat, val])
    upper_yellow = np.array([35, 255, 255])
    # Find the colors within the boundaries
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# %%
def add_centeroid(frame, mask, text):
    # Segment only the detected region
    segmented_img = cv2.bitwise_and(frame, frame, mask=mask)

    # convert image to grayscale image
    gray_image = segmented_img[:, :, 2]

    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

    M = cv2.moments(thresh)
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX = 0
        cY = 0

    # put text and highlight the center
    cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(frame, text, (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


# %%
def find_colors(frame):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_mask = create_red_mask(img_hsv)
    green_mask = create_green_mask(img_hsv)
    blue_mask = create_blue_mask(img_hsv)
    yellow_mask = create_yellow_mask(img_hsv)

    # calculate and add centroid
    frame = add_centeroid(frame, red_mask, "red ball")
    frame = add_centeroid(frame, green_mask, "green ball")
    frame = add_centeroid(frame, blue_mask, "blue ball")
    frame = add_centeroid(frame, yellow_mask, "yellow ball")

    # Find contours from the mask
    contours_red, hierarchy_red = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, hierarchy_green = cv2.findContours(green_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, hierarchy_blue = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, hierarchy_yellow = cv2.findContours(yellow_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contour on image
    output = cv2.drawContours(frame, contours_red, -1, (0, 0, 255), 3)
    output = cv2.drawContours(frame, contours_green, -1, (0, 255, 0), 3)
    output = cv2.drawContours(frame, contours_blue, -1, (255, 0, 0), 3)
    output = cv2.drawContours(frame, contours_yellow, -1, (0, 255, 255), 3)

    return output


# %%
cap = cv2.VideoCapture('rgb_ball_720.mp4')

if not cap.isOpened():
    print("Error opening video file")

while cap.isOpened():
    cv2.startWindowThread()
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    det_frame = find_colors(frame)
    cv2.imshow('frame, click q to quit', det_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
