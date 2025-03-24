# %%
import cv2  # Importing OpenCV library
import sys  # Importing sys module for system-specific parameters and functions
import numpy as np  # Importing numpy library for numerical computations
import matplotlib.pyplot as plt  # Importing pyplot module from matplotlib for plotting

# %%
import matplotlib

matplotlib.use('TkAgg')  # had problems with plt.show() in print_image()


# %%
# Custom function for showing the image because(cv2.imshow() is not working on MacOS)
def print_image(image, title=""):
    """
   Print image on matplotlib canvas

   :param np.ndarray image: Array representing the image
   """
    # Convert BGR to RGB for proper visualization
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = np.array(image)
    plt.imshow(pixels)
    plt.title(title)
    plt.show()


# %%
# Read the image
img = cv2.imread("red_ball.jpg")
if img is None:
    sys.exit("Could not read the image.")  # Exit if image not found
print_image(img)

# %%
# Convert the image to HSV color space
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img_hsv =
# h, s, v =
# hsv_img =
print_image(hsv_img, "hsv_img no converts")

# %%
# Define lower and upper bounds for Red color
# Lower mask (0-10)
sat = 50
val = 45

lower_red = np.array([0, sat, val])
upper_red = np.array([10, 255, 255])
# Find the colors within the boundaries
mask0 = cv2.inRange(hsv_img, lower_red, upper_red)

# Upper mask (170-180)
lower_red = np.array([170, sat, val])
upper_red = np.array([180, 255, 255])
# Find the colors within the boundaries
mask1 = cv2.inRange(hsv_img, lower_red, upper_red)

# Join masks
mask = mask0 + mask1
# Display the mask
print_image(mask, "mask")

# %%
# Remove unnecessary noise from the mask
# Define kernel size
kernel = np.ones((7, 7), np.uint8)  # Create a 7x7 8-bit integer matrix
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove unnecessary black noises from the white region
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Remove white noise from the black region of the mask
print_image(mask, "mask after removing noises")

# %%
# Segment only the detected region
segmented_img = cv2.bitwise_and(img, img, mask=mask)
print_image(segmented_img, "segmented_img")

# %%
# Convert the segmented image to grayscale
gray_image = segmented_img[:, :, 2]
print_image(gray_image)

# %%
# Convert the grayscale image to binary image
ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
print_image(thresh, "thresh")

# %%
# Calculate moments of the binary image
M = cv2.moments(thresh)
print(M)

# %%
# Calculate x, y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# %%
# Put text and highlight the center
cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
cv2.putText(img, "red ball", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
print_image(img)

# %%
# Find contours from the mask
contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Draw contour on image
output = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
print_image(output, "output")
