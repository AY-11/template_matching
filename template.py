

import cv2
import numpy as np

# Load the images
image = cv2.imread('original1.png')
template = cv2.imread('templat.png')

# Convert to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Perform template matching
result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# Get the best match position
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Define the rectangle around the matched region
h, w = template_gray.shape
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# Draw the rectangle
cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# Show the result
cv2.imshow('Matched Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()