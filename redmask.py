import cv2
from matplotlib import pyplot as plt

img = cv2.imread("image.jpg")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # image in hsv, is better for masking.

red_lower = (0, 180, 200)   # Definitions for red
red_upper = (24, 255, 255)  

red_mask = cv2.inRange(img_hsv, red_lower, red_upper) # apply mask to hsv
result = cv2.bitwise_and(img_rgb, img_rgb, mask=red_mask) # apply mask to rgb

plt.imshow(result)
plt.show()