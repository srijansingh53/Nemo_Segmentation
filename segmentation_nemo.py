"""
Created on Fri Feb 15 00:05:27 2019

@author: sri
"""



import cv2
import numpy as np
import matplotlib.pyplot as plt
""" # To know available conversions conversions

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

for i in range(len(flags)):
    print(i,flags[i])
"""    
img = cv2.imread('./images/nemo2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.resize(img, (0,0), fx=1, fy=1)
img_hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
""" # displaying in a separate frame
cv2.imshow('Image', img_hsv)
cv2.waitKey(0) """

# masking on the basis of 
light_orange = (1, 150, 150)
dark_orange = (18, 255, 255)

mask_orange = cv2.inRange(img_hsv, light_orange, dark_orange)
# imposing the mask onto the original image 
result_orange = cv2.bitwise_and(img, img, mask=mask_orange)
"""# to display in cv2 frames
result1 = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.subplot(1, 2, 1)
cv2.imshow('Mask', mask)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
cv2.imshow('Imposed', result1)
"""
plt.subplot(1, 2, 1)
plt.imshow(mask_orange, cmap="gray")
plt.subplot(1, 2, 2)

plt.imshow(result_orange)
plt.show()
cv2.waitKey(0)


# Adding a second mask for white part in nemo 
light_white = (0, 0, 200)
dark_white = (145, 60, 255)

mask_white = cv2.inRange(img_hsv, light_white, dark_white)
result_white = cv2.bitwise_and(img, img, mask=mask_white)

plt.subplot(1, 2, 1)
plt.imshow(mask_white, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result_white)
plt.show()

# Adding the two masks to complete the segmented image

final_mask = mask_orange + mask_white

final_result = cv2.bitwise_and(img, img, mask=final_mask)
plt.subplot(1, 2, 1)
plt.imshow(final_mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(final_result)
plt.show() 

blur = cv2.GaussianBlur(final_result, (7, 7), 0)
plt.imshow(blur)
plt.show()







