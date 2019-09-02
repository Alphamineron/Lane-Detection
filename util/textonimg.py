import numpy as np
import cv2

# Create a black image
height, width = (512,512)
img = np.zeros((height,width,3), np.uint8)

# Write some Text

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20,height-20)
fontScale              = 0.75
fontColor              = (255,255,255)
lineType               = 2

cv2.putText(img,'Hello World!',
    bottomLeftCornerOfText,
    font,
    fontScale,
    fontColor,
    lineType)

#Display the image
cv2.imshow("img",img)

# # Save image
# cv2.imwrite("out.jpg", img)

cv2.waitKey(0)
