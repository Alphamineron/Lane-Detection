# --------------------------------------------------------------------------
#
#                              <File Information>
#                             Color Space Selection
#      This program is meant to aid in choosing the best color space to
#       be used for lane detecion given YELLOW & WHITE markings on the
#                               road of interest.
#
#           Dev Notes:  Pass the images through the entire detection pipeline
#
#           Update Log:
#
#     Alphamineron --  Created: 13/12/18        Updated:
#
#
# --------------------------------------------------------------------------
import cv2 as cv

# Reading the Soucre
frame = cv.imread("data/test_image.jpg", cv.IMREAD_COLOR)

# Making Multiple Color Scheme Versions to Compare
images = []
images.append((frame, "BGR"))
images.append((cv.cvtColor(frame, cv.COLOR_BGR2GRAY), "GREY"))
images.append((cv.cvtColor(frame, cv.COLOR_BGR2Lab), "LAB"))
images.append((cv.cvtColor(frame, cv.COLOR_BGR2HLS), "HLS"))
images.append((cv.cvtColor(frame, cv.COLOR_BGR2HSV), "HSV"))
images.append((cv.cvtColor(frame, cv.COLOR_BGR2LUV), "LUV"))
images.append((cv.cvtColor(frame, cv.COLOR_BGR2XYZ), "XYZ"))
images.append((cv.cvtColor(frame, cv.COLOR_BGR2YCrCb), "YCrCb"))
images.append((cv.cvtColor(frame, cv.COLOR_BGR2YUV), "YUV"))

# Display the Images List with appropriate window names
i = 0
while(True):
    cv.imshow(images[i][1], images[i][0])
    k = cv.waitKey(0)
    cv.destroyAllWindows()
    if(k == 2):    # Left Arrow Key
        if(i == 0): i=len(images)-1
        else: i-=1
    if(k == 3):    # Right Arrow Key
        if(i == len(images)-1): i=0
        else: i+=1
    if(k == 27):    # Esc Key
        break
