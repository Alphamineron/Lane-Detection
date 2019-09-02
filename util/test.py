import numpy as np
import cv2 as cv


def get_thresholded_image(img):
    # convert to gray scale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) # NOTE: RGB Image Required!

    height, width = gray.shape

    # apply gradient threshold on the horizontal gradient
    sx_binary = abs_sobel_thresh(gray, 'x', 10, 200)

    # apply gradient direction threshold so that only edges closer to vertical are detected.
    dir_binary = dir_threshold(gray, thresh=(np.pi/6, np.pi/2))

    # combine the gradient and direction thresholds.
    combined_condition = ((sx_binary == 1) & (dir_binary == 1))

    # R & G thresholds so that yellow lanes are detected well.
    color_threshold = 150
    R = img[:,:,0]
    G = img[:,:,1]
    color_combined = np.zeros_like(R)
    r_g_condition = (R > color_threshold) & (G > color_threshold)


    # color channel thresholds
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    S = hls[:,:,2]
    L = hls[:,:,1]

    # S channel performs well for detecting bright yellow and white lanes
    s_thresh = (100, 255)
    s_condition = (S > s_thresh[0]) & (S <= s_thresh[1])

    # We put a threshold on the L channel to avoid pixels which have shadows and as a result darker.
    l_thresh = (120, 255)
    l_condition = (L > l_thresh[0]) & (L <= l_thresh[1])

    # combine all the thresholds
    # A pixel should either be a yellowish or whiteish
    # And it should also have a gradient, as per our thresholds
    color_combined[(r_g_condition & l_condition) & (s_condition | combined_condition)] = 1

    # apply the region of interest mask
    mask = np.zeros_like(color_combined)
    region_of_interest_vertices = np.array([[0,height-1], [width/2, int(0.5*height)], [width-1, height-1]], dtype=np.int32)
    cv.fillPoly(mask, [region_of_interest_vertices], 1)

    # Generate thresholded mask having values in Boolean
    thresholded = cv.bitwise_and(color_combined, mask)

    return cv.inRange(thresholded, 1, 255)   # Boolean ==> CV

def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255):
    if orient == 'x':
        sobel = cv.Sobel(gray, cv.CV_64F, 1, 0)
    else:
        sobel = cv.Sobel(gray, cv.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    max_value = np.max(abs_sobel)
    binary_output = np.uint8(255*abs_sobel/max_value)
    threshold_mask = np.zeros_like(binary_output)
    threshold_mask[(binary_output >= thresh_min) & (binary_output <= thresh_max)] = 1
    return threshold_mask

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Take the gradient in x and y separately
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobel_y,abs_sobel_x)
    direction = np.absolute(direction)
    # 5) Create a binary mask where direction thresholds are met
    mask = np.zeros_like(direction)
    mask[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return mask




def edge_detect(frame, minthres=50, maxthres=150):
    # GrayScaling the Image
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred_frame = cv.GaussianBlur(gray_frame, (5, 5), 0)
    return cv.Canny(blurred_frame, minthres, maxthres)

# def lanes_mask(image):


# Reading the Source
frame = cv.imread("/Users/AI-Mac1/Developer/Python/Tutorials/OpenCV/data/road.jpeg", cv.IMREAD_COLOR)
# Converting into HSV Space since color distinction is much more intuitive here
# This makes selecting a specific color range for our purposes easier
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
# mask = lanes_mask(hsv)
y_mask  = cv.inRange(hsv, np.array([0, 0, 150]), np.array([70, 255, 255]))
res = cv.bitwise_and(frame, frame, mask=y_mask)
res = cv.GaussianBlur(res, (5, 5), 0)


# cv.Canny(res, 50, 150)
canny_frame = edge_detect(frame)

cv.imshow("image", canny_frame)
cv.waitKey(0)






# Traces a Half-Compressed Hexagonal ROI
# ROI_vertices = np.array([[0, height],
#                          [0, height*(2/3)],
#                          [width/4, height/3],
#                          [width*0.75, height/3],
#                          [width, height*(2/3)],
#                          [width, height]])
# global roi_height
# roi_height = height*0.6
# ROI_vertices = np.array([[0, height],
#                          [0, height*0.8],
#                          [width/4, roi_height],
#                          [width*0.75, roi_height],
#                          [width, height*0.8],
#                          [width, height]])
