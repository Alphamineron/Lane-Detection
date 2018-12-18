# --------------------------------------------------------------------------
#
#                              <File Information>
#                 Linear Lane Detection Algorithm using OpenCV
#      Draws a calculated linear line representing the lane in which a car is
#       driving, assuming a footage from the front dashboard of the car and
#           optimal video quality. No use of Neural Networks, uses basic
#           algorithmic machine learning like regression to plot out the lane.
#       System works in three basic steps that can be intuitively understood from
#             the pipeline:  Image Pre-Processing
#                            Edge detection
#                            Lines Recognition using Hough Probalistic Transform
#                            Lines Filtering, Processing into single Lane Lines
#                            Lane Stablisation
#                            Drawing
#
#
#
#
#           Dev Notes: - Port over to ROS
#                      - Improve Performance on the harder_challenge_video
#                      - Refactor all functions, write docstrings
#                      - Work on Non-Linear Lane Detection Program
#
#          Update Log: - Minor Refactoring
#                      - Added Buffer Mechanism to improve stability
#                      - Added Polygonal Detection area indicating
#                        detection stability
#                      - Added Lane Filling Mechanism using Buffer to
#                        compensate when the detection fails using check_BUF()
#                      - Added Dashboard feature in subsitution to multiple
#                        output windows which were used before
#
#
#     Alphamineron --  Created: 13/12/18        Updated: 18/12/18
#
#
# --------------------------------------------------------------------------

from copy import deepcopy
import numpy as np
import cv2 as cv

height, width = (600,600)  # Dimension of the image, crucial for coord sys
                           # functioning. These values are arbitary, it'll
                           # be updated locally within the image pipeline

class Buffer(object):
    N_BUFFER_FRAMES = 5

    left = None
    right = None

    def __init__(self, line):
        self.stack = np.array(line)  # Line = [x1, y2, x2, y2]
        self.current_frames = 0

    def push(self, line):
        if self.current_frames < Buffer.N_BUFFER_FRAMES:
            self.stack = np.vstack((line, self.stack))
            self.current_frames+=1
        else:
            self.stack = np.delete(self.stack, 4, axis = 0)
            self.stack = np.vstack((line, self.stack))



# ██    ██ ████████ ██ ██      ██ ████████ ██    ██
# ██    ██    ██    ██ ██      ██    ██     ██  ██
# ██    ██    ██    ██ ██      ██    ██      ████
# ██    ██    ██    ██ ██      ██    ██       ██
#  ██████     ██    ██ ███████ ██    ██       ██

COLORS = {   # Colors to be Used
        "lane_color" : (244, 65, 205),
        "region_stable" : (60, 120, 0),
        "region_unstable" : (60, 80, 255),
        "left_line" : (220, 40, 60),
        "right_line" : (255, 0, 255),
}

def gimp_to_opencv_hsv(*hsv): # Simple GIMP => CV2 HSV format converter.
    return (hsv[0]/2, hsv[1]/100*255, hsv[2]/100*255)

# White and yellow color thresholds for lines masking.
# Optional "kernel" key is used for additional morphology
WHITE_LINES = { 'low_th': gimp_to_opencv_hsv(0, 0, 80),
                'high_th': gimp_to_opencv_hsv(359, 10, 100) }

YELLOW_LINES = { 'low_th': gimp_to_opencv_hsv(35, 20, 30),
                 'high_th': gimp_to_opencv_hsv(65, 100, 100),
                 'kernel': np.ones((3,3),np.uint64)}

def get_lane_lines_mask(hsv_image, colors):
    """
     Image binarization using a list of colors. The result is a binary mask
     which is a sum of binary masks for each color.
    """
    masks = []
    for color in colors:
        if 'low_th' in color and 'high_th' in color:
            mask = cv.inRange(hsv_image, color['low_th'], color['high_th'])
            if 'kernel' in color:
                mask = cv.morphologyEx(mask, cv.MORPH_OPEN, color['kernel'])

            masks.append(mask)
        else: raise Exception('High or low threshold values missing')

    if masks:                  # Since cv.add() doesn't take lists of images. It takes images.
        return cv.add(*masks)  # '*masks' opens the list and passes it as arguments

def ROI(img):
    ROI_vertices = np.array([[0,height-1],
                             [width/2, int(0.5*height)],
                             [width-1, height-1]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv.fillPoly(mask, [ROI_vertices], (255, 255, 255))
    masked = cv.bitwise_and(img, mask)
    return masked

def make_coords(l_parameters):  # Create coords from given line parameters
    """
     Generate end points coordinates using given slope & intercept
     parameters for a line.
     Args: l_parameters - numpy array of [slope, intercept]      # TODO: Check if "numpy" array is req or list is enough

     Result: Returns a numpy array(4,) representing a line as a list
             of [x1, y1, x2, y2].
    """
    slope, intercept = l_parameters
    y1 = int(height)
    y2 = int(y1*(0.70))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def clean_lane_lines(img, lines):
    """
     Process Hough Lines and generate final lane Prediction.

     |==> Classify Lines onto one side
     |==> Filter out Noise Lines
     |==> Merge Filtered Lines into their corresponding Lanes

     Result: Returns a numpy array(2, 4) of the two filtered lane lines,
             representing each line as a list of [x1, y1, x2, y2]
    """
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if(abs(slope) > 1.73 or abs(slope) < 0.46): #Basic Filtering
                continue

            if slope > 0:
                right_fit.append((slope, intercept))
            else:
                left_fit.append((slope, intercept))


        if left_fit != [] and right_fit != []:
            left_fit_avg = np.average(left_fit, axis=0)
            left_lines = make_coords(left_fit_avg)
            right_fit_avg = np.average(right_fit, axis=0)
            right_lines = make_coords(right_fit_avg)
            return np.array([left_lines, right_lines]), np.array([1, 1])

        elif left_fit != []:
            left_fit_avg = np.average(left_fit, axis=0)
            left_lines = make_coords(left_fit_avg)
            return np.array([left_lines, np.array([0, 0, 0, 0])]), np.array([1, 0])

        elif right_fit != []:
            right_fit_avg = np.average(right_fit, axis=0)
            right_lines = make_coords(right_fit_avg)
            return np.array([np.array([0, 0, 0, 0]), right_lines]), np.array([0, 1])

        else:
            return np.array([np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])]), np.array([0, 0])
    else:
        print("No Lines Detected!")
        return None, np.array([0, 0])

def check_BUF(lines):
    """
     Manages the Buffer Operations such as input and output.

     Parameters
     ----------
        :lines: A nparray of shape (2,4). It contains the 4 coords
                describing a line, in this case there are 2 lines.
    """
    if lines is not None:
        if Buffer.left is None and Buffer.right is None:
            Buffer.left = Buffer(lines[0])
            Buffer.right = Buffer(lines[1])

        if np.array_equal(lines[0], np.array([0, 0, 0, 0])):
            lines[0] = np.average(Buffer.left.stack, axis=0)

        if np.array_equal(lines[1], np.array([0, 0, 0, 0])):
            lines[1] = np.average(Buffer.right.stack, axis=0)

        Buffer.left.push(lines[0])
        Buffer.right.push(lines[1])
    else:
        if Buffer.left is not None:
            lines = np.array([np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])])
            lines[0] = np.average(Buffer.left.stack, axis=0)

        if Buffer.right is not None:
            if lines is None:     # Which would happen if the Buffer.left was None
                lines = np.array([np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])])
            lines[1] = np.average(Buffer.right.stack, axis=0)

    return lines

def check_slope_condition(s1, s2):
    CRITICAL_SLOPE_CHANGE = 0.1  # (About 6 degrees)
    if abs(s1 - s2) > CRITICAL_SLOPE_CHANGE:
        return False
    return True

def cal_slope(line):
    x1, y1, x2, y2 = line
    if x2-x1 == 0:
        return 2147483647
    return ((y2-y1)/(x2-x1))

def stablise_lines(lines):
    DECISION_MAT = [[.1, .9], [1, 0]]
    stable_lines = []
    stablestat = np.array([], dtype="int64")
    if lines is not None:
        left_bufline = np.average(Buffer.left.stack, axis=0)
        right_bufline = np.average(Buffer.right.stack, axis=0)
        buflines = [left_bufline, right_bufline]

        for bufline, laneline in zip(buflines, lines):
            bufline_slope = cal_slope(bufline)
            lane_slope = cal_slope(laneline)

            stability = check_slope_condition(bufline_slope, lane_slope)
            stablestat = np.append(stablestat, int(stability))  # Storing Stability

            weights = DECISION_MAT[stability]
            # 0.1 * detected + 0.9 * avg from buffer  : In case of unstable lane.
            # 1 * detected + 0 * avg from buffer      : In case of stable lane.
            stable_lines.append(np.dot(weights, np.vstack([laneline, bufline])))
        lines = np.array([stable_lines[0], stable_lines[1]])
    else:
        stablestat = np.array([0, 0])

    return lines, stablestat

def draw_lanes(img, lines):
    lanes = np.zeros_like(img)  # "img" must have the same color space as original
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            x1, y1, x2, y2 = map(int,  [x1, y1, x2, y2])  # cv.line requires int inputs
            cv.line(lanes, (x1, y1), (x2, y2), COLORS["lane_color"], 5)
    return lanes

def draw_lane_polygon(img, lines, stability):
    offset_from_lane_edge = 0
    color = COLORS["region_stable"]
    poly_img = np.zeros_like(img)

    if lines is not None:
        p1 = [lines[0][0], lines[0][1]]
        p2 = [lines[0][2] + offset_from_lane_edge, lines[0][3] + offset_from_lane_edge]

        p3 = [lines[1][2] + offset_from_lane_edge, lines[1][3] + offset_from_lane_edge]
        p4 = [lines[1][0], lines[1][1]]

        polygon_points = np.array([p1, p2, p3, p4], np.int32).reshape((-1,1,2))

        if not stability[0] or not stability[1]:
            color = COLORS["region_unstable"]

        cv.fillPoly(poly_img, [polygon_points], color)

    return poly_img

def draw_dashboard(img, snapshot1, snapshot2, snapshot3):
    # 160 width for each window
    cv.CV_FILLED = -1
    image_copy = deepcopy(img)
    cv.rectangle(image_copy, (0,0), (540+260,175), (0,0,0), cv.CV_FILLED)
    img = cv.addWeighted(image_copy, 0.3, img, 0.7, 0)

    if(len(snapshot1.shape) != 3):
        snapshot1 = cv.cvtColor(snapshot1, cv.COLOR_GRAY2BGR)
    if(len(snapshot2.shape) != 3):
        snapshot2 = cv.cvtColor(snapshot2, cv.COLOR_GRAY2BGR)
    if(len(snapshot3.shape) != 3):
        snapshot3 = cv.cvtColor(snapshot3, cv.COLOR_GRAY2BGR)

    img[20:155,20:260,:] = snapshot1
    img[20:155,280:520,:] = snapshot2
    img[20:155,540:780,:] = snapshot3
    return img

# ██████  ██ ██████  ███████ ██      ██ ███    ██ ███████
# ██   ██ ██ ██   ██ ██      ██      ██ ████   ██ ██
# ██████  ██ ██████  █████   ██      ██ ██ ██  ██ █████
# ██      ██ ██      ██      ██      ██ ██  ██ ██ ██
# ██      ██ ██      ███████ ███████ ██ ██   ████ ███████

def image_pipeline(frame):
                                                                                                        # frame = cv.GaussianBlur(frame, (5, 5), 0)
                                                                                                        # frame = cv.bilateralFilter(frame, 9, 75, 75)
    global height, width  #To Access the Global Variable
    height, width, _ = frame.shape   #IMPORTANT FOR THE CORRECT FUNCTIONING COORDs
    hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # HSV: Inuitive Color distribution -> Easy thresholding
    thresholded = get_lane_lines_mask(hsv_image, [WHITE_LINES, YELLOW_LINES])


    #====================++++++++++++================+++++++==============
    canny = cv.Canny(thresholded, 50, 150)  # TODO: Play with this threshold
    #====================++++++++++++================+++++++==============


    roi = ROI(canny)

    lines = cv.HoughLinesP(roi, 2, np.pi/180, 50, None, minLineLength=20, maxLineGap=100)
    lane_lines, stability1 = clean_lane_lines(frame, lines)
    lane_lines = check_BUF(lane_lines)
    lane_lines, stability2 = stablise_lines(lane_lines)
    stability = cv.bitwise_and(stability1, stability2)


    lane = draw_lanes(frame, lane_lines)
    poly = draw_lane_polygon(frame, lane_lines, stability)
    lane = cv.addWeighted(lane, 1.0, poly, 0.4, 0)
    detection = cv.addWeighted(frame, 0.8, lane, 1, 0)


    snapshot1 = cv.resize(deepcopy(thresholded), (240,135))     # thresholded Image
    snapshot2 = cv.resize(deepcopy(roi), (240,135))  #Raw Hough Lines
    snapshot3 = cv.resize(deepcopy(cv.addWeighted(frame, 0.8, draw_lanes(frame, lines), 1, 0)), (240,135))  #Raw Hough Lines
    detection = draw_dashboard(detection, snapshot1, snapshot2, snapshot3)

    return detection



# Open the capture stream
vin = cv.VideoCapture("test_data/challenge_video.mp4")
if(not vin.isOpened()):
    vin.open()
while(True):
    # cameraturing the video feed frame by frame
    ret, frame = vin.read()

    if(ret):
        # Operations on the frames
        detection = image_pipeline(frame)

        # Displaying the frames
        cv.imshow("detection", detection)
        if(cv.waitKey(1) == 27):
            break

    else:
        print("Video Feed Terminated")
        break
vin.release()
cv.destroyAllWindows()











#
