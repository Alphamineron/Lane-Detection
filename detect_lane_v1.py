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
#     Alphamineron --  Created: 13/12/18        Updated: 20/1/19
#
#
# --------------------------------------------------------------------------
import math
from copy import deepcopy
import numpy as np
import cv2 as cv
import os
from subprocess import Popen, PIPE



height, width = (600,600)  # Dimension of the image, crucial for coord sys
                           # functioning. These values are arbitary, it'll
                           # be updated locally within the image pipeline
INHOUSEVID = "OpenCV/detect_lanes/test_data/challenge_video.mp4"
INHOUSEVIDHARD = "OpenCV/detect_lanes/test_data/harder_challenge_video.mp4"
DOWNLOADEDVID = "/Users/AI-Mac1/Downloads/challenge.mp4"
# Assign an address to the VIDEO string variable to work on that video
VIDEO = DOWNLOADEDVID



class Buffer(object):
    N_BUFFER_FRAMES = 6  # No. of History Lines excluding current detection line

    # Containers for holding buffer stacks throughout the program execution
    frames = None     # Saves Frames of video (Buffer Object)
    left = None
    right = None

    def __init__(self, line, bufsize = None):
        self.stack = np.array(line)  # Line = [x1, y2, x2, y2]

        if(bufsize == None):
            bufsize = Buffer.N_BUFFER_FRAMES
        self.N_BUFFER_FRAMES = bufsize
        self.current_frames = 0

    def push(self, line):
        if self.current_frames < self.N_BUFFER_FRAMES:
            self.stack = np.vstack((line, self.stack))
            self.current_frames+=1
        else:
            self.stack = np.delete(self.stack, self.N_BUFFER_FRAMES-1, axis = 0)
            self.stack = np.vstack((line, self.stack))

class Printer(object):
    Xterm = None    # To ensure that the Xterm is opened only when it's needed
                    # and only once during the entire program lifespan
    @staticmethod
    def Xterm(string, *_):
        if(Printer.Xterm == None):
            PIPE_PATH = "/tmp/Acceleration_Prediction"
            if not os.path.exists(PIPE_PATH):
                os.mkfifo(PIPE_PATH)
            Popen(['xterm', '-e', 'tail -f %s' % PIPE_PATH])

        with open(PIPE_PATH, "w") as p:
            p.write("\n" + string)

    @staticmethod
    def Term(string, *_):
        print(string)

    @staticmethod
    def OpenCV(string, dashboard, color = None):
        colors = {   # Colors to be Used
               "WHITE" : (255, 255, 255),
               "BLUE"  : (250, 100, 20),
               "GREEN" : (60, 255, 0),
               "RED"   : (0, 0, 255),
               "BLACK" : (0, 0, 0),
        }
        img = np.zeros((height,width,3), dtype = "uint8")  # As default dtype is 'float64'

        if color is not None:
            image_copy = deepcopy(img)
            cv.CV_FILLED = -1
            cv.rectangle(image_copy, (540+260,0), (540+260+100,175), colors[color], cv.CV_FILLED)
            img = cv.addWeighted(image_copy, 0.4, img, 0.6, 0)
        else:
            font                   = cv.FONT_HERSHEY_SIMPLEX
            upperRightCorner = (width - 460, 50)
            fontScale              = 0.75
            fontColor              = (255,255,255)
            lineType               = 2
            cv.putText(img, string,
                upperRightCorner,
                font,
                fontScale,
                fontColor,
                lineType)

        return img


class Acc_Predictor(object):
    object = None
    CALLSTEP = 5
    STACK_SIZE = 2

    def __init__(self, frame):  # Intended to be working on ROI Image
        Buffer.frames = Buffer(frame[np.newaxis, ...], Acc_Predictor.STACK_SIZE)

    def learn(self, frame):
        Buffer.frames.push(frame[np.newaxis, ...])

    def distance(self, frame):
        # Buffer.frames.stack.shape(STACK_SIZE, 720, 1280)
        # frame.shape(720, 1280)
        return np.mean(np.abs(Buffer.frames.stack - frame))

    def predict_acc(self, frame, output = "Xterm", dashboard = None, colors = False):
        printfunc = getattr(Printer, output)

        img = None  # To capture the output image in case outputSytle is OpenCV
        if(colors == True):  # Little if-else to give extra control to calling statement
            colors = ["BLACK", "GREEN", "RED"]
        else:             # It just helps us toggle between the two modes easily
            colors = [None, None, None]


        if Buffer.frames.current_frames < Buffer.frames.N_BUFFER_FRAMES:
            img = printfunc("Prediction Unavailable! Still Training...", dashboard, colors[0])
        else:
            MAX_IMAGE_DIFFERENCE = 1.7
            if(self.distance(frame) < MAX_IMAGE_DIFFERENCE):
                img = printfunc("Suggested Acceleration: +ve or 0", dashboard, colors[1])
            else:
                img = printfunc("Suggested Acceleration: -ve", dashboard, colors[2])
        return img




# ██    ██ ████████ ██ ██      ██ ████████ ██    ██
# ██    ██    ██    ██ ██      ██    ██     ██  ██
# ██    ██    ██    ██ ██      ██    ██      ████
# ██    ██    ██    ██ ██      ██    ██       ██
#  ██████     ██    ██ ███████ ██    ██       ██

COLORS = {   # Colors to be Used
        "lane_color" : (250, 100, 20),
        "region_stable" : (60, 120, 0),
        "region_unstable" : (60, 80, 255),
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

def distance_to_lane_line(l1, l2):
    x = (l1[2] + l1[0]) / 2     # Find Average Point
    y = (l1[3] + l1[1]) / 2     # on Current Line Obj
    m = cal_slope(l2)
    c = cal_intercept(l2, m)
    distance = abs(m*x - y + c) / math.sqrt(m**2 + 1)
    return distance

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
        else:
            Buffer.left.push(lines[0])   # Only push when buffer is not used

        if np.array_equal(lines[1], np.array([0, 0, 0, 0])):
            lines[1] = np.average(Buffer.right.stack, axis=0)
        else:
            Buffer.right.push(lines[1])   # Only push when buffer is not used

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

def check_dist_condition(l1, l2):
    CRITICAL_DISTANCE_CHANGE = 8
    dist = distance_to_lane_line(l1, l2)
    if dist > CRITICAL_DISTANCE_CHANGE:
        return False
    return True

def cal_slope(line):
    x1, y1, x2, y2 = line
    if x2-x1 == 0:
        return 2147483647
    return ((y2-y1)/(x2-x1))

def cal_intercept(line, slope):
    return line[1] - slope * line[0]

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

            stability1 = check_slope_condition(bufline_slope, lane_slope)
            stability2 = check_dist_condition(laneline, bufline)
            stability = stability1 & stability2
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

def use_predictor(frame):
    canny_accPredict = cv.Canny(frame, 50, 150)
    roi_accPredict = ROI(canny_accPredict)
    if(Acc_Predictor.object == None):
        Acc_Predictor.object = Acc_Predictor(roi_accPredict)
    Acc_Predictor.object.learn(roi_accPredict)
    return Acc_Predictor.object.predict_acc(roi_accPredict, output = "OpenCV", dashboard = frame, colors = False)

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
vin = cv.VideoCapture(VIDEO)
if(not vin.isOpened()):
    vin.open()

first = True
frame_count = 0
while(True):
    # cameraturing the video feed frame by frame
    ret, frame = vin.read()

    if(ret):
        # Operations on the frames
        detection = image_pipeline(frame)


        if first == True:   # Silly way of making sure that prediction is only
                            # initalized once, after the global h & w are updated
            prediction = np.zeros((height,width,3), dtype = "uint8")
            first = False
        if(frame_count == Acc_Predictor.CALLSTEP):
            prediction = use_predictor(detection)
            frame_count = 0
        else: frame_count += 1


        detection = cv.add(detection, prediction)

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
