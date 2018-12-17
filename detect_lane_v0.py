## TODO: STABILITY Bugs

import numpy as np
import cv2 as cv
import math

height, width = (600,600)  # Dimension of the image, crucial for coord sys
                           # functioning. These values are arbitary, it'll
                           # be updated locally within the image pipeline

class Lane(object):
    """
     Domain knowledge for lane recognition and lane filtering. It serves as a
     container for all the neccessary operations and the elemental lines defining
     the line itself.
    """

    CRITICAL_SLOPE_CHANGE = 0.1
    MIN_SLOPE = 0.46
    MAX_SLOPE = 1.73
    MAX_SLOPE_DIFFERENCE = 0.8
    MAX_DISTANCE_FROM_LINE = 20
    N_BUFFER_FRAMES = 10
    COLORS = {
            "lane_color" : (0, 0, 255),
            "region_stable" : (60, 80, 0),
            "region_unstable" : (60, 80, 255),
            "left_line" : (220, 40, 60),
            "right_line" : (255, 0, 255),
    }
    THICKNESS = 5
    LINE_SIDE_RANGES =  { "left_line" : range(int(width/2)), # String characters used
                          "right_line": range(int(width/2), int(width))} # in fit_lane_line()
    DECISION_MAT = [[.1, .9], [1, 0]]
    # 0.1 * frame position + 0.9 * avg from buffer  : In case of unstable lane.
    # 1 * frame position + 0 * avg from buffer      : In case of stable lane.

    left_line = None    # DataType: <class= Lane> Object
    right_line = None

    @staticmethod
    def status():
        # return all([Lane.left_line, Lane.right_line])
        return (Lane.left_line is not None or Lane.right_line is not None)

    @staticmethod
    def purge():
        Lane.left_line = None
        Lane.right_line = None

    @staticmethod
    def update_vanishing_point(left, right):
        """
         Calculates the Point of Intersection of the two provided lines

         Parameters
         ---------
            'Lane' class Objects

         Result
         -------
            Updates the LaneObj.vanishing_point
        """
        equation = left.line.parameters - right.line.parameters
        x = -equation[1] / equation[0]
        y = np.poly1d(left.line.parameters)(x)
        x, y = map(int, [x, y])

        left.vanishing_point = [x, y]
        right.vanishing_point = [x, y]


    @staticmethod
    def junk_filter(lines):
        parameters = []
        if lines is not None:
            for line in lines:
                if line.candidate:
                        parameters.append(line.parameters)
        parameters_avg = np.average(parameters, axis=0)
        if parameters_avg.size == 2:
            return Line(parameters = parameters_avg)
        else: return Line()



    def __init__(self, lines):
        # Stability flag. Tripped to False if the slope changes too rapidly
        self.stable = True

        self.detected_line = Lane.junk_filter(lines)

        # Buffer for lane line smoothing
        self.buffer = np.array(Lane.N_BUFFER_FRAMES * [self.detected_line.parameters])

        # Final publicly available lane line object
        if np.all(self.buffer[0]):
            self.line = Line(parameters = self.buffer[0])
        else:   self.line = Line()

    @property
    def m(self):              # Slope of the lane line
        if len(self.line.parameters) > 2:
            return Exception("ERROR: Slope can't be computed for a Higher order polynomial Lane!")
        return self.line.parameters[0]

    @property
    def c(self):              # Intercept of the lane line
        if len(self.line.parameters) > 2:
            return Exception("ERROR: Slope can't be computed for a Higher order polynomial Lane!")
        return self.line.parameters[1]

    def update_lane_line(self, lines):
        """
         The main client method for dealing with lane updates
        """
        average_buffer = np.average(self.buffer, axis=0)     # Initialising Stuff
        # self.line.parameters = np.average(self.buffer, axis=0)
        self.update_detected_line(lines)

        weights = Lane.DECISION_MAT[self.stable]
        current_buffer_top = np.dot(weights, np.vstack([self.detected_line.parameters, average_buffer]))
        self.buffer = np.insert(self.buffer, 0, current_buffer_top, axis=0)[:-1]  # Updating Buffer
        if np.all(self.buffer[0]):
            self.line = Line(parameters = self.buffer[0])  # Updating the Lane Line
        else:   self.line = Line()


    def update_detected_line(self, lines):
        average_buffer = np.average(self.buffer, axis=0)

        detected_line = Lane.junk_filter(lines)   # Generating Values of Points
        if detected_line.parameters is [0,0]:                           # Describing a Lane Line
            detected_line = Line(parameters = average_buffer)
            self.stable = False

        self.detected_line = detected_line   # Updating currentLLC to object


        buffer_slope = average_buffer[0]            # Checking Stability
        current_slope = self.detected_line.parameters[0]
        if abs(current_slope - buffer_slope) > Lane.CRITICAL_SLOPE_CHANGE:
            self.stable = False             # No need to return anything
        else: self.stable = True            # We directly update the object


class Line(object):
    """
     Line: y = mx + c.
     A line can be described by its pair of coordinates (x1, y1), (x2, y2).
     To formalize a line, we need to compute its slope (m) and intercept (c).

     Object Creation: Coords List = [x1,y1,x2,y2] or Parameters: = [m, c]
    """
    def __init__(self, coords = None, parameters = None):
        # if(not np.all(parameters)):
        #     print("\n\n\n FFFFFFF \n\n\n")
        #     self.x1, self.y1, self.x2, self.y2, self.m, self.c, self.position = 0,0,0,0,0,0,None
        #     self.parameters = [self.m, self.c]
        #     self.coords = [self.x1, self.y1, self.x2, self.y2]

        if coords is not None:
            self.x1, self.y1, self.x2, self.y2 = coords     # Unpack Coordinates
            self.m = self.cal_slope()    # Compute Attributes
            self.c = self.cal_intercept()
            self.coords = coords                                # init coords
            self.parameters = [self.m, self.c]                  # init parameters
            self.position = self.assign_position()             # init position

        elif parameters is not None:
            self.m, self.c = parameters                                    # Unpack Parameters
            self.x1, self.y1, self.x2, self.y2 = Line.make_coords(np.array([self.m, self.c]))  # Compute Attributes
            self.coords = [self.x1, self.y1, self.x2, self.y2]      # init coords
            self.parameters = parameters                            # init parameters
            self.position = self.assign_position()
        else:
            self.x1, self.y1, self.x2, self.y2, self.m, self.c, self.position = 0,0,0,0,0,0,None
            self.parameters = [self.m, self.c]
            self.coords = [self.x1, self.y1, self.x2, self.y2]

    def __repr__(self):         # Magic Function for Printing Object through print()
        return 'Line: x1={}, y1={}, x2={}, y2={}, m={}, c={}, candidate={}, position={}'.format(
                self.x1, self.y1, self.x2, self.y2, round(self.m,2),
                round(self.c,2), self.candidate, self.position)

    def cal_slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    def cal_intercept(self):
        return self.y1 - self.m * self.x1

    def assign_position(self):
        if self.m < 0.0: return 'left_line'
        else: return 'right_line'

    @staticmethod
    def make_coords(l_parameters):  # Create coords from given line parameters
        """
         Generate end points coordinates using given slope & intercept
         parameters for a line.
         Args: l_parameters - numpy array of [slope, intercept]      # TODO: Check if "numpy" array is req or list is enough

         Result: Returns a numpy array(4,) representing a line as a list
                 of [x1, y1, x2, y2].
        """
        slope, intercept = l_parameters
        y1 = height
        y2 = int(y1*(0.67))
        # y2 = int(roi_height)   # Making sure that the lane lines are within the ROI
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return np.array([x1, y1, x2, y2])

    @property
    def distance_to_lane_line(self):
        lane_line = getattr(Lane, self.position) # self.position == "left_line" string
        if lane_line is None: return None   # Basically Lane.position, which is a Lane obj
                                            # Equalivalent to: Lane.left_line
        x = (self.x2 + self.x1) / 2     # Find Average Point
        y = (self.y2 + self.y1) / 2     # on Current Line Obj
        m = lane_line.line.m
        c = lane_line.line.c
        distance = abs(m*x - y + c) / math.sqrt(m**2 + 1)
        return distance

    @property
    def candidate(self):
        """
        A simple domain logic to check whether this hough line can be a candidate
        for being a segment of a lane line.
        1. The line cannot be horizontal and should have a reasonable slope.
        2. The difference between lane line's slope and this hough line's cannot be too high.
        3. The hough line should not be far from the lane line it belongs to.
        4. The hough line should be below the vanishing point.
        """
        if(abs(self.m) > Lane.MAX_SLOPE or abs(self.m) < Lane.MIN_SLOPE): return False
        lane_line = getattr(Lane, self.position)  # CLEVER TRICK to map the object to its side lane line
        if lane_line:
            if abs(self.m - lane_line.line.parameters[0]) > Lane.MAX_SLOPE_DIFFERENCE: return False
        return True


def ROI(img):
    ROI_vertices = np.array([[0,height-1],
                             [width/2, int(0.5*height)],
                             [width-1, height-1]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv.fillPoly(mask, [ROI_vertices], (255, 255, 255))
    masked = cv.bitwise_and(img, mask)
    return masked

def draw_lanes(img, lines):
    lanes = np.zeros_like(img)  # "img" must have the same color space as original
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(lanes, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lanes

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
            left_lines = Line.make_coords(left_fit_avg)
            right_fit_avg = np.average(right_fit, axis=0)
            right_lines = Line.make_coords(right_fit_avg)
            return np.array([left_lines, right_lines])

        elif left_fit != []:
            left_fit_avg = np.average(left_fit, axis=0)
            left_lines = Line.make_coords(left_fit_avg)
            return np.array([left_lines, np.array([0, 0, 0, 0])])

        elif right_fit != []:
            right_fit_avg = np.average(right_fit, axis=0)
            right_lines = Line.make_coords(right_fit_avg)
            return np.array([np.array([0, 0, 0, 0]), right_lines])

        else:
            return np.array([np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])])
    else:
        print("No Lines Detected!")

# ██    ██ ████████ ██ ██      ██ ████████ ██    ██
# ██    ██    ██    ██ ██      ██    ██     ██  ██
# ██    ██    ██    ██ ██      ██    ██      ████
# ██    ██    ██    ██ ██      ██    ██       ██
#  ██████     ██    ██ ███████ ██    ██       ██


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

def hough_line_transform(image, rho, theta, threshold, min_line_length, max_line_gap):
    """
     A modified implementation of a suggested `HoughLinesP` function which allows
     Line objects initialization and in-place line filtering.

     Result
     -------
     Returns a list of Line instances which are considered segments of a lane.

     Functioning
     -------
        - map() takes one element from the iterable "lines", then applies the function to it
        - Inside the function, the line.shape(1,4). It takes the first & only element with line[0]
        - It then passes the this shape(4,) list as a list of arguments to initialize the Line object
        - NOTE: We don't do line[0] instead *line[0] because we are defining a function with arbitary args [REMOVE]
        - Those Line objects are then filtered through using the self.candidate method of the object
    """
    lines = cv.HoughLinesP(image, rho, theta, threshold, np.array([]),  #lines.shape(n,1,4)
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is not None:
        filtered_lines = list(  filter(lambda l: l.candidate,   map(lambda line: Line(line[0]), lines))  )
        # filtered_lines = list(map(lambda line: Line(line[0]), lines))
        return filtered_lines
    else: return None

def update_lane_events(lines):
    if lines is not None:
        left = [line for line in lines if line.position == 'left_line']
        right = [line for line in lines if line.position == 'right_line']
        if not Lane.status():
            Lane.left_line = Lane(left)
            Lane.right_line = Lane(right)
        # Lane.update_vanishing_point(Lane.left_line, Lane.right_line)
        Lane.left_line.update_lane_line([l for l in left if l.candidate])
        Lane.right_line.update_lane_line([r for r in right if r.candidate])
    else:
        print('No lane lines detected.')

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv.addWeighted(initial_img, α, img, β, λ)

def draw_lane_lines(img, lane_lines, shade_background=False):
    if shade_background: α = 0.8
    else: α = 1.
    lane_line_image = np.zeros_like(img)
    if lane_lines is not None:
        for lane_line in lane_lines:
            cv.line(lane_line_image, (lane_line.line.x1, lane_line.line.y1), (lane_line.line.x2, lane_line.line.y2),
                        Lane.COLORS['lane_color'], Lane.THICKNESS)
    return weighted_img(lane_line_image, img, α=α, β=1.)

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
    thresholded = cv.Canny(thresholded, 50, 150)  # TODO: Play with this threshold
    #====================++++++++++++================+++++++==============


    roi = ROI(thresholded)
    cv.imshow("ROI", roi)
    cv.waitKey(1)

    # lines = cv.HoughLinesP(roi, 2, np.pi/180, 50, None, minLineLength=20, maxLineGap=100)
    # averaged_lanes = clean_lane_lines(frame, lines)
    #
    # lanes = draw_lanes(frame, averaged_lanes)
    # detection = cv.addWeighted(frame, 0.8, lanes, 1, 0)
    #
    #
    lines = hough_line_transform(roi, 2, np.pi/180, 50, 20, 100)

    update_lane_events(lines)

    detection = draw_lane_lines(frame, [Lane.left_line, Lane.right_line], shade_background=True)
    return detection



# Open the capture stream
vin = cv.VideoCapture("/Users/AI-Mac1/Desktop/harder_challenge_video.mp4")
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
