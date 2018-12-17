import cv2 as cv

def edge_detect(frame, minthres, maxthres):
    # GrayScaling the Image
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred_frame = cv.GaussianBlur(gray_frame, (5, 5), 0)
    return cv.Canny(blurred_frame, minthres, maxthres)


# Reading the Source
frame = cv.imread("data/test_image.jpg", cv.IMREAD_COLOR)
images = []
for i in range(0, 240, 20):
    for j in range(0, 240, 20):
        images.append((edge_detect(frame, i, j), str(i)+", "+str(j)))


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
