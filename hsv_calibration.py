import numpy as np
import cv2


# read the image
image = cv2.imread("./three_ships_horizon.JPG")

# filter with bilateral filter to remove noise and remain the sharp edges 
blur = cv2.bilateralFilter(image,9,75,75)


# convert from BGR to HSV color space
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

# upper and lower values for hsv color space
uh = 255
us = 255
uv = 255
lh = 0
ls = 0
lv = 0

lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])


# Threshold the HSV image 
mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

window_name = "HSV Calibrator"
cv2.namedWindow(window_name)

def nothing(x):
    print("Trackbar value: " + str(x))
    pass

# create trackbars for Upper HSV
cv2.createTrackbar('UpperH',window_name,0,255,nothing)
cv2.setTrackbarPos('UpperH',window_name, uh)

cv2.createTrackbar('UpperS',window_name,0,255,nothing)
cv2.setTrackbarPos('UpperS',window_name, us)

cv2.createTrackbar('UpperV',window_name,0,255,nothing)
cv2.setTrackbarPos('UpperV',window_name, uv)

# create trackbars for Lower HSV
cv2.createTrackbar('LowerH',window_name,0,255,nothing)
cv2.setTrackbarPos('LowerH',window_name, lh)

cv2.createTrackbar('LowerS',window_name,0,255,nothing)
cv2.setTrackbarPos('LowerS',window_name, ls)

cv2.createTrackbar('LowerV',window_name,0,255,nothing)
cv2.setTrackbarPos('LowerV',window_name, lv)

font = cv2.FONT_HERSHEY_SIMPLEX


while(1):
    # Threshold the HSV image 
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    cv2.putText(mask,'Lower HSV: [' + str(lh) +',' + str(ls) + ',' + str(lv) + ']', (10,30), font, 0.5, (200,255,155), 1, cv2.LINE_AA)
    cv2.putText(mask,'Upper HSV: [' + str(uh) +',' + str(us) + ',' + str(uv) + ']', (10,60), font, 0.5, (200,255,155), 1, cv2.LINE_AA)

    cv2.imshow(window_name,mask)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    # get current positions of Upper HSV trackbars
    uh = cv2.getTrackbarPos('UpperH',window_name)
    us = cv2.getTrackbarPos('UpperS',window_name)
    uv = cv2.getTrackbarPos('UpperV',window_name)
    upper_blue = np.array([uh,us,uv])
    # get current positions of Lower HScv2 trackbars
    lh = cv2.getTrackbarPos('LowerH',window_name)
    ls = cv2.getTrackbarPos('LowerS',window_name)
    lv = cv2.getTrackbarPos('LowerV',window_name)
    upper_hsv = np.array([uh,us,uv])
    lower_hsv = np.array([lh,ls,lv])


cv2.destroyAllWindows()

