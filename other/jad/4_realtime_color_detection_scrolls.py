import numpy as np
import cv2

#Trackbar callback
def on_trackbar(val):
    # Extract values of trackbars
    hue_min = cv2.getTrackbarPos("Hue Min", "TrackedBars")
    hue_max = cv2.getTrackbarPos("Hue Max", "TrackedBars")
    sat_min = cv2.getTrackbarPos("Sat Min", "TrackedBars")
    sat_max = cv2.getTrackbarPos("Sat Max", "TrackedBars")
    int_min = cv2.getTrackbarPos("Int Min", "TrackedBars")
    int_max = cv2.getTrackbarPos("Int Max", "TrackedBars")
    
    # define range of blue color in HSV
    lower_col = np.array([hue_min,sat_min,int_min])
    upper_col = np.array([hue_max,sat_max,int_max])
        
    # Threshold the HSV image using inRange function to get only red colors
    mask = cv2.inRange(hsv, lower_col, upper_col)
    
    # Transform the mask to BGR to plot with colors
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Stack the images to display
    numpy_horizontal = np.hstack((frame, mask))

    # Display the resulting frame
    cv2.imshow("TrackedBars", numpy_horizontal)

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(0)

# Create trackbars to change the value of hue min and max and sat min
cv2.namedWindow("TrackedBars")
cv2.createTrackbar("Hue Min", "TrackedBars", 0, 179, on_trackbar)
cv2.createTrackbar("Hue Max", "TrackedBars", 0, 179, on_trackbar)
cv2.createTrackbar("Sat Min", "TrackedBars", 0, 255, on_trackbar)
cv2.createTrackbar("Sat Max", "TrackedBars", 0, 255, on_trackbar)
cv2.createTrackbar("Int Min", "TrackedBars", 0, 255, on_trackbar)
cv2.createTrackbar("Int Max", "TrackedBars", 0, 255, on_trackbar)


# If the input is the camera, pass 0 instead of the video file name
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Converting the image to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Trackbar callback
    on_trackbar(0)
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(1) == ord('q'):
      break

# Closes all the frames
cv2.destroyAllWindows()