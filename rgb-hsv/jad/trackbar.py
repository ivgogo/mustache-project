import numpy as np
import cv2

# Trackbar callback
def on_trackbar(val):
    # Extract values of trackbars
    hue_min = cv2.getTrackbarPos("Hue Min", "TrackedBars")
    hue_max = cv2.getTrackbarPos("Hue Max", "TrackedBars")
    sat_min = cv2.getTrackbarPos("Sat Min", "TrackedBars")
    sat_max = cv2.getTrackbarPos("Sat Max", "TrackedBars")
    
    # Opencv mostly works with --> 8bit images so we need to scale the image.

    # Scale the image to the range 0-255
    scaled_frame = (frame * 255).astype(np.uint8)

    # Converting the image to HSV
    hsv = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_col = np.array([hue_min, sat_min, 0])
    upper_col = np.array([hue_max, sat_max, 255])
        
    # Threshold the HSV image using inRange function to get only red colors
    mask = cv2.inRange(hsv, lower_col, upper_col)
    
    # Transform the mask to BGR to plot with colors
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Stack the images to display
    numpy_horizontal = np.hstack((scaled_frame, mask))

    # Display the resulting frame
    cv2.imshow("TrackedBars", numpy_horizontal)

# Load an image (replace 'your_image.npy' with the actual file path)
frame = np.load("/home/ivan/Descargas/mustache-project/rgb-hsv/jad/test.npy")
print(f'Image shape: {frame.shape}')

# Create trackbars to change the value of hue min and max and sat min
cv2.namedWindow("TrackedBars", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Hue Min", "TrackedBars", 0, 179, on_trackbar)
cv2.createTrackbar("Hue Max", "TrackedBars", 0, 179, on_trackbar)
cv2.createTrackbar("Sat Min", "TrackedBars", 0, 255, on_trackbar)
cv2.createTrackbar("Sat Max", "TrackedBars", 0, 255, on_trackbar)

# Trackbar callback
on_trackbar(0)

# Wait for a key event and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
