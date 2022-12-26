import cv2
import time

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video
    ret, frame = cap.read()
    
    # Save the frame as an image
    cv2.imwrite('image.jpg', frame)
    
    # Wait for 1 minute before capturing another image
    time.sleep(60)

# Release the video capture
cap.release()
