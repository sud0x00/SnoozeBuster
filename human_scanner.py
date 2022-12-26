import cv2

# Load the image
img = cv2.imread('image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the pre-trained Haar cascade classifier for detecting people
classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Detect people in the image
people = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Print the number of people detected
print(f'Number of people detected: {len(people)}')

# Loop through the detected people and draw a rectangle around them
for (x, y, w, h) in people:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

# Show the image with the rectangles drawn around the detected people
cv2.imshow('People Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Loads an image and converts it to grayscale
#Uses a pre-trained Haar cascade classifier to detect people in the image
#Prints the number of people detected and draws a rectangle around each person
#Displays the image with the rectangles drawn on it