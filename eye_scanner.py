import cv2

# Load the image
img = cv2.imread('image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the pre-trained Haar cascade classifier for detecting people
classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Detect people in the image
people = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# If no people are detected, print a message and exit
if len(people) == 0:
    print('No people detected in the image.')
    exit()

# Otherwise, loop through the detected people
for (x, y, w, h) in people:
    # Crop the face from the image
    face = img[y:y+h, x:x+w]
    
    # Convert the face to grayscale
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained Haar cascade classifier for detecting closed eyes
    eye_classifier = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    
    # Detect closed eyes in the face
    eyes = eye_classifier.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5)
    
    # If closed eyes are detected, print a message indicating that the person is sleeping
    if len(eyes) > 0:
        print('Person is sleeping.')
    # Otherwise, print a message indicating that the person is awake
    else:
        print('Person is awake.')

# Show the image with the rectangles drawn around the detected people
cv2.imshow('People Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#This code uses a Haar cascade classifier to detect people in the image.
#It then crops and analyzes their faces to determine if their eyes are closed or open.
#Closed eyes indicate sleep, while open eyes indicate wakefulness.
#If closed eyes are detected, a message is printed indicating that the person is sleeping.
#If open eyes are detected, a message is printed indicating that the person is awake.