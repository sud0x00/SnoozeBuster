import cv2

def is_sleeping(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar cascade classifier for detecting people
    classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

    # Detect people in the image
    people = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # If no people are detected, return False
    if len(people) == 0:
        return False

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

        # If closed eyes are detected, return True (indicating sleep)
        if len(eyes) > 0:
            return True

    # If no closed eyes are detected, return False (indicating wakefulness)
    return False

def is_slumped(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar cascade classifier for detecting people
    classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

    # Detect people in the image
    people = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # If no people are detected, return False
    if len(people) == 0:
        return False

    # Otherwise, loop through the detected people
    for (x, y, w, h) in people:
        # Calculate the angle of the body relative to the horizontal
        angle = calculate_body_angle(img, (x, y, w, h))

        # If the angle is greater than a certain threshold, return True (indicating a slumped posture)
        if angle > 20:
            return True

    # If the body angle is not greater than the threshold, return False (indicating an upright posture)
    return False



#    The first function, is_sleeping, uses a Haar cascade classifier to detect people in the image and analyzes their faces to determine if they are sleeping
#    The second function, is_slumped, uses a similar approach to determine if a person is slumped