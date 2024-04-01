import cv2

# Initialiser le soustracteur de fond MOG2
backSub = cv2.createBackgroundSubtractorMOG2()

# Ouvrir la vidéo
cap = cv2.VideoCapture('choregraphy/chore1/videos/1.MP4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Appliquer le soustracteur de fond à la frame
    fgMask = backSub.apply(frame)

    # Afficher le résultat de la soustraction de fond
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

cap.release()
cv2.destroyAllWindows()