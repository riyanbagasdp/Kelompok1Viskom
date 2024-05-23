import cv2 as cv
import numpy as np

capture = cv.VideoCapture('data/mouse.mp4')
if not capture.isOpened():
    exit(0)


#knn
subsKNN = cv.createBackgroundSubtractorKNN()



i = 0
while capture.isOpened():
    re, frame = capture.read()
    scale = 20

    if isinstance(frame, type(None)):
        break

    width = int(frame.shape[1] * scale / 100)
    height = int(frame.shape[0] * scale / 100)

    dim = (width, height)
    image = cv.resize(frame,dim, cv.INTER_AREA)
    gaussian = np.array([
        [1.0, 4.0, 7.0, 4.0, 1.0],
        [4.0, 16.0, 26.0, 16.0, 4.0],
        [7.0, 26.0, 41.0, 26.0, 7.0],
        [4.0, 16.0, 26.0, 16.0, 4.0],
        [1.0, 4.0, 7.0, 4.0, 1.0]
    ])/273
    image = cv.filter2D(image,-1,gaussian)
    if i == 0:
        frame1 = image
        grayscaleframe1 = cv.cvtColor(frame1, cv.COLOR_RGBA2GRAY)
        i = i+1

    grayscaleframe = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
    framedelta = cv.absdiff(grayscaleframe1,grayscaleframe)


    #knn
    blobKNN = subsKNN.apply(image)

    cv.imshow("image knn",blobKNN)
    keyword = cv.waitKey(30)
    if keyword=='q' or keyword==27:
        break
cv.destroyAllWindows()
exit(0)