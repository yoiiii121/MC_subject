import math as ma

import cv2
import numpy as np
from matplotlib import pyplot as plt
dir = "./barcodes/"

#name = dir + "QRCode_twotypes.png"
#name = dir + "QRCode_screenshot.png"

#name = dir + "QRCode_bmp.bmp"


#name = dir + "QRCode_mobile.png"
#name = dir + "QRCode_tatto.jpg"
name = dir + "QRCode_web.png"

# name = "./ruta/image"

def angle(pt1, pt2, pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return (dx1 * dx2 + dy1 * dy2) / ma.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10)

"""
def findSquaresInImage(image):

    blurred = np.mat(image.shape,np.dtype('u8'))
    blurred = cv2.blur(blurred, (5, 5))
    cv2.medianBlur(image, 9, blurred)
    #gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    #ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.GRAY

    #gray0 =np.mat(,"uint8")
    gray0 = np.mat(blurred.copy(), np.dtype('u8'))
    gray = np.mat(image.shape,np.dtype("u8"))
    squares = []
    contours = []
    for c in range(3):
        ch = [c, 0]

        cv2.mixChannels(blurred, gray0, ch)
        # mixChannels, 1, &gray0, 1, ch, 1)
        threshold_level = 2
        for l in range(threshold_level):
            if l == 0:

                cv2.Canny(gray0, 10, 20, gray, 3)

                cv2.dilate(gray, np.array([]), gray,(-1,-1))
            else:
                gray = gray0 >= (l + 1) * 255 / threshold_level

            (_, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        approx = []
        for i in range(len(contours)):

            cv2.approxPolyDP(contours[i], approx, cv2.arcLength(contours[i], True) * 0.02, True)
            if (len(approx) == 4 and np.fabs(cv2.contourArea(approx)) > 1000 and cv2.isContourConvex(approx)):
                maxCosine = 0

                for j in range(2, 5):

                    cosine = np.fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]))
                    maxCosine = max(maxCosine, cosine)

                    if maxCosine < 0.3:
                        squares.append(approx)
    return squares
"""
# origin image read
image = cv2.imread(name, cv2.IMREAD_COLOR)
#image = findSquaresInImage(image)
image_original = image.copy()
#cv2.imshow("Image", image)
#cv2.waitKey(0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# You need to choose 4 or 8 for connectivity type
connectivity = 8
# Perform the operation
output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]
# The fourth cell is the centroid matrix
centroids = output[3]
ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("Image", image)
cv2.waitKey(0)

h, w = image.shape

ratio = round(10 * h / w)

# kernel creation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ratio, ratio))
# Morphology operations
closed = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)

# find the contours in the thresholded image, then sort the contours0
# by their area, keeping only the largest one
(_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)

box = np.intc(cv2.boxPoints(rect))
box = np.intp(box)
# vertex selection
min_x = 99999
max_x = 0
min_y = 99999
max_y = 0
for i in box:
    if i[0] < min_x:
        min_x = i[0]
    if i[1] < min_y:
        min_y = i[1]
    if i[0] > max_x:
        max_x = i[0]
    if i[1] > max_y:
        max_y = i[1]
# cut the image
image = image[min_y:max_y, min_x:max_x]
image=cv2.resize(image,(450,450))

cv2.imshow("Image", image)
cv2.waitKey(0)

ratio = round(7 * h / w)
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (ratio, ratio))


closed2 = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel2)

(_, contours, _) = cv2.findContours(closed2.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
squares = []
approx = []
for i in range(len(contours)):

    approx=cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i], True) * 0.02, True)

    if (len(approx) == 4 and ma.fabs(cv2.contourArea(approx)) > 1000 and cv2.isContourConvex(approx)):
        maxCosine = 0

        for j in range(2, 5):

            cosine = ma.fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]))
            maxCosine = max(maxCosine, cosine)

            if maxCosine < 0.3:
                squares.append(approx)

#
boxes = []
for i in squares:
    rect = cv2.minAreaRect(i)
    box = np.intc(cv2.boxPoints(rect))
    box = np.intp(box)
    boxes.extend(box)
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imshow("Image", image)
cv2.waitKey(0)
centroids =[]
aux = []
for i in range(0,len(boxes),4):
    min_x = 99999
    max_x = 0
    min_y = 99999
    max_y = 0
    if boxes[i][0] < min_x:
        min_x = boxes[i][1]
    if boxes[i][1] < min_y:
        min_y = boxes[i][1]
    if boxes[i][0] > max_x:
        max_x = boxes[i][0]
    if boxes[i][1] > max_y:
        max_y = boxes[i][1]
    centroids.append(ma.fabs((max_x-min_y)/2+(max_y-min_x)/2))
less_centroids = []
"""for i in centroids:
    if i not in less_centroids:
        less_centroids.append(i)
        for j in less_centroids:
            if i-29 <= j <= i+29 and i != j:
                less_centroids.remove(i)

print(less_centroids)"""