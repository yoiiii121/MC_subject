import math as ma

import cv2
import numpy as np
from matplotlib import pyplot as plt
dir = "./barcodes/"

#name = dir + "QRCode_screenshot.png"
#name = dir + "QRCode_bmp.bmp"
#name = dir + "QRCode_tatto.jpg"
name = dir + "QRCode_web.png"
# name = "./ruta/image"
space = 255
bar = 0

def read_container(image, x, y, limit,limit2):
    pattern = [space,bar,space,bar,space,bar,space]
    cont = 0
    xcopy = x
    value = int((limit2 + y)/2)
    for i in range(0, len(pattern)):
        if x <= limit and image[x][value] == pattern[i]:
            cont +=1
        while  x <= limit and image[x][value] == pattern[i]:
            x += 1
    value2 = int((limit + xcopy) / 2)
    for i in range(0, len(pattern)):
        if y <= limit2 and image[value2][y] == pattern[i]:
            cont += 1
        while y <= limit2 and image[value2][y] == pattern[i]:
            y += 1
    if len(pattern)*2 -cont == 1:
        return True
    else:
        return False

def angle(pt1, pt2, pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    aux1 = (dx1 * dx2 + dy1 * dy2)
    aux2 =(dx2 * dx2 + dy2 * dy2)
    aux3 = (dx1 * dx1 + dy1 * dy1)
    try:
        return aux1/ ma.sqrt(aux3 * aux2 + 1e-10)
    except RuntimeWarning:
        return 0

"""
def findSquaresInImage(image):

    blurred = cv2.blur(image, (5, 5))
    blurred = cv2.medianBlur(blurred, 9)
    gray0 = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    #ret, gray0 = cv2.threshold(gray0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.GRAY

    #
    gray = np.mat(image.shape,np.dtype("u8"))

    squares = []
    contours = []
    for c in range(3):
        ch = [c, 0]

        cv2.mixChannels( blurred,gray0, ch)
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
#exit(0)
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


image_original = image.copy()

h, w = image.shape

boolean = False
for rat in range(20,1, -1):
    boolean = False
    cont_loop = 0

    while boolean == False:

        try:

            ratio = round(rat * h / w)
            image = image_original.copy()
            # kernel creation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ratio, ratio))
            # Morphology operations
            closed = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

            closed = cv2.erode(closed, None, iterations=4)
            closed = cv2.dilate(closed, None, iterations=4)

            # find the contours in the thresholded image, then sort the contours0
            # by their area, keeping only the largest one
            (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            c = sorted(cnts, key=cv2.contourArea, reverse=True)[cont_loop]
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

            image_modify = image.copy()

            for cont_loop2 in range(2,12):
                ratio = cont_loop2
                kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (ratio, ratio))

                image = image_modify.copy()
                closed2 = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel2)
                closed2 = cv2.erode(closed2, None, iterations=1)
                closed2 = cv2.dilate(closed2, None, iterations=1)
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
                output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

                ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                image_without_draw = image.copy()
                boxes = []
                for i in squares:
                    rect = cv2.minAreaRect(i)
                    box = np.intc(cv2.boxPoints(rect))
                    box = np.intp(box)
                    boxes.extend(box)
                    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)


                centroids =[]
                aux = []
                min_x = 99999
                max_x = 0
                min_y = 99999
                max_y = 0
                boolean2 = False

                for i in range(0,len(boxes)):
                    if boxes[i][0] < min_x:
                        min_x = boxes[i][1]
                    if boxes[i][1] < min_y:
                        min_y = boxes[i][1]
                    if boxes[i][0] > max_x:
                        max_x = boxes[i][0]
                    if boxes[i][1] > max_y:
                        max_y = boxes[i][1]

                    #print(min_x,max_x)
                    #for x_container in range(0,int(len(image)/2),10):
                    if( min_x >max_x):
                        aux = min_x
                        min_x = max_x
                        max_x = aux
                    if(min_y > max_y):
                        aux = min_y
                        min_y = max_y
                        max_y = aux
                    if not boolean2:
                        boolean2=read_container(image_without_draw,int(min_x),int(min_y),int(max_x),int(max_y))
                    if boolean2:
                        centroids.append(ma.fabs((max_x-min_y)/2+(max_y-min_x)/2))
                boolean2 = False
                less_centroids = []
                value = 0
                # redundant value eliminate
                if len(centroids) >=9:
                    for i in centroids:
                       if i not in less_centroids:
                            less_centroids.append(i)
                group_tree = []
                maximun_space = 50.

                #Check numbers of centroids
                eliminate = less_centroids.copy()
                for i in range(len(less_centroids)-1):

                    for j in range(i+1,len(less_centroids)):
                        if less_centroids[j] in eliminate and maximun_space>= ma.fabs(less_centroids[i] - less_centroids[j]) >= 0.:
                            eliminate.remove(less_centroids[j])
                if len(eliminate) >=3:

                    # right-angled triangle
                    for i in range(len(less_centroids)-2):
                        for j in range(i+1,len(less_centroids)-1):
                            for k in range(j+1,len(less_centroids)):
                                if boolean == False and 1.1>= round(ma.atan((less_centroids[i] - less_centroids[j])
                                                                                    / (less_centroids[j] - less_centroids[k])),1) >= 0.8:
                                    value = ma.fabs(round(ma.atan((less_centroids[i] - less_centroids[j])
                                                                  / (less_centroids[i] - less_centroids[k])),2)
                                                    - round(ma.atan((less_centroids[j] - less_centroids[k]) / (less_centroids[i] - less_centroids[k])),2))
                                    if 0.00 <= value <= 1.0:
                                        boolean = True
                if boolean:

                    cv2.imshow("Image", image)
                    cv2.waitKey(0)
                    print("Possibly QR - detected")
                    break
                else:
                    cont_loop += 1
            else:
                cont_loop +=1
        except:
            boolean = True
            pass
cv2.destroyAllWindows()
