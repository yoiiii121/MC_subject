from os import listdir

import cv2
import numpy as np
from matplotlib import pyplot as plt

filenames = []
dir = "./barcodes/"
for element in listdir(dir):
    filenames.append(element)

# names contains of all photos in filenames
# you can change the number for changing the photo
name = dir + filenames[3]
# Linux
# name = "./ruta/image"
# Windows
# name = ".\\ruta\\image"

# origin image read
image = cv2.imread(name, cv2.IMREAD_COLOR)
image_original = image.copy()
cv2.imshow("Image", image)
cv2.waitKey(0)
# is a bmp image?
if name[-3:].lower() != "bmp":
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    r = 600 / image.shape[1]
    dim = (600, int(image.shape[0] * r))
    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", resized)
    cv2.waitKey(0)
    # gray scale converter
    image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
else:
    # gray scale converter
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# canny and threshold the image
canny = cv2.Canny(image, 100, 200)
(_, thresh) = cv2.threshold(canny, 25, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# kernel creation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
# Morphology operations
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# closed = cv2.erode(closed, None, iterations=4)
# closed = cv2.dilate(closed, None, iterations=4)

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
(_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.intc(cv2.boxPoints(rect))
box = np.intp(box)
image2 = image.copy()
cv2.drawContours(image2, [box], -1, (0, 255, 0), 3)
cv2.imshow("Image", image2)
cv2.waitKey(0)

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
cv2.imshow("Image", image)
cv2.waitKey(0)

# perform the actual resizing of the image and show it
dim = (600, 600)
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


# cv2.imwrite("recorte.jpg",image)

def stand_out_borers(image):
    # Load the image, convert it to grayscale, and blur it
    # slightly to remove high frequency edges that we aren't
    # interested in

    # perform a series of erosions and dilations

    # When performing Canny edge detection we need two values
    # for hysteresis: threshold1 and threshold2. Any gradient
    # value larger than threshold2 are considered to be an
    # edge. Any value below threshold1 are considered not to
    # ben an edge. Values in between threshold1 and threshold2
    # are either classified as edges or non-edges based on how
    # the intensities are "connected". In this case, any gradient
    # values below 30 are considered non-edges whereas any value
    # above 150 are considered edges.
    # canny = cv2.Canny(image, 30, 150)
    # cv2.imshow("Image", canny)
    # cv2.waitKey(0)

    # Read the image you want connected components of
    # Threshold it so it becomes binary
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # You need to choose 4 or 8 for connectivity type
    connectivity = 4
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
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    return image


image = stand_out_borers(resized)


# Hough transformation
def hough_transformation(image):
    edges = cv2.Canny(image, 100, 200, apertureSize=3)
    minLineLength = 10
    maxLineGap = 20
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite("hough.jpg", image)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    return image


image = hough_transformation(image)


# Fourier transformation
def fourier_transformation(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    rows, cols = image.shape
    crow, ccol = rows / 2, cols / 2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[int(crow - 30):int(crow + 30), int(ccol - 30):int(ccol + 30)] = 1

    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_back, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    return image
image = fourier_transformation(image)

# Morphology transformations
image = cv2.dilate(image, None, iterations=1)
image = cv2.erode(image, None, iterations=2)
image = cv2.dilate(image, None, iterations=1)
cv2.imshow("Image", image)
cv2.waitKey(0)

# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction

cv2.imshow("Image", image)
cv2.imwrite("Image.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Decode algorithm
# Validation algorithm
def control_digit(digits):
    control = 0
    for i in range(0, len(digits), 2):
        control += int(digits[i]) * 3
    for i in range(1, len(digits), 2):
        control += int(digits[i])
    numero = 0
    for i in range(0, len(digits) - 1):
        if (control % 10 != 0):
            numero += 1
    return str(numero)


space = 0
bar = 255

# the next bar
def align_boundary(image, x, y, start, end):
    if (image[y][x] == end):
        while (image[y][x] == end):
            x += 1
    else:
        while (image[y][x - 1] == start):
            x -= 1
    return x

# Read a digit
def read_digit(image, xcurr, ycurr, unit_width, l_code, g_code, r_code, position):
    # Read the 7 consecutive bits.
    pattern = [0, 0, 0, 0, 0, 0, 0]
    for i in range(0, len(pattern)):
        for j in range(0, unit_width):
            if image[ycurr, xcurr] == bar:
                pattern[i] += 1
            xcurr += 1

        # See below for explanation.
        if (pattern[i] == 1 and image[ycurr][xcurr] == bar or pattern[i] == unit_width - 1 and image[ycurr][
            xcurr] == space):
            xcurr -= 1

    # Convert to binary, consider that a bit is set if the number of bars encountered is greater than a threshold.
    threshold = unit_width / 2
    v = ""
    for i in range(0, len(pattern)):
        v += "1" if pattern[i] >= threshold else "0"

    # Lookup digit value.
    digit = ""
    if position == "LEFT":
        if parity(v) == 1:
            # odd parity
            digit = l_code.get(v)
            encoding = "L"
        else:
            # even parity
            digit = g_code.get(v)
            encoding = "G"
        xcurr = align_boundary(image, xcurr, ycurr, space, bar)

    else:

        digit = r_code.get(v)
        encoding = "R"
        xcurr = align_boundary(image, xcurr, ycurr, bar, space)
    cv2.imshow("numbers", image)
    cv2.imwrite("numbers.jpg", image)
    return (xcurr, digit, encoding)

# parity calculation
def parity(cad):
    cont = 0
    for i in cad:
        if i == "1":
            cont += 1
    return cont % 2

# first bar
def skip_quiet_zone(image, x, y):
    while image[y][x] == space:
        x += 1
    return x

# first left digit
def read_lguard(image, x, y):
    widths = [0, 0, 0]
    pattern = [bar, space, bar]
    for i in range(0, len(pattern)):
        while (image[y][x] == pattern[i]):
            x += 1
            widths[i] += 1
    return (x, widths[0])

# skip control middle zone
def skip_mguard(image, x, y):
    pattern = [space, bar, space, bar, space]
    for i in range(0, len(pattern)):
        while image[y][x] == pattern[i]:
            x += 1
    return x


def __checkDigit(digits):
    total = sum(digits) + sum(digits[-1::-2] * 2)
    return (10 - (total % 10)) % 10


def validateCheckDigit(barcode=''):
    if len(barcode) in (8, 12, 13, 14) and barcode.isdigit():
        digits = list(map(int, barcode))
        checkDigit = __checkDigit(digits[0:-1])
        return checkDigit == digits[-1]
    return False


#  barcode read all
def read_barcode(image):
    digits = []
    # invert colors
    image = cv2.bitwise_not(image)

    l_code = {"0001101": 0, "0011001": 1, "0010011": 2, "0111101": 3, "0100011": 4,
              "0110001": 5, "0101111": 6, "0111011": 7, "0110111": 8, "0001011": 9}

    g_code = {"0100111": 0, "0110011": 1, "0011011": 2, "0100001": 3, "0011101": 4,
              "0111001": 5, "0000101": 6, "0010001": 7, "0001001": 8, "0010111": 9}

    r_code = {"1110010": 0, "1100110": 1, "1101100": 2, "1000010": 3, "1011100": 4,
              "1001110": 5, "1010000": 6, "1000100": 7, "1001000": 8, "1110100": 9}

    first_digit = {"LLLLLL": 0, "LLGLGG": 1, "LLGGLG": 2, "LLGGGL": 3, "LGLLGG": 4,
                   "LGGLLG": 5, "LGGGLL": 6, "LGLGLG": 7, "LGLGGL": 8, "LGGLGL": 9}

    position = {0: "LEFT", 1: "RIGHT"}

    for i in range(0, len(image[0])):
        xcurr = int(0)
        ycurr = int(i)
        list_d = []

        try:
            xcurr = skip_quiet_zone(image, xcurr, ycurr)

            (xcurr, unit_width) = read_lguard(image, xcurr, ycurr)
            digits_line = []
            encodigns_line = []
            # 6 left digits read
            for j in range(0, 6):
                d = "0000000"
                (xcurr, d, encodign) = read_digit(image, xcurr, ycurr, unit_width, l_code, g_code, r_code, position[0])
                digits_line.append([d, encodign])
                list_d.extend([d, encodign])

            xcurr = skip_mguard(image, xcurr, ycurr)
            # 6 right digits read
            for j in range(0, 6):
                d = "0000000"
                (xcurr, d, encodign) = read_digit(image, xcurr, ycurr, unit_width, l_code, g_code, r_code, position[1])
                digits_line.append([d, encodign])
                list_d.extend([d, encodign])
            digits.append(digits_line)
            cad = ""
            cad2 = ""
            size = len(list_d)
            if size == 24:
                for item in range(1, int(size / 2), 2):
                    cad += str(list_d[item])
                check = first_digit.get(cad)
                for item in range(0, size, 2):
                    cad2 += str(list_d[item])
                if check is not None:
                    list_complete = str(check) + cad2
                else:
                    list_complete = str(-1) + cad2
                if validateCheckDigit(list_complete):
                    return list_complete
        except:
            pass
    final_digits = []
    final_encodings = ""
    final_cad = ""
    for i in range(len(digits[0])):
        index = -1
        matches = []
        maxim_value = -1
        cont_value = -1
        for j in range(len(digits)):

            cad = ""
            for k in range(0, 10):
                matches.append(-1)
            for k in range(0, 10):
                if str(k) == str(digits[j][i][0]):
                    matches[k] += 1
                    cad += str(digits[j][i][0]) + digits[j][i][1]
            for k in range(0, 10):
                if (matches[k] > maxim_value):
                    maxim_value = matches[k]
                    index = k

            if i < 6:
                if (index != -1):
                    for n in range(0, 10):

                        for k in ["L", "G"]:
                            number = cad.count(str(n) + k)
                            if (cont_value < number):
                                cont_value = number
                                final_cad = str(n) + k
                else:
                    final_cad = ""
            else:
                if (index != -1):
                    final_cad = "0R"
                else:
                    final_cad = ""

        final_digits.append(str(index))
        final_encodings += final_cad[1:]
    print(final_encodings)
    first = first_digit.get(final_encodings[0:6])
    if first is not None:
        list = [first]
        list.extend(final_digits)
    else:
        list = ["-1"]
        list.extend(final_digits)
    return list

digits = read_barcode(image)
print("Digits: {}".format(digits))
cad = ""
for i in digits:
    if i is not None:
        cad += str(i)
print("Decode Validation: {}".format(validateCheckDigit(cad)))
