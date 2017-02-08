import math as ma
import cv2
import numpy as np
from matplotlib import pyplot as plt

windows_name = "Photos"


def input_read(text, from_number, to_number):
    print(text)
    condition = True
    input_number = 0
    while condition:
        try:
            input_number = int(input("Input a number from {} to {}: ".format(from_number, to_number)))
        except ValueError:
            print("Value incorrect")
            input_number = 5
        if from_number <= input_number <= to_number:
            condition = False
    return input_number


def mean_square_error(img, other_img):
    m, n = img.shape[:2]
    err = np.double(0.0)
    const = ma.sqrt(1 / (m * n))
    for c1 in range(m):
        for c2 in range(n):
            err += (img[c1][c2] - other_img[c1][c2]) ** 2
    return const * err


def signal_to_noise_ratio_error(img, other_img):
    m, n = img.shape[:2]
    err = np.double(0.0)
    err_den = np.double(0.0)
    for c1 in range(m):
        for c2 in range(n):
            err += img[c1][c2] ** 2
            err_den += (other_img[c1][c2] - img[c1][c2]) ** 2
    return err / err_den


def similarity_measure_based_on_histogram(img, other_img):
    try:
        m, n = img.shape[:2]
        err = np.double(0.0)
        for c1 in range(256):
            err += np.abs(img[c1] - other_img[c1])
        return err / (2 * m * n)
    except ValueError:
        print("Function Similarity measure based on histogram not available")
        print()


def show_write_photo(text, img):
    img_copy = img.copy()
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.putText(img, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(windows_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = img_copy.copy()
    cv2.imwrite("./images/{}_result.jpg".format(text), img)


def color_read(text):
    return cv2.imread(text, cv2.IMREAD_COLOR)


def gray_scale_read(text):
    return cv2.imread(text, cv2.IMREAD_GRAYSCALE)


def unchanged_read(text):
    return cv2.imread(text, cv2.IMREAD_UNCHANGED)


def color_to_gray_scale(img):
    try:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        print("function Color to gray scale not available")
        print()


def color_to_gray_scale2(img):
    try:
        h, l, s = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
        return l
    except cv2.error:
        print("function Color to gray scale2 not available")
        print()


def negative_filter(img):
    return cv2.bitwise_not(img)


def threshold_filter(img):
    return cv2.threshold(img, 200.0, 255.0, cv2.THRESH_TRUNC)[1]


def median_normalized_filter(img):
    return cv2.medianBlur(img, 5)


def bilateral_normalized_filter(img):
    return cv2.bilateralFilter(img, 9, 75, 75)


def gaussian_normalized_filter(img):
    return cv2.GaussianBlur(img, (5, 5), 0)


def mean_normalized_filter(img):
    return cv2.blur(img, (5, 5))


def gamma_correlation_filter(img):
    gamma = 2.0
    inv_gamma = 1.0 / gamma
    table = np.array([((i_local / 255.0) ** inv_gamma) * 255
                      for i_local in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def equalize_hist(img):
    try:
        equ = cv2.equalizeHist(img)
        aux = np.hstack((img, equ))  # stacking images side-by-side
        return aux
    except cv2.error:
        print("Function Equalize hist not available")
        print()
    return None
    # similarity_measure_based_on_histogram(aux, equ)


def add_noise_randomly(img):
    aux = img.copy()
    cv2.randn(aux, aux.mean(), aux.std() / 5)
    cv2.add(img, aux, aux, mask=None)
    return aux


def contrast_filter(img):
    try:
        contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return contrast.apply(img)
    except cv2.error:
        print("Function Contrast filter not available")
        print()


def show_hist():
    plt.title(windows_name)
    plt.show()
    plt.close()
    cv2.destroyAllWindows()


def show_switcher(switcher):
    for i in switcher:
        print("{}: {}".format(i, switcher.get(i)))
    print()


bool_plot1 = False

def hist_cdf_normalized(img):
        if bool_plot1 == True:
            # http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
            hist, bins = np.histogram(img.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()

            plt.plot(cdf_normalized, color='b')
            plt.hist(img.flatten(), 256, [0, 256], color='r')
            plt.xlim([0, 256])
            plt.legend(('cdf', 'histogram'), loc='upper left')

bool_plot2 = False

def hist_historic(img):
    if bool_plot2 == True:
        try:
            color = ('b', 'g', 'r')
            for i_local, col in enumerate(color):
                historic = cv2.calcHist([img], [i_local], None, [256], [0, 256])
                plt.plot(historic, color=col)
                plt.xlim([0, 256])
            plt.legend(('b', 'g', 'r', 'histogram'), loc='upper left')
        except cv2.error:
            print("function hist_historic not available")
            print()


text_photo_switcher = {
    0: "poor_illumination1.jpg",
    1: "poor_illumination2.jpg",
    2: "poor_contrast1.jpg",
    3: "poor_contrast2.jpg",
    4: "noisy_photo1.jpg",
    5: "noisy_photo2.jpg"

}

text_read_switcher = {
    0: "Color read",
    1: "Gray scale read",
    2: "Original read"
}

text_function_switcher = {
    0: "Color to GrayScale",
    1: "Color to GrayScale 2",
    2: "Negative filter",
    3: "Equalize Hist filter",
    4: "CreateCLAHE filter",
    5: "Median filter normalized",
    6: "Bilateral filter normalized",
    7: "Gaussian filter normalized",
    8: "Mean filter normalized",
    9: "Threshold filter value",
    10: "Gamma correlation filter",
    11: "Adding noise randomly",
    12: "historic cdf normalized",
    13: "Calculate historic"
}

text_error_switcher = {
    0: "Mean square error",
    1: "Signal to noise ratio error",
    2: "Similarity measure based on histogram"
}

show_switcher(text_photo_switcher)
number_img_text = input_read("Select the photo", 0, 5)
img_select = text_photo_switcher.get(number_img_text)
print()

img_switcher = {
    0: color_read(img_select),
    1: gray_scale_read(img_select),
    2: unchanged_read(img_select)
}

show_switcher(text_read_switcher)
number_read_text = input_read("Select the type of read", 0, 2)
read_text = text_read_switcher.get(number_read_text)
img_read_select = img_switcher.get(number_read_text)
print()

img_read_select_original = img_read_select.copy()
condition_main = True
while condition_main:
    show_switcher(text_function_switcher)
    number_function_text = input_read("Select the type of function", 0, 13)
    if number_function_text == 12:
        bool_plot1 = True
    if number_function_text == 13:
        bool_plot2 = True
    function_switcher = {
        0: color_to_gray_scale(img_read_select),
        1: color_to_gray_scale2(img_read_select),
        2: negative_filter(img_read_select),
        3: equalize_hist(img_read_select),
        4: contrast_filter(img_read_select),
        5: median_normalized_filter(img_read_select),
        6: bilateral_normalized_filter(img_read_select),
        7: gaussian_normalized_filter(img_read_select),
        8: mean_normalized_filter(img_read_select),
        9: threshold_filter(img_read_select),
        10: gamma_correlation_filter(img_read_select),
        11: add_noise_randomly(img_read_select),
        12: hist_cdf_normalized(img_read_select),
        13: hist_historic(img_read_select)
    }


    function_text = text_function_switcher.get(number_function_text)
    img_function_select = function_switcher.get(number_function_text)
    print()
    bool_plot1 = False
    bool_plot2 = False
    try:
        bool_show = input_read("Select 1 if you want to show the img, otherwise pulse 0", 0, 1)
        if bool_show:
            limit_of_function_no_hist = 11
            if 0 <= number_function_text <= limit_of_function_no_hist:
                show_write_photo(img_select, img_function_select)
            else:
                show_hist()

        bool_error = input_read("Select 1 if you want to show the error of the img, otherwise pulse 0", 0, 1)
        if bool_error:
            error_switcher = {
                0: mean_square_error(img_read_select_original, img_function_select),
                1: signal_to_noise_ratio_error(img_read_select_original, img_function_select),
                2: similarity_measure_based_on_histogram(img_read_select_original, img_function_select)
            }
            show_switcher(text_error_switcher)
            number_error = input_read("Select the type of error", 0, 2)
            print("Error output of error type {} is {}: ".format(number_error, error_switcher.get(number_error)))
    except cv2.error:
        print("you selected a function not available")
        print()
    condition_main = input_read("Select 1 if you want to do another operation, otherwise pulse 0", 0, 1)
    img_read_select = img_function_select.copy()
