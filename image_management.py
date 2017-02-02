import cv2
import numpy as np
from matplotlib import pyplot as plt
import time as tm
import math as ma

windows_name = "Photos"


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
    smooth = m * n
    for c1 in range(m):
        for c2 in range(n):
            err += img[c1][c2] ** 2
            err_den +=(other_img[c1][c2] - img[c1][c2]) ** 2
    return err / err_den


switcher = {
    0: "poor_illumination1.jpg",
    1: "poor_illumination2.jpg",
    2: "poor_illumination3.jpg",
    3: "noisy_photo1.jpg",
    4: "noisy_photo2.jpg",
    5: "noisy_photo3.jpg"
}

it = []
ft = []
na = []
ejt = []
m_e = []
s_e = []
show = False
loop = 2
for s in range(loop):
    init_time = []
    final_time = []
    names = []
    execution_time = []
    mean_error = []
    signal_error = []

    image_name = switcher.get(s, "Nothing")

    text = "Color read"
    names.append(text)
    init_time.append(tm.time())
    img1 = cv2.imread(image_name, cv2.IMREAD_COLOR)
    final_time.append(tm.time())
    if show:
        img_copy = img1.copy()
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(img1, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, img1)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
        img1 = img_copy.copy()
    else:
        mean_error.append(0.0)
        signal_error.append(0.0)
    cv2.imwrite("./images/{}{}.jpg".format(text, s), img1)

    text = "Black and white read"
    names.append(text)
    init_time.append(tm.time())
    img2 = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    final_time.append(tm.time())
    if show:
        img_copy = img2.copy()
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(img2, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, img2)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
        img2 = img_copy.copy()
    else:
        mean_error.append(0.0)
        signal_error.append(0.0)
    cv2.imwrite("./images/{}{}.jpg".format(text, s), img2)


    text = "Original read"
    names.append(text)
    init_time.append(tm.time())
    img3 = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    final_time.append(tm.time())
    if show:
        img_copy = img3.copy()
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(img3, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, img3)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
        img3 = img_copy.copy()
    else:
        mean_error.append(0.0)
        signal_error.append(0.0)
    cv2.imwrite("./images/{}{}.jpg".format(text, s), img3)


    text = "Color to black and white"
    names.append(text)
    init_time.append(tm.time())
    aux = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)


    text = "Color to HLS"
    names.append(text)
    init_time.append(tm.time())
    aux = cv2.cvtColor(img1, cv2.COLOR_BGR2HLS)
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)

    text = "Median filter normalized"
    names.append(text)
    init_time.append(tm.time())
    aux = cv2.medianBlur(img3, 5)
    final_time.append(tm.time())
    if show:
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)


    text = "Bilateral filter normalized"
    names.append(text)
    init_time.append(tm.time())
    aux = cv2.bilateralFilter(img3, 9, 75, 75)
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)

    text = "Gaussian filter normalized"
    names.append(text)
    init_time.append(tm.time())
    aux = cv2.GaussianBlur(img3, (5, 5), 0)
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)


    text = "Mean filter normalized"
    names.append(text)
    init_time.append(tm.time())
    aux = cv2.blur(img3, (5, 5))
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)

    text = "Negative filter"
    names.append(text)
    init_time.append(tm.time())
    aux = cv2.bitwise_not(img1)
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)

    text = "Filling border"
    names.append(text)
    init_time.append(tm.time())
    row, col = img3.shape[:2]
    bottom = img3[row - 2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    border_size = 30
    aux = cv2.copyMakeBorder(img3,
                             top=border_size,
                             bottom=border_size,
                             left=border_size,
                             right=border_size,
                             borderType=cv2.BORDER_CONSTANT,
                             value=[mean, mean, mean])
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)

    text = "Detecting edges"
    names.append(text)
    init_time.append(tm.time())
    fast = cv2.FastFeatureDetector_create(0)
    kp = fast.detect(img3, None)
    aux = img3.copy()
    cv2.drawKeypoints(img3, kp, aux, color=(255, 0, 0))
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)

    text = "Threshold filter medium value"
    names.append(text)
    init_time.append(tm.time())
    aux = cv2.threshold(img1, 100.0, 255.0, cv2.THRESH_TRUNC)[1]
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)

    text = "Threshold filter high value"
    names.append(text)
    init_time.append(tm.time())
    aux = cv2.threshold(img1, 200.0, 255.0, cv2.THRESH_TRUNC)[1]
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)

    text = "Threshold filter low value"
    names.append(text)
    init_time.append(tm.time())
    aux = cv2.threshold(img1, 50.0, 255.0, cv2.THRESH_BINARY)[1]
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)

    # http://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-openCV-python/33333692
    text = "Gamma correlation filter"
    names.append(text)
    init_time.append(tm.time())
    gamma = 2.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255

                      for i in np.arange(0, 256)]).astype("uint8")
    aux = cv2.LUT(img1, table)
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)

    text = "Adding noise randomly"
    names.append(text)
    init_time.append(tm.time())
    aux = img2.copy()
    cv2.randn(aux, aux.mean(), aux.std() / 5)
    cv2.add(img2, aux, aux, mask=None)
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)

    text = "Equalize Hist filter"
    names.append(text)
    init_time.append(tm.time())
    equ = cv2.equalizeHist(img2)
    aux = np.hstack((img2, equ))  # stacking images side-by-side
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)

    text = "CreateCLAHE filter"
    names.append(text)
    init_time.append(tm.time())
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    aux = contrast.apply(img2)
    final_time.append(tm.time())
    if show:
        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.putText(aux, text, (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows_name, aux)
        cv2.waitKey(0)
        mean_error.append(0.0)
        signal_error.append(0.0)
    else:
        mean_error.append(mean_square_error(img1, aux))
        signal_error.append(signal_to_noise_ratio_error(img1, aux))
    cv2.imwrite("./images/{}{}.jpg".format(text, s), aux)

    for j in range(len(init_time)):
        execution_time.append((final_time[j] - init_time[j]) * 1000)
    it.append(init_time)
    ft.append(final_time)
    na.append(names)
    ejt.append(execution_time)
    m_e.append(mean_error)
    s_e.append(signal_error)
    if show:
        # http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
        hist, bins = np.histogram(img3.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        plt.plot(cdf_normalized, color='b')
        plt.hist(img3.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.title(windows_name)
        plt.show()
        plt.close()

        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            historic = cv2.calcHist([img1], [i], None, [256], [0, 256])
            plt.plot(historic, color=col)
            plt.xlim([0, 256])
        plt.legend(('b', 'g', 'r', 'histogram'), loc='upper left')
        plt.title(windows_name)
        plt.show()
        plt.close()

infile = open("./images/Execution time.txt", "w+")
for j in range(loop):
    print("Photo: {}".format(j))
    infile.write("Photo: {}\n".format(j))
    for i in range(len(ejt[0])):
        print("Execution time{}: {} ==> {}ms".format(j, na[j][i], ejt[j][i]))
        print("Mean square error{}: {} ==> {}".format(j, na[j][i], m_e[j][i]))
        print("Signal to noise ratio error{}: {} ==> {}".format(j, na[j][i], s_e[j][i]))
        infile.write("Execution time{}: {} ==> {}ms\n".format(j, na[j][i], ejt[j][i]))
        infile.write("Mean square error{}: {} ==> {}\n".format(j, na[j][i], m_e[j][i]))
        infile.write("Signal to noise ratio error{}: {} ==> {}\n".format(j, na[j][i], s_e[j][i]))
        infile.write("\n")
    infile.write("\n")
    plt.plot(ejt[j])
    plt.xlim(0, len(it[j]) - 1)
infile.close()
plt.legend(('a', 'b', 'c', 'd', 'e', 'f', 'histogram'), loc='upper left')
plt.title(windows_name)
plt.show()
plt.close()
cv2.destroyAllWindows()
