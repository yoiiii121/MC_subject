import cv2
import numpy as np
from matplotlib import pyplot as plt
import time as tm

windows_name = "Photos"

switcher = {
    0: "poor_illumination.jpg",
    1: "noisy_photo.jpg",
    2: "noisy_photo2.jpg"
}

it = []
ft = []
na = []
ejt = []
loop = 3
for s in range(loop):
    init_time = []
    final_time = []
    names = []
    execution_time = []
    image_name = switcher.get(s, "Nothing")

    names.append("Color read")
    init_time.append(tm.time())
    img1 = cv2.imread(image_name, cv2.IMREAD_COLOR)
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, img1)
    cv2.waitKey(0)

    names.append("Black and white read")
    init_time.append(tm.time())
    img2 = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, img2)
    cv2.waitKey(0)

    names.append("Original read")
    init_time.append(tm.time())
    img3 = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, img3)
    cv2.waitKey(0)

    names.append("Color to black and white")
    init_time.append(tm.time())
    aux = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)

    names.append("Color to HLS")
    init_time.append(tm.time())
    aux = cv2.cvtColor(img1, cv2.COLOR_BGR2HLS)
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)

    names.append("Median filter normalized")
    init_time.append(tm.time())
    aux = cv2.medianBlur(img3, 5)
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)

    names.append("Bilateral Filter filter normalized")
    init_time.append(tm.time())
    aux = cv2.bilateralFilter(img3, 9, 75, 75)
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)

    names.append("Gaussian filter normalized")
    init_time.append(tm.time())
    aux = cv2.GaussianBlur(img3, (5, 5), 0)
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)

    names.append("Mean filter normalized")
    init_time.append(tm.time())
    aux = cv2.blur(img3, (5, 5))
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)

    names.append("Negative filter")
    init_time.append(tm.time())
    aux = cv2.bitwise_not(img1)
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)

    names.append("Filling border")
    init_time.append(tm.time())
    row, col = img3.shape[:2]
    bottom = img3[row-2:row, 0:col]
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
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)

    names.append("Detecting edges")
    init_time.append(tm.time())
    fast = cv2.FastFeatureDetector_create(0)
    kp = fast.detect(img3, None)
    aux = img3.copy()
    cv2.drawKeypoints(img3, kp, aux, color=(255, 0, 0))
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)

    names.append("Threshold filter medium value")
    init_time.append(tm.time())
    aux = cv2.threshold(img1, 100.0, 255.0, cv2.THRESH_TRUNC)[1]
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)

    names.append("Threshold filter high value")
    init_time.append(tm.time())
    aux = cv2.threshold(img1, 200.0, 255.0, cv2.THRESH_TRUNC)[1]
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)

    names.append("Threshold filter low value")
    init_time.append(tm.time())
    aux = cv2.threshold(img1, 50.0, 255.0, cv2.THRESH_BINARY)[1]
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)

    names.append("Adding noise randomly")
    init_time.append(tm.time())
    aux = img2.copy()
    cv2.randn(aux, aux.mean(), aux.std()/5)
    cv2.add(img2, aux, aux, mask=None)
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)

    names.append("Equalize Hist filter")
    init_time.append(tm.time())
    equ = cv2.equalizeHist(img2)
    aux = np.hstack((img2, equ))  # stacking images side-by-side
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)

    names.append("CreateCLAHE filter")
    init_time.append(tm.time())
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    aux = contrast.apply(img2)
    final_time.append(tm.time())
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, aux)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    for j in range(len(init_time)):
        execution_time.append((final_time[j] - init_time[j]) * 1000)
    it.append(init_time)
    ft.append(final_time)
    na.append(names)
    ejt.append(execution_time)

for j in range(loop):
    print("Photo: {}".format(j))
    for i in range(len(ejt[0])):
        print("Execution time: {}ms ==> {}".format(ejt[j][i], na[j][i]))
    plt.plot(ejt[j])
    plt.xlim(0, len(it[j]) - 1)
plt.legend(('a', 'b', 'c', 'histogram'), loc='upper left')
plt.title(windows_name)
plt.show()
plt.close()