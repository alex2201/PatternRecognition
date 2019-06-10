from copy import deepcopy
from queue import Queue
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

'''
    PROCCESSING: tornillos...
        [ 2.47458874e-03  6.31686186e-06  2.93298365e-10  2.38687478e-10  3.61293606e-19  8.74767639e-13  2.05674831e-22]
        
    PROCCESSING: rondanas...
        [ 8.64076791e-04  3.66039137e-07  1.18592534e-10  2.93904353e-11  5.99083568e-21 -3.00451403e-15 -1.39047102e-20]
        
    PROCCESSING: tuercas...
        [ 6.91657581e-04  2.72502904e-09  4.08372179e-13  1.98929007e-14  4.81857767e-29 -8.61217988e-19  5.30228787e-28]
'''


class ROIObjectClass:
    def __init__(self, name: str, members: list):
        self.name = name
        self.vectors = np.array([roi.attribute_vector for roi in members])
        self.centroid = sum(self.vectors) / self.vectors.shape[0]

    def euclidean(self, roi: 'ImgROI'):
        vector = roi.attribute_vector
        distance = LA.norm(self.centroid - vector)

        return distance

    def mahalanobis(self, roi: 'ImgROI'):
        vector = roi.attribute_vector
        d = self.centroid - vector
        v = np.array([vector - self.centroid for v in self.vectors])
        sigma = 1 / len(self.vectors) * np.transpose(v).dot(v)
        distance = np.sqrt(np.transpose(d).dot(LA.inv(sigma).dot(d)))

        return distance


class ImgROI:
    def __init__(self, coords: np.array, mask: np.array, img: np.array, label: int):
        self.x0, self.x1 = coords[0, :]
        self.y0, self.y1 = coords[1, :] + 1
        mask = mask.astype('uint8')
        masked = cv2.bitwise_and(img, img, mask=mask)
        self.img_segment = masked[self.x0:self.x1, self.y0:self.y1]
        self.label = label
        self.__calculate_attributes_vector()

    def __calculate_attributes_vector(self):
        moments = cv2.moments(self.img_segment)
        # self.centroid = ((moments["m10"] / moments["m00"]), (moments["m01"] / moments["m00"]))
        # print(self.centroid)
        hu = cv2.HuMoments(moments, 4).flatten()
        self.attribute_vector = hu  # np.take(hu, [0, 1, 3, 6])

    @property
    def coords(self):
        return np.array(((self.x0, self.x1), (self.y0, self.y1)))


def mark_label(img: np.array, shape: list, labeled_img: np.array, pos: list, n: int):
    segment_range = np.array(([pos[0], pos[0]], [pos[1], pos[1]]))  # (x0, x1), (y0, y1)

    # init bfs
    q = Queue()
    q.put(pos)

    while not q.empty():
        i, j = q.get()
        for r, c in [(i + x, j + y) for x in range(-1, 2) for y in range(-1, 2)]:
            if 0 <= r < shape[0] and 0 <= c < shape[1] and img[r, c] == 255 and labeled_img[r, c] == 0:
                labeled_img[r, c] = n

                if r < segment_range[0, 0]:
                    segment_range[0][0] = r

                if r > segment_range[0, 1]:
                    segment_range[0, 1] = r

                if c < segment_range[1, 0]:
                    segment_range[1, 0] = c

                if c > segment_range[1, 1]:
                    segment_range[1, 1] = c

                q.put((r, c))

    mask = deepcopy(labeled_img)
    mask[mask != n] = 0

    return ImgROI(segment_range, mask, deepcopy(img), n)


def label_objects(img: np.array):
    obj_count = 1
    shape = img.shape
    labeled_img = np.zeros(shape, int)
    rois = []

    for i in range(shape[0]):
        for j in range(shape[1]):
            pixel = img[i, j]
            lbl = labeled_img[i, j]

            if pixel == 255 and lbl == 0:
                roi = mark_label(img, shape, labeled_img, [i, j], obj_count)
                area = (roi.coords[0, 0] - roi.coords[0, 1]) * (roi.coords[1, 0] - roi.coords[1, 1])
                if area > 1000:
                    rois.append(roi)

                obj_count += 1

    return rois


def cam():
    cam = cv2.VideoCapture(1)
    cv2.namedWindow('original')
    cv2.namedWindow('thr')

    while True:
        ret_val, img = cam.read()
        img = cv2.flip(img, 1)
        GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        BLUR = cv2.bilateralFilter(GRAY, 9, 50, 50)
        otsu, THR = cv2.threshold(BLUR, 0, 255, cv2.THRESH_OTSU)
        cv2.imshow('original', img)
        cv2.imshow('thr', THR)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def image_process(img_path, plot=0):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    BLUR = cv2.bilateralFilter(img, 11, 0, 300)
    kernel = np.ones((15, 15), np.uint8)
    MORPH = cv2.morphologyEx(BLUR, cv2.MORPH_DILATE, kernel)
    ret, THR = cv2.threshold(MORPH, 240, 255, cv2.THRESH_BINARY)

    # fig = plt.figure()
    # ax = fig.add_subplot(311)
    # ax.imshow(BLUR, cmap='gray')
    # ax = fig.add_subplot(312)
    # ax.imshow(THR, cmap='gray')
    # ax = fig.add_subplot(313)
    # ax.hist(BLUR.ravel())
    # print(sorted(BLUR.ravel(), reverse=True)[:10])
    #
    # plt.show()
    #
    # return

    rois = label_objects(THR)

    if plot == 1:
        for i, roi in enumerate(rois):
            coords = roi.coords
            distances = [c.euclidean(roi) for c in obj_classes]
            belongs = min(distances)
            index = distances.index(belongs)
            label = obj_classes[index].name
            cv2.putText(img, f'{label}', (coords[1][0], coords[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.imshow(roi.img_segment, cmap='gray')

        fig1 = plt.figure()
        fig1.suptitle(f'{img_path}: {len(rois)}', fontsize=8)
        # fig2 = plt.figure()
        #
        ax1 = fig1.add_subplot(211)
        ax2 = fig1.add_subplot(212)
        # ax_hist1 = fig2.add_subplot(211)
        # ax_hist2 = fig2.add_subplot(212)
        #
        ax1.imshow(img, cmap='gray')
        ax2.imshow(THR, cmap='gray')
        # ax_hist1.hist(img.ravel())
        # ax_hist2.hist(THR.ravel())
        plt.show()
        plt.clf()

    return rois


def train():
    global obj_classes
    global obj_types

    for img_class in obj_types:
        print(f'PROCCESSING: {img_class}...')

        img_paths = glob.glob(f'./resources/images/{img_class}/*.jpg')

        rois = []

        for img_path in img_paths:
            roi = image_process(img_path)[0]
            rois.append(roi)

        obj_class = ROIObjectClass(img_class, rois)
        obj_classes.append(obj_class)


def test_models():
    global obj_classes
    global obj_types

    l = len(obj_types)
    conf_matrix = np.zeros((l, l))
    count = []

    for i, img_class in enumerate(obj_types):
        img_paths = glob.glob(f'./resources/images/{img_class}/*.jpg')
        print(f'TESTING: {img_class}...')

        rois = []

        for img_path in img_paths:
            roi = image_process(img_path)[0]
            distances = [c.euclidean(roi) for c in obj_classes]
            belongs = min(distances)
            index = distances.index(belongs)
            label = obj_classes[index].name
            conf_matrix[i, index] += 1

        conf_matrix[i] = conf_matrix[i] / len(img_paths) * 100

        # distances = [c.mahalanobis(roi) for c in obj_classes]
        # belongs = min(distances)
        # index = distances.index(belongs)
        # label = obj_classes[index].name
        # print(f'MAHALANOBIS->{img_path} => {label}...')

    print(conf_matrix)
    efficiency = np.trace(conf_matrix) / l
    print(f'Efficiency: {efficiency}%')


obj_classes = []
obj_types = ['tornillos', 'rondanas', ]


def predict():
    img_paths = glob.glob(f'./resources/images/predict/*.jpg')

    for img_path in img_paths:
        print(f'PREDICTING: {img_path}...')
        image_process(img_path, 1)


if __name__ == '__main__':
    train()
    test_models()
    predict()
    # image_process(r"./resources/images/tornillos\WIN_20190606_10_42_47_Pro.jpg")
