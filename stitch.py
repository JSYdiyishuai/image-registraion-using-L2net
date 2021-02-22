# -*- coding: utf-8 -*-
from enum import Enum
from typing import List, Tuple, Union
from numpy import *
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from k_means import k_means, get_group_center
from ransac import *
from blend import *
from L2_Net import L2Net
import time

def show_image(image: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).show()


class Method(Enum):

    SURF = cv2.xfeatures2d.SURF_create
    SIFT = cv2.xfeatures2d.SIFT_create
    ORB = cv2.ORB_create


colors = ((123, 234, 12), (23, 44, 240), (224, 120, 34), (21, 234, 190),
          (80, 160, 200), (243, 12, 100), (25, 90, 12), (123, 10, 140))


class Area:

    def __init__(self, *points):

        self.points = list(points)

    def is_inside(self, x: Union[float, Tuple[float, float]], y: float=None):
        if isinstance(x, tuple):
            x, y = x
        raise NotImplementedError()


class Matcher:
    def __init__(self, image1: np.ndarray, image2: np.ndarray, method: Enum = Method.SIFT, threshold=800, ratio=400):
        """输入两幅图像，计算其特征值
        此类用于输入两幅图像，计算其特征值，输入两幅图像分别为numpy数组格式的图像，
        其中的method参数要求输入SURF、SIFT或者ORB，threshold参数为特征值检测所需的阈值。

        Args:
            image1 (np.ndarray): 图像一
            image2 (np.ndarray): 图像二
            method (Enum, optional): Defaults to Method.SIFT. 特征值检测方法
            ratio (int, optional): Defaults to 400. L2-Net特征向量比重
            threshold (int, optional): Defaults to 800. 特征值阈值

        """

        self.image1 = image1
        self.image2 = image2
        self.method = method
        self.ratio = ratio

        self.ros1 = []
        # type np.ndarray
        self.ros1_3 = None
        self.ros2 = []
        # type np.ndarray
        self.ros2_3 = None
        self.loc1 = []
        # type list
        self.loc2 = []
        # type list
        self.threshold = threshold

        self._keypoints1 = None
        # type List[cv2.KeyPoint]
        self._descriptors1 = None
        # type np.ndarray
        self._keypoints2 = None
        # type List[cv2.KeyPoint]
        self._descriptors2 = None
        # type np.ndarray

        if self.method == Method.ORB:
            # error if not set this
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            # self.matcher = cv2.FlannBasedMatcher()

        self.match_points = []
        self.match_points1 = []

        self.image_points1 = np.array([])
        self.image_points2 = np.array([])

        self.keypoints1_loc = None
        self.keypoints2_loc = None

        self.x = 0
        self.y = 0

        self.vggd1 = []
        self.vggd2 = []

        self.l2_1des = None
        self.l2_2des = None

        self.save1 = []
        self.save2 = []
        self.save3 = []

    def compute_keypoint(self) -> None:
        """计算特征点
        利用给出的特征值检测方法对图像进行特征值检测。

        Args:
            image (np.ndarray): 图像
        """
        print('Compute keypoint by SIFT')
        print(Method.SIFT)


        feature = self.method.value(self.threshold)
        self._keypoints1, self._descriptors1 = feature.detectAndCompute(
            self.image1, None)
        self._keypoints2, self._descriptors2 = feature.detectAndCompute(
            self.image2, None)
        self.keypoints1_loc = cv2.KeyPoint_convert(self._keypoints1)
        self.keypoints2_loc = cv2.KeyPoint_convert(self._keypoints2)

    def get_the_ros(self):
        """选取待匹配的特征块
                将灰度图像选取特定区域，并转换维度为(?, 32, 32, 1)
                """

        for i in range(len(self.keypoints1_loc)):
            self.loc1.append(cv2.KeyPoint(x=self.keypoints1_loc[i][0], y=self.keypoints1_loc[i][1], _size=32,
                                          _angle=-1, _response=0.018, _octave=1, _class_id=-1))
            if self.keypoints1_loc[i][0] < 16:
                self.x = 16
            elif self.keypoints1_loc[i][0] > 240:
                self.x = 240
            else:
                self.x = int(self.keypoints1_loc[i][0])

            if self.keypoints1_loc[i][1] < 16:
                self.y = 16
            elif self.keypoints1_loc[i][1] > 240:
                self.y = 240
            else:
                self.y = int(self.keypoints1_loc[i][1])
            self.ros1.append(self.image1[self.x-16: self.x+16, self.y-16: self.y+16])

        for i in range(len(self.keypoints2_loc)):
            self.loc2.append(cv2.KeyPoint(x=self.keypoints2_loc[i][0], y=self.keypoints2_loc[i][1], _size=32,
                                          _angle=-1, _response=0.018, _octave=1, _class_id=-1))
            if self.keypoints2_loc[i][0] < 16:
                self.x = 16
            elif self.keypoints2_loc[i][0] > 240:
                self.x = 240
            else:
                self.x = int(self.keypoints2_loc[i][0])

            if self.keypoints2_loc[i][1] < 16:
                self.y = 16
            elif self.keypoints2_loc[i][1] > 240:
                self.y = 240
            else:
                self.y = int(self.keypoints2_loc[i][1])
            self.ros2.append(self.image2[self.x-16: self.x+16, self.y-16: self.y+16])

        # self.ros1, self.loc1 = slice(self.image1)
        # self.ros2, self.loc2 = slice(self.image2)


    def compute_kepoint_by_L2_Net(self) -> None:
        """
        通过VGG计算描述向量+PCA
        筛选并存储描述向量和keypoint。

        """
        self.get_the_ros()
        self.ros1 = np.array(self.ros1)
        self.ros2 = np.array(self.ros2)

        print('Compute keypoint by L2-net')
        l2net = L2Net("L2Net-HP+")

        self.l2_1des = l2net.calc_descriptors(self.ros1)
        self.l2_2des = l2net.calc_descriptors(self.ros2)


    def match(self, max_match_lenth=200, threshold=0.04, show_match=False):
        """对两幅图片计算得出的特征值进行匹配，对ORB来说使用OpenCV的BFMatcher算法，而对于其他特征检测方法则使用FlannBasedMatcher算法。

            max_match_lenth (int, optional): Defaults to 20. 最大匹配点数量
            threshold (float, optional): Defaults to 0.04. 默认最大匹配距离差
            show_match (bool, optional): Defaults to False. 是否展示匹配结果
        """

        self.compute_keypoint()
        self.compute_kepoint_by_L2_Net()
        good = []

        '''计算两张图片中的配对点，并至多取其中最优的`max_match_lenth`个'''

        self.match_points = self.matcher.knnMatch(self._descriptors1, self._descriptors2, k=2)

        for m, n in self.match_points:

            if m.distance < 0.6 * n.distance:
                good.append(m)

        print(len(good))

        ptsA = np.float32([self._keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ptsB = np.float32([self._keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)

        matchesMask = status.ravel().tolist()

        h, w = 256, 256
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)

        img2 = cv2.polylines(self.image2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)
        img3 = cv2.drawMatches(self.image1, self._keypoints1, img2, self._keypoints2, good, None, **draw_params)
        show_image(img3)
        '''由最佳匹配取得匹配点对，并进行形变拼接'''


def get_weighted_points(image_points: np.ndarray):

    average = np.average(image_points, axis=0)

    max_index = np.argmax(np.linalg.norm((image_points - average), axis=1))
    return np.append(image_points, np.array([image_points[max_index]]), axis=0)


def pca(image, rate=128):
    mean_value = mean(image, axis=0)
    image = image-mean_value
    c = cov(image, rowvar=False)
    eigvalue, eigvector = linalg.eig(mat(c))

    index_vec = np.argsort(-eigvalue)
    n_largest_index = index_vec[:rate]
    t = eigvector[:, n_largest_index]
    print(np.shape(image), np.shape(t))
    new_image = np.dot(image, t)
    return new_image, t, mean_value


class Stitcher:

    def __init__(self, image1: np.ndarray, image2: np.ndarray, method: Enum = Method.SIFT, use_kmeans=False):
        """输入图像和匹配，对图像进行拼接
        目前采用简单矩阵匹配和平均值拼合

        Args:
            image1 (np.ndarray): 图像一
            image2 (np.ndarray): 图像二
            matcher (Matcher): 匹配结果
            use_kmeans (bool): 是否使用kmeans 优化点选择
        """
        self.image_points1, self.image_points2 = None, None
        self.image1 = image1
        self.image2 = image2
        self.method = method
        self.use_kmeans = use_kmeans
        self.matcher = Matcher(image1, image2, method=method)
        self.M = np.eye(3)

        self.image = None

    def stich(self, show_result=True, max_match_lenth=40, show_match_point=True, use_partial=False,
              use_new_match_method=False, use_gauss_blend=True):
        """对图片进行拼合

            show_result (bool, optional): Defaults to True. 是否展示拼合图像
            show_match_point (bool, optional): Defaults to True. 是否展示拼合点
        """
        self.matcher.match(max_match_lenth=max_match_lenth,
                           show_match=show_match_point)

        if self.use_kmeans:
            self.image_points1, self.image_points2 = get_group_center(
                self.matcher.image_points1, self.matcher.image_points2)
        else:
            self.image_points1, self.image_points2 = (
                self.matcher.image_points1, self.matcher.image_points2)

        if use_new_match_method:
            self.M = GeneticTransform(self.image_points1, self.image_points2).run()
        else:
            self.M, _ = cv2.findHomography(self.image_points1, self.image_points2, method=cv2.RANSAC)

        print("Good points and average distance: ", GeneticTransform.get_value(
            self.image_points1, self.image_points2, self.M))

        left, right, top, bottom = self.get_transformed_size()
        width = int(max(right, self.image2.shape[1]) - min(left, 0))
        height = int(max(bottom, self.image2.shape[0]) - min(top, 0))

        if use_partial:
            self.partial_transform()

        # 移动矩阵
        self.adjustM = np.array(
            [[1, 0, max(-left, 0)],  # 横向
             [0, 1, max(-top, 0)],  # 纵向
             [0, 0, 1]
             ], dtype=np.float64)
        self.M = np.dot(self.adjustM, self.M)
        transformed_1 = cv2.warpPerspective(
            self.image1, self.M, (width, height))
        transformed_2 = cv2.warpPerspective(
            self.image2, self.adjustM, (width, height))

        self.image = self.blend(transformed_1, transformed_2, use_gauss_blend=use_gauss_blend)

        if show_match_point:
            for point1, point2 in zip(self.image_points1, self.image_points2):
                point1 = self.get_transformed_position(tuple(point1))
                point1 = tuple(map(int, point1))
                point2 = self.get_transformed_position(tuple(point2), M=self.adjustM)
                point2 = tuple(map(int, point2))
                cv2.circle(self.image, point1, 10, (20, 20, 255), 5)
                cv2.circle(self.image, point2, 8, (20, 200, 20), 5)


    def blend(self, image1: np.ndarray, image2: np.ndarray, use_gauss_blend=True) -> np.ndarray:
        """对图像进行融合

        Args:
            image1 (np.ndarray): 图像一
            image2 (np.ndarray): 图像二
            use_gauss_blend
        Returns:
            np.ndarray: 融合结果
        """

        mask = self.generate_mask(image1, image2)
        print("Blending")
        if use_gauss_blend:
            result = gaussian_blend(image1, image2, mask, mask_blend=10)
        else:
            result = direct_blend(image1, image2, mask, mask_blend=0)

        return result

    def generate_mask(self, image1: np.ndarray, image2: np.ndarray):
        """生成供融合使用的遮罩，由变换后图像的垂直平分线来构成分界线

        Args:
            image1
            image2

        Returns:
            np.ndarray: 01数组
        """
        print("Generating mask")
        # x, y
        center1 = self.image1.shape[1] / 2, self.image1.shape[0] / 2
        center1 = self.get_transformed_position(center1)
        center2 = self.image2.shape[1] / 2, self.image2.shape[0] / 2
        center2 = self.get_transformed_position(center2, M=self.adjustM)
        x1, y1 = center1
        x2, y2 = center2

        def function(y, x, *z):
            return (y2 - y1) * y < -(x2 - x1) * (x - (x1 + x2) / 2) + (y2 - y1) * (y1 + y2) / 2

        mask = np.fromfunction(function, image1.shape)

        mask = np.logical_and(mask, np.logical_not(image2)) \
            + np.logical_and(mask, image1)\
            + np.logical_and(image1, np.logical_not(image2))

        return mask

    def get_transformed_size(self) -> Tuple[int, int, int, int]:
        """计算形变后的边界
        计算形变后的边界，从而对图片进行相应的位移，保证全部图像都出现在屏幕上。

        Returns:
            Tuple[int, int, int, int]: 分别为左右上下边界
        """

        conner_0 = (0, 0)
        conner_1 = (self.image1.shape[1], 0)
        conner_2 = (self.image1.shape[1], self.image1.shape[0])
        conner_3 = (0, self.image1.shape[0])
        points = [conner_0, conner_1, conner_2, conner_3]


        top = min(map(lambda x: self.get_transformed_position(x)[1], points))
        bottom = max(
            map(lambda x: self.get_transformed_position(x)[1], points))
        left = min(map(lambda x: self.get_transformed_position(x)[0], points))
        right = max(map(lambda x: self.get_transformed_position(x)[0], points))

        return left, right, top, bottom

    def get_transformed_position(self, x: Union[float, Tuple[float, float]], y: float = None, M = None) -> Tuple[float, float]:
        """求得某点在变换矩阵（self.M）下的新坐标

        Args:
            x (Union[float, Tuple[float, float]]): x坐标或(x,y)坐标
            y (float, optional): Defaults to None. y坐标，可无
            M (np.ndarray, optional): Defaults to None. 利用M进行坐标变换运算

        Returns:
            Tuple[float, float]:  新坐标
        """

        if isinstance(x, tuple):
            x, y = x
        p = np.array([x, y, 1])[np.newaxis].T
        if M is not None:
            M = M
        else:
            M = self.M
        pa = np.dot(M, p)
        return pa[0, 0] / pa[2, 0], pa[1, 0] / pa[2, 0]


if __name__ == "__main__":

    os.chdir(os.path.dirname(__file__))

    start_time = time.time()
    img1 = cv2.imread("./example/label.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("./example/4.jpg", cv2.IMREAD_GRAYSCALE)
    img1 = np.expand_dims(img1, axis=2)
    img2 = np.expand_dims(img2, axis=2)


    stitcher = Stitcher(img1, img2, Method.SIFT, False)
    stitcher.stich(max_match_lenth=50, use_partial=False, use_new_match_method=True, use_gauss_blend=False)

    cv2.imwrite('./example/merge.jpg', stitcher.image)

    print("Time: ", time.time() - start_time)
    print("M: ", stitcher.M)
