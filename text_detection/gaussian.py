"""gaussian.py"""

from math import exp
import cv2
import numpy as np


class GenerateGaussian(object):

    def __init__(self, image_size, region_threshold=0.4, affinity_threshold=0.2):

        self.distance_ratio = 3.34
        self.image_size = image_size
        self.gaussian_template = self._gaussian(image_size, self.distance_ratio)

        _, binary = cv2.threshold(self.gaussian_template, region_threshold * 255, 255, 0)
        np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.regionbox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        _, binary = cv2.threshold(self.gaussian_template, affinity_threshold * 255, 255, 0)
        np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.affinitybox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        self.oribox = np.array(
            [[0, 0, 1], [image_size - 1, 0, 1], [image_size - 1, image_size - 1, 1], [0, image_size - 1, 1]],
            dtype=np.int32)

    def _gaussian(self, image_size, distance_ratio):

        scaled_gaussian = lambda x: exp(-(1 / 2) * (x ** 2))
        heat = np.zeros((image_size, image_size), np.uint8)
        for i in range(image_size):
            for j in range(image_size):
                distance_from_center = np.linalg.norm(np.array([i - image_size / 2, j - image_size / 2]))
                distance_from_center = distance_ratio * distance_from_center / (image_size / 2)
                scaled_gaussian_prob = scaled_gaussian(distance_from_center)
                heat[i, j] = np.clip(scaled_gaussian_prob * 255, 0, 255)
        return heat

    def add_gaussian_box(self, image, target_bbox, regionbox=None, FLAG=None):

        target_bbox = np.array(target_bbox)

        if np.any(target_bbox < 0) or np.any(target_bbox[:, 0] > image.shape[1]) or np.any(
                target_bbox[:, 1] > image.shape[0]):
            return image

        affi = False
        if regionbox is None:
            regionbox = self.regionbox.copy()
        else:
            affi = True

        M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(target_bbox))
        size = self.image_size
        oribox = np.array([[[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]]], dtype=np.float32)

        real_target_box = cv2.perspectiveTransform(oribox, M)[0]

        real_target_box = np.int32(real_target_box)
        real_target_box[:, 0] = np.clip(real_target_box[:, 0], 0, image.shape[1])
        real_target_box[:, 1] = np.clip(real_target_box[:, 1], 0, image.shape[0])

        if np.any(target_bbox[0] < real_target_box[0]) or (
                target_bbox[3, 0] < real_target_box[3, 0] or target_bbox[3, 1] > real_target_box[3, 1]) or (
                target_bbox[1, 0] > real_target_box[1, 0] or target_bbox[1, 1] < real_target_box[1, 1]) or (
                target_bbox[2, 0] > real_target_box[2, 0] or target_bbox[2, 1] > real_target_box[2, 1]):

            warped = cv2.warpPerspective(self.gaussian_template.copy(), M, (image.shape[1], image.shape[0]))
            warped = np.array(warped, np.uint8)
            image = np.where(warped > image, warped, image)

        else:
            xmin = real_target_box[:, 0].min()
            xmax = real_target_box[:, 0].max()
            ymin = real_target_box[:, 1].min()
            ymax = real_target_box[:, 1].max()

            width = xmax - xmin
            height = ymax - ymin
            _target_box = target_bbox.copy()
            _target_box[:, 0] -= xmin
            _target_box[:, 1] -= ymin
            _M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))
            warped = cv2.warpPerspective(self.gaussian_template.copy(), _M, (width, height))
            warped = np.array(warped, np.uint8)
            if warped.shape[0] != (ymax - ymin) or warped.shape[1] != (xmax - xmin):
                print("region (%d:%d,%d:%d) warped shape (%d,%d)" % (
                    ymin, ymax, xmin, xmax, warped.shape[1], warped.shape[0]))
                return image

            image[ymin:ymax, xmin:xmax] = np.where(warped > image[ymin:ymax, xmin:xmax], warped,
                                                   image[ymin:ymax, xmin:xmax])
        return image

    def region(self, image, bboxes):
        h, w, _ = image.shape
        canvas = np.zeros([h, w], dtype=np.float32)
        for k, (labels_name, bbox) in enumerate(bboxes):
            region_score_GT = self.add_gaussian_box(canvas, bbox).astype(float)

        return region_score_GT

    def calculate_affinity_box(self, target, charBBox, next_charBBox, charBBox_class, next_charBBox_class):

        if charBBox_class == next_charBBox_class:
            center_box, center_next_box = np.mean(charBBox, axis=0), np.mean(next_charBBox, axis=0)
            top_triangle_center_point = np.mean([charBBox[0], charBBox[1], center_box], axis=0)
            bot_triangle_center_point = np.mean([charBBox[2], charBBox[3], center_box], axis=0)
            top_triangle_next_center_point = np.mean([next_charBBox[0], next_charBBox[1], center_next_box], axis=0)
            bot_triangle_next_center_point = np.mean([next_charBBox[2], next_charBBox[3], center_next_box], axis=0)
            affinityBBox = np.array(
                [top_triangle_center_point, top_triangle_next_center_point, bot_triangle_next_center_point,
                 bot_triangle_center_point])
            width = top_triangle_next_center_point[0] - top_triangle_center_point[0]
            if width <= 0: return target
            return self.add_gaussian_box(target, affinityBBox.copy(), self.affinitybox, FLAG='affinity')
        else:
            return target

    def affinity(self, image, bboxes):
        h, w, _ = image.shape
        target = np.zeros([h, w], dtype=np.float32)

        for k in range(len(bboxes)):
            if k + 1 == len(bboxes): break
            target = self.calculate_affinity_box(target, bboxes[k][1], bboxes[k + 1][1], bboxes[k][0],
                                                 bboxes[k + 1][0]).astype(float)
        affinity_score_GT = target
        return affinity_score_GT
