from math import exp
import math
import cv2
import numpy as np
import imgproc
import config


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


if __name__ == '__main__':
    # image = cv2.imread('./train/images/05_3.png')
    # bboxes = [['a', [[118.17204301075265, 261.2903225806451], [152.04301075268813, 261.8279569892473],
    #                  [145.59139784946234, 300.0], [111.72043010752685, 298.1182795698925]]], ['a', [
    #     [152.04301075268813, 259.9462365591398], [180.53763440860212, 261.55913978494624],
    #     [174.89247311827955, 300.5376344086021], [145.59139784946234, 298.6559139784946]]], ['a', [
    #     [180.53763440860212, 262.63440860215053], [209.83870967741933, 261.8279569892473],
    #     [203.11827956989248, 301.34408602150535], [172.74193548387098, 296.505376344086]]], ['a', [
    #     [211.4516129032258, 261.8279569892473], [245.59139784946234, 262.9032258064516],
    #     [240.21505376344084, 301.0752688172043], [210.1075268817204, 300.5376344086021]]], ['a', [
    #     [243.97849462365588, 261.55913978494624], [273.01075268817203, 263.7096774193548],
    #     [267.09677419354836, 299.7311827956989], [239.67741935483866, 296.23655913978496]]], ['a', [
    #     [271.39784946236557, 258.06451612903226], [293.17204301075265, 262.36559139784947],
    #     [284.30107526881716, 299.7311827956989], [264.13978494623655, 294.6236559139785]]], ['b', [
    #     [455.26881720430106, 338.7096774193548], [497.4731182795698, 340.0537634408602],
    #     [492.0967741935483, 391.66666666666663], [450.43010752688167, 389.247311827957]]], ['b', [
    #     [453.1182795698925, 390.8602150537634], [498.54838709677415, 391.1290322580645],
    #     [490.752688172043, 440.05376344086017], [450.43010752688167, 438.44086021505376]]], ['c', [
    #     [460.10752688172045, 443.81720430107526], [476.505376344086, 445.1612903225806],
    #     [468.70967741935476, 487.3655913978494], [451.23655913978496, 486.02150537634407]]], ['c', [
    #     [476.23655913978496, 443.81720430107526], [496.9354838709677, 444.89247311827955],
    #     [488.8709677419355, 490.59139784946234], [467.0967741935483, 486.02150537634407]]]]
    #
    # gauss = GenerateGaussian(512, region_threshold=0.4, affinity_threshold=0.2)
    # region_score_GT = gauss.region(image, bboxes)
    # affinity_score_GT = gauss.affinity(image, bboxes)
    # region_score_GT = imgproc.cvt2HeatmapImg(region_score_GT / 255.0)
    # affinity_score_GT = imgproc.cvt2HeatmapImg(affinity_score_GT / 255.0)
    # cv2.imwrite('./region_GT.png', region_score_GT)
    # cv2.imwrite('./affinity_GT.png', affinity_score_GT)
    pass
