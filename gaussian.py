import cv2
import numpy as np
import math
import preprocess
import config
import os


class GenerateGaussian(object):

    def __init__(self, sigma, spread):
        print("Generate Gaussian")
        self.sigma = sigma
        self.spread = spread
        self.extent = int(spread * sigma)
        self.gaussian_template =  self._gaussian(self.extent, self.sigma, self.spread)

    @staticmethod
    def _gaussian(extent, sigma, spread):
        isotropicGaussian2dMap = np.zeros((2 * extent, 2 * extent), dtype=np.float32)
        for i in range(2 * extent):
            for j in range(2 * extent):
                isotropicGaussian2dMap[i, j] = float(1) / 2 / np.pi / (sigma ** 2) * np.exp(
                    -1 / 2 * ((i - spread * sigma - 0.5) ** 2 + (j - spread * sigma - 0.5) ** 2) / (sigma ** 2))
        isotropicGaussian2dMap = (isotropicGaussian2dMap / np.max(isotropicGaussian2dMap) * 255).astype(np.uint8)

        h, w = isotropicGaussian2dMap.shape

        adjust_gaussian_heat_map = np.zeros((h + 2, w + 2)).astype(np.uint8)
        adjust_gaussian_heat_map[:h, :w] = isotropicGaussian2dMap[:, :]
        adjust_gaussian_heat_map[:h, w] = isotropicGaussian2dMap[:, 1]
        adjust_gaussian_heat_map[:h, w + 1] = isotropicGaussian2dMap[:, 0]
        adjust_gaussian_heat_map[h + 1] = adjust_gaussian_heat_map[0]
        adjust_gaussian_heat_map[h] = adjust_gaussian_heat_map[1]
        print('_gaussian')
        return adjust_gaussian_heat_map

    @staticmethod
    def perspective_transform(gauss, box, flags = None):

        max_x, max_y = np.int32(math.ceil(np.max(box[:, 0]))), np.int32(math.ceil(np.max(box[:, 1])))
        if flags == 'affinity':
            x_center1, y_center1 = sum(box[:2,0])/float(2), sum(box[:2,1])/float(2)
            x_center2, y_center2 = sum(box[2:4,0])/float(2), sum(box[2:4,1])/float(2)
            affinity_tl = (box[0, 0] + x_center1) / float(2), (box[0, 1] + y_center1) / float(2)
            affinity_tr = (box[1, 0] + x_center1) / float(2), (box[1, 1] + y_center1) / float(2)
            affinity_br = (box[2, 0] + x_center2) / float(2), (box[2, 1] + y_center2) / float(2)
            affinity_bl = (box[3, 0] + x_center2) / float(2), (box[3, 1] + y_center2) / float(2)
            tr = np.array(affinity_tr) - np.array(affinity_tl)
            br = np.array(affinity_br) - np.array(affinity_tl)
            bl = np.array(affinity_bl) - np.array(affinity_tl)
            resize_affinity = np.array([[0,0], tr, br, bl], np.float32)
            box = resize_affinity[:]

        h, w = gauss.shape[:2]
        #gaussian region size -> character_box size
        gauss_region = np.array([[0, 0], [w - 1, 0], [h - 1, w - 1], [0, h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src= gauss_region, dst = box)
        warped = cv2.warpPerspective(gauss,  M, (max_x, max_y), borderValue = 0, borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite('./train/temp.jpg', warped)
        return warped

    @staticmethod
    def box_cover_check(char_box, word_box, word_box_len, flags=None):
        apex_lists = [[True, True], [False, True], [False, False], [True, False]]
        word_box, char_box, idx = np.array(word_box), np.array(char_box), 0
        check_bound_list = list()
        neg = config.neg_link_threshold
        pos = config.pos_link_threshold
        thresholds = [neg, pos]
        for i in range(word_box_len):
            for apex_list in apex_lists:
                for threshold in thresholds:
                    apex = ((word_box[i][idx] - char_box[idx] <= threshold) == apex_list).all()
                    if ((word_box[i][idx] - char_box[idx] == 0) == [True,
                                                                    True]).all():  # word boundary == char boundary
                        apex = True
                    if apex:
                        break
                check_bound_list.append(apex)
                idx += 1
            if np.array(check_bound_list).all():
                if flags == 'region': return np.append(word_box[i], char_box).tolist()
                if flags == 'affinity': return True
            check_bound_list[:] = list()
            idx = 0
        return False

    # def perspective_transform(self, gauss, box, i, flags = None):
    #
    #     if flags == 'text':
    #         max_x, max_y = np.int32(math.ceil(np.max(box[:, 0]))), np.int32(math.ceil(np.max(box[:, 1])))
    #     if flags == 'affinity':
    #         max_x, max_y = np.int32(math.ceil(np.max(box[:, 0]))), np.int32(math.ceil(np.max(box[:, 1])))
    #         x_center1, y_center1 = sum(box[:2,0])/float(2), sum(box[:2,1])/float(2)
    #         x_center2, y_center2 = sum(box[2:4,0])/float(2), sum(box[2:4,1])/float(2)
    #         affinity_tl = (box[0, 0] + x_center1) / float(2), (box[0, 1] + y_center1) / float(2)
    #         affinity_tr = (box[1, 0] + x_center1) / float(2), (box[1, 1] + y_center1) / float(2)
    #         affinity_br = (box[2, 0] + x_center2) / float(2), (box[2, 1] + y_center2) / float(2)
    #         affinity_bl = (box[3, 0] + x_center2) / float(2), (box[3, 1] + y_center2) / float(2)
    #         tr = np.array(affinity_tr) - np.array(affinity_tl)
    #         br = np.array(affinity_br) - np.array(affinity_tl)
    #         bl = np.array(affinity_bl) - np.array(affinity_tl)
    #         resize_affinity = np.array([[0,0], tr, br, bl], np.float32)
    #         box = resize_affinity[:]
    #
    #     h, w = gauss.shape[:2]
    #     #gaussian region size -> character_box size
    #     gauss_region = np.array([[0, 0], [w - 1, 0], [h - 1, w - 1], [0, h - 1]], dtype="float32")
    #     M = cv2.getPerspectiveTransform(src= gauss_region, dst = box)
    #     warped = cv2.warpPerspective(gauss,  M, (max_x, max_y), borderValue = 0, borderMode=cv2.BORDER_CONSTANT)
    #     return warped


    def add_gaussian_box(self, canvas, box, flags = None):
        box = np.array(box)
        if np.any(box < 0) or np.any(box[:, 0] > canvas.shape[1]) or np.any(box[:, 1] > canvas.shape[0]):
            return canvas

        top_left = np.array([np.min(box[:, 0]), np.min(box[:, 1])]).astype(np.int32)
        box -= top_left[None, :]
        warped = self.perspective_transform(self.gaussian_template, box.astype(np.float32), flags)

        start_row = max(top_left[1], 0) - top_left[1]
        start_col = max(top_left[0], 0) - top_left[0]
        end_row = min(top_left[1] + warped.shape[0], canvas.shape[0])  # H
        end_col = min(top_left[0] + warped.shape[1], canvas.shape[1])  # W

        canvas[max(top_left[1], 0):end_row, max(top_left[0], 0):end_col] += warped[start_row:end_row - top_left[1],
                                                                           start_col:end_col - top_left[0]]
        return canvas

    # def add_character(self, image, bbox, gaussian_heat_map, i, flags=None):
    #     bbox = np.array(bbox)
    #     if np.any(bbox < 0) or np.any(bbox[:, 0] > image.shape[1]) or np.any(bbox[:, 1] > image.shape[0]):
    #         return image
    #
    #     top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
    #     bbox -= top_left[None, :]
    #     transformed = self.perspective_transform(gaussian_heat_map, bbox.astype(np.float32), i, flags)
    #
    #     start_row = max(top_left[1], 0) - top_left[1]
    #     start_col = max(top_left[0], 0) - top_left[0]
    #     end_row = min(top_left[1] + transformed.shape[0], image.shape[0])  # H
    #     end_col = min(top_left[0] + transformed.shape[1], image.shape[1])  # W
    #
    #     image[max(top_left[1], 0):end_row, max(top_left[0], 0):end_col] += transformed[start_row:end_row - top_left[1],
    #                                                                        start_col:end_col - top_left[0]]
    #     return image


    # def text_box_valid_check(self, char_box, word_box):
    #
    #     filename, file_ext = os.path.splitext(os.path.basename(path))
    #     word_gt_path = './psd/word_ground_truth/' + filename + file_ext
    #     word_gt, word_gt_len = preprocess.loadText(word_gt_path)
    #     apex_lists = [[True, True], [False, True], [False, False], [True, False]]
    #     word_gt, char_box, idx = np.array(word_gt), np.array(char_box), 0
    #     check_bound_list = list()
    #     neg = config.neg_link_threshold
    #     pos = config.pos_link_threshold
    #     flags = [neg, pos]
    #     for i in range(word_gt_len):
    #         for apex_list in apex_lists:
    #             for flag in flags:
    #                 apex = ((word_gt[i][idx] - char_box[idx] <= flag) == apex_list).all()
    #                 if ((word_gt[i][idx] - char_box[idx] == 0) == [True,True]).all(): # word boundary == char boundary
    #                     apex = True
    #                 if apex:
    #                     break
    #             check_bound_list.append(apex)
    #             idx +=1
    #         if np.array(check_bound_list).all():
    #             return np.append(word_gt[i], char_box).tolist()
    #         check_bound_list[:] = list()
    #         idx = 0
    #
    #     return False


    def region(self, item, charBBox, wordBBox, charBBox_len, wordBBox_len):
        h, w, _ = item['image'].shape
        canvas = np.zeros([h, w], dtype=np.float32)
        box_in_word = list()
        for i in range(charBBox_len):
            box_in_word.append(self.box_cover_check(charBBox[i], wordBBox, wordBBox_len, flags = 'region')) #charbox : 1 , wordbox : all
            region_score_GT = self.add_gaussian_box(canvas, charBBox[i], flags='region').astype(float)
        preprocess.sort_charBBox_order(box_in_word, len(box_in_word), item['name'])
        cv2.imwrite('./gauss/region.jpg', region_score_GT)
        return region_score_GT

        # def region(self, img, gt, gt_len, path):
        #     h, w, _ = img.shape
        #     target = np.zeros([h, w], dtype=np.float32)
        #     box_in_word = list()
        #     for i in range(gt_len):
        #         box_in_word.append(self.text_box_valid_check(gt[i], path))
        #         target = self.add_character(target, gt[i], self.gaussian_template, i, flags='text').astype(float)
        #         # target_tmp = cv2.applyColorMap(target.astype(np.uint8), cv2.COLORMAP_JET)
        #     #     cv2.imwrite('./gauss/region_' + str(i) + '.jpg', target_tmp)
        #     # cv2.imwrite('./gauss/final_region_img' + str(k) + '.jpg', target_tmp)
        #     preprocess.sort_charBBox_order(box_in_word, len(box_in_word), path)

    # def affinity_box_cover_check(self, affinityBBox, wordBBox, flags = None):
    #     filename, file_ext = os.path.splitext(os.path.basename(path))
    #     word_gt_path = './psd/word_ground_truth/' + filename + file_ext
    #     word_gt, word_gt_len = preprocess.loadText(word_gt_path)
    #     apex_lists = [[True,True], [False,True], [False, False], [True,False]]
    #     word_gt, affinityBBox, idx = np.array(word_gt), np.array(affinityBBox), 0
    #     check_bound_list = list()
    #     neg = config.neg_link_threshold
    #     pos = config.pos_link_threshold
    #     flags =[neg, pos]
    #     for i in range(word_gt_len):
    #         for apex_list in apex_lists:
    #             for flag in flags:
    #                 apex = ((word_gt[i][idx] - box[idx] <= flag) == apex_list).all()
    #
    #                 if apex:
    #                     break
    #             check_bound_list.append(apex)
    #             idx +=1
    #         if np.array(check_bound_list).all(): return True
    #         check_bound_list[:] = list()
    #         idx = 0
    #     else: return False


    # def affinity_box_cover_check(self, box, path):
    #     filename, file_ext = os.path.splitext(os.path.basename(path))
    #     word_gt_path = './psd/word_ground_truth/' + filename + file_ext
    #     word_gt, word_gt_len = preprocess.loadText(word_gt_path)
    #     apex_lists = [[True, True], [False, True], [False, False], [True, False]]
    #     word_gt, box, idx = np.array(word_gt), np.array(box), 0
    #     check_bound_list = list()
    #     neg = config.neg_link_threshold
    #     pos = config.pos_link_threshold
    #     flags = [neg, pos]
    #     for i in range(word_gt_len):
    #         for apex_list in apex_lists:
    #             for flag in flags:
    #                 apex = ((word_gt[i][idx] - box[idx] <= flag) == apex_list).all()
    #
    #                 if apex:
    #                     break
    #             check_bound_list.append(apex)
    #             idx += 1
    #         if np.array(check_bound_list).all(): return True
    #         check_bound_list[:] = list()
    #         idx = 0
    #     else:
    #         return False

    def calculate_affinity_box(self, target, charBBox, next_charBBox, wordBBox, wordBBox_len):

        center_box, center_next_box = np.mean(charBBox, axis=0), np.mean(next_charBBox, axis=0)
        top_triangle_center_point = np.mean([charBBox[0], charBBox[1], center_box], axis=0)
        bot_triangle_center_point = np.mean([charBBox[2], charBBox[3], center_box], axis=0)
        top_triangle_next_center_point = np.mean([next_charBBox[0], next_charBBox[1], center_next_box], axis=0)
        bot_triangle_next_center_point = np.mean([next_charBBox[2], next_charBBox[3], center_next_box], axis=0)

        affinityBBox = np.array(
            [top_triangle_center_point, top_triangle_next_center_point, bot_triangle_next_center_point, bot_triangle_center_point])
        if self.box_cover_check(affinityBBox, wordBBox, wordBBox_len, flags= 'affinity'):
            return self.add_gaussian_box(target, affinityBBox, flags = 'affinity')
        else: return target

    def affinity(self, item, charBBox, wordBBox, charBBox_len, wordBBox_len):
        h, w, _ = item['image'].shape
        target = np.zeros([h, w], dtype=np.float32)

        # generate affinity_region_GT
        for i in range(charBBox_len-1):
            affinity_score_GT = self.calculate_affinity_box(target, charBBox[i], charBBox[i+1], wordBBox, wordBBox_len).astype(float)
        cv2.imwrite('./gauss/affinity.jpg', affinity_score_GT)
        return affinity_score_GT


    # def affinity(self, img, gt, gt_len, k, path):
    #     gaussian_heat_map = self.gaussian_template
    #     h, w, _ = img.shape
    #     target = np.zeros([h, w], dtype=np.float32)
    #
    #     # generate affinity_region_GT
    #     for i in range(gt_len-1):
    #         target = self.add_affinity(target, gt[i], gt[i+1], gaussian_heat_map, i, path).astype(float)
    #         target_tmp = cv2.applyColorMap(target.astype(np.uint8), cv2.COLORMAP_JET)
    #         #cv2.imwrite('./gauss/affinity_' + str(i) + '.jpg', target_tmp)
    #     cv2.imwrite('./gauss/final_affinity_img' + str(k) +'.jpg', target_tmp)

    # for i in range(1,8):
    #     temp_img = './psd/resized_jpg_images/' + 'tmp_' + str(i) + '.jpg'
    #     img = preprocess.loadImage(temp_img)
    #     temp_gt = './psd/char_ground_truth/' + 'tmp' + str(i) + '.txt'
    #     gt, gt_len = preprocess.loadText(temp_gt)
    #     generate_text_region(img, gt, gt_len, i, temp_gt)
    #     generate_affinity_region(img, gt, gt_len, i, temp_gt)
