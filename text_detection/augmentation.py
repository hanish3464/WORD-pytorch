import numpy as np
import cv2
import random
import opt


class Data_Augmentation(object):

    def __init__(self, image, region_score_GT, affinity_score_GT, confidence):
        self.image = image
        self.region_score_GT = region_score_GT
        self.affinity_score_GT = affinity_score_GT
        self.confidence = confidence

    def select_augmentation_method(self):
        if random.random() >= 0.5 and opt.flip: self.flip()
        if random.random() >= 0.5 and opt.crop: self.crop() #cropping is some bugs.
        if random.random() >= 0.5 and opt.rotate: self.rotate()
        return self.image, self.region_score_GT, self.affinity_score_GT, self.confidence

    def flip(self):
        opt = random.choice(['left-right', 'top-bottom'])
        image = self.image[:, ::-1, ::-1] if opt == 'left-right' else self.image[::-1, :, ::-1]
        self.image = np.ascontiguousarray(image, dtype=np.uint8)

        region_score_GT = self.region_score_GT[:, ::-1] if opt == 'left-right' else self.region_score_GT[::-1, :]
        self.region_score_GT = np.ascontiguousarray(region_score_GT, dtype=np.uint8)

        affinity_score_GT = self.affinity_score_GT[:, ::-1] if opt == 'left-right' else self.affinity_score_GT[::-1, :]
        self.affinity_score_GT = np.ascontiguousarray(affinity_score_GT, dtype=np.uint8)

        confidence = self.confidence[:, ::-1] if opt == 'left-right' else self.confidence[::-1, :]
        self.confidence = np.ascontiguousarray(confidence, dtype=np.uint8)

    def crop(self):
        opt = random.randint(0, 4)
        image_h, image_w, _ = self.image.shape
        region_h, region_w = self.region_score_GT.shape
        affinity_h, affinity_w = self.affinity_score_GT.shape
        confidence_h, confidence_w = self.confidence.shape

        new_image_h, new_image_w = image_h // 2, image_w // 2
        new_region_h, new_region_w = region_h // 2, region_w // 2
        new_affinity_h, new_affinity_w = affinity_h // 2, affinity_w // 2
        new_confidence_h, new_confidence_w = confidence_h // 2, confidence_w // 2
        crop_image = self.image.copy()
        crop_region = self.region_score_GT.copy()
        crop_affinity = self.affinity_score_GT.copy()
        crop_confidence = self.confidence.copy()

        if opt == 0:  # left-top cropping
            self.image = crop_image[:new_image_h, :new_image_w, :]
            self.region_score_GT = crop_region[:new_region_h, :new_region_w]
            self.affinity_score_GT = crop_affinity[:new_affinity_h, :new_affinity_w]
            self.confidence = crop_confidence[:new_confidence_h, :new_confidence_w]

        elif opt == 1:  # right-top cropping
            self.image = crop_image[:new_image_h, new_image_w:, :]
            self.region_score_GT = crop_region[:new_region_h, new_region_w:]
            self.affinity_score_GT = crop_affinity[:new_affinity_h, new_affinity_w:]
            self.confidence = crop_confidence[:new_confidence_h, new_confidence_w:]

        elif opt == 2:  # left-bottom cropping
            self.image = crop_image[new_image_h:, :new_image_w, :]
            self.region_score_GT = crop_region[new_region_h:, :new_region_w]
            self.affinity_score_GT = crop_affinity[new_affinity_h:, :new_affinity_w]
            self.confidence = crop_confidence[new_confidence_h:, :new_confidence_w]

        elif opt == 3:  # right-bottom cropping
            self.image = crop_image[new_image_h:, new_image_w:, :]
            self.region_score_GT = crop_region[new_region_h:, new_region_w:]
            self.affinity_score_GT = crop_affinity[new_affinity_h:, new_affinity_w:]
            self.confidence = crop_confidence[:new_confidence_h:, :new_confidence_w:]

        elif opt == 4:  # center cropping
            half_n_image_h, half_n_image_w = new_image_h // 2, new_image_w // 2
            self.image = crop_image[half_n_image_h:new_image_h + half_n_image_h, half_n_image_w:new_image_w + half_n_image_w, :]

            half_n_region_h, half_n_region_w = new_region_h // 2, new_region_w // 2
            self.region_score_GT = crop_region[half_n_region_h:new_region_h + half_n_region_h,
                                   half_n_region_w:new_region_w + half_n_region_w]

            half_n_affinity_h, half_n_affinity_w = new_affinity_h // 2, new_affinity_w // 2
            self.affinity_score_GT = crop_affinity[half_n_affinity_h:new_affinity_h + half_n_affinity_h,
                                   half_n_affinity_w:new_affinity_w + half_n_affinity_w]

            half_n_confidence_h, half_n_confidence_w = new_confidence_h // 2, new_affinity_w // 2
            self.confidence = crop_affinity[half_n_confidence_h:new_confidence_h + half_n_confidence_h,
                                   half_n_confidence_w:new_confidence_w + half_n_confidence_w]

    @staticmethod
    def rotate_method(img, angle):
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_W = int((h * sin) + (w * cos))
        new_H = int((h * cos) + (w * sin))

        M[0, 2] += (new_W / 2) - center_x
        M[1, 2] += (new_H / 2) - center_y

        return cv2.warpAffine(img, M, (new_W, new_H))

    def rotate(self):
        items = [self.image, self.region_score_GT, self.affinity_score_GT, self.confidence]
        angle = random.randint(0, 11) * 30
        store = []
        for item in items:
            item = self.rotate_method(item, angle)
            store.append(item)

