# -*- coding: utf8
# import scipy.misc
import os

import scipy.misc
import numpy as np

__author__ = 'haizhu'

import face_recognition
from PIL import Image, ImageDraw
import math


def recognition_landmarks(image):
    image = scipy.misc.fromimage(image, flatten=False, mode='RGB')

    face_landmarks_list = face_recognition.face_landmarks(image)
    print(face_landmarks_list)
    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        facial_features = [
            'chin',
            'left_eyebrow',
            'right_eyebrow',
            'nose_bridge',
            'nose_tip',
            'left_eye',
            'right_eye',
            'top_lip',
            'bottom_lip'
        ]

        for facial_feature in facial_features:
            print("The {} in this face has the following points: {}".format(facial_feature,
                                                                            face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
        image = Image.fromarray(image)
        d = ImageDraw.Draw(image)

        for facial_feature in facial_features:
            d.line(face_landmarks[facial_feature], width=5)

    image.show()
    pass


def resizeFaceCenter(image):
    """
    把图像摆正
    :param image:
    :return:
    """
    image = scipy.misc.fromimage(image, flatten=False, mode='RGB')

    face_landmarks_list = face_recognition.face_landmarks(image)
    if len(face_landmarks_list) < 1:
        return None

    face_landmarks = face_landmarks_list[0]
    # for face_landmarks in face_landmarks_list:

    # Print the location of each facial feature in this image
    facial_features = [
        # 'chin',
        # 'left_eyebrow',
        # 'right_eyebrow',
        # 'nose_bridge',
        # 'nose_tip',
        'left_eye',
        'right_eye',
        # 'top_lip',
        # 'bottom_lip'
    ]

    # for facial_feature in facial_features:
    #     print("The {} in this face has the following points: {}".format(facial_feature,
    #                                                                     face_landmarks[facial_feature]))

    # Let's trace out each facial feature in the image with a line!



    left_eye = face_landmarks['left_eye']
    right_eye = face_landmarks['right_eye']
    image = Image.fromarray(image)
    # d = ImageDraw.Draw(image)
    # d.line(left_eye, width=5)
    # d.line(right_eye, width=5)

    # print('left_eye', left_eye)
    center_leye = np.sum(left_eye, axis=0) / len(left_eye)
    center_reye = np.sum(right_eye, axis=0) / len(right_eye)
    # print('center_leye', center_leye)
    # print('center_reye', center_reye)

    # center = np.add(center_reye, center_leye) / 2
    # print('center', center)

    cons = np.subtract(center_reye, center_leye)
    sin_value = cons[1] / cons[0]
    degree = math.degrees(math.atan(sin_value))

    image = image.rotate(degree).copy()
    # .save('/Users/haizhu/Desktop/jiemo/test_male/tmp/rotate.jpg')

    # image.show()
    return image
    pass


class Faces(object):
    def getFaceImage(self, name='', padding_rate=0.2):
        """
        :param name:
        :return: image,location (top, right, bottom, left)
        """
        # path = '/Users/haizhu/Desktop/jiemo/test/khKSUTBpHJsD8s41TXBPxw.jpg'
        im = Image.open(name)
        resize_img = resizeFaceCenter(im)
        if resize_img is not None:
            im = resize_img
        # image = face_recognition.load_image_file(path)
        image = scipy.misc.fromimage(im, flatten=False, mode='RGB')
        face_locations = face_recognition.face_locations(image)  # (top, right, bottom, left)

        if len(face_locations) == 1:
            loc = face_locations[0]
            padding = padding_rate * (loc[2] - loc[0])
            print('padding=', padding)
            # im = im.crop((loc[0],loc[1],loc[2]+loc[0],loc[3]+loc[1]))#(left, upper, right, lower)
            # (left, upper, right, lower)
            left = loc[3] - padding
            upper = loc[0] - padding
            right = loc[1] + padding
            lower = loc[2] + padding

            padding_y = 0
            padding_x = 0
            if right - left > lower - upper:
                padding_y = ((right - left) - (lower - upper)) / 2
            elif right - left < lower - upper:
                padding_x = ((lower - upper) - (right - left)) / 2

            width = im.size[0]
            height = im.size[1]
            im = im.crop((0 if left - padding_x < 0 else left - padding_x,  #
                          0 if upper - padding_y < 0 else (upper - padding_y),  #
                          width if right + padding_x > width else right + padding_x,  #
                          height if lower + padding_y > height else lower + padding_y))


        else:
            return None, None

        return im, face_locations[0]

    def getFaceLocations(self, name=''):
        """
        :param name:
        :return: 坐标数组，注意坐标点排布[(top, right, bottom, left),...]
        """
        # path = '/Users/haizhu/Desktop/jiemo/test/khKSUTBpHJsD8s41TXBPxw.jpg'
        im = Image.open(name)
        # image = face_recognition.load_image_file(path)
        image = scipy.misc.fromimage(im, flatten=False, mode='RGB')
        face_locations = face_recognition.face_locations(image)  # (top, right, bottom, left)

        return face_locations

    def getFaceImageInfo(self, dir):
        temDir = os.path.join(dir, 'tmp')

        if not os.path.exists(temDir):
            os.makedirs(temDir)
        images_map = {}
        for f in os.listdir(dir):
            if f.find('.jp') > 0:
                image_path = os.path.join(dir, f)
                faceImage, local = self.getFaceImage(image_path)
                if faceImage is None:
                    continue
                tmp_path = os.path.join(temDir, '%d_tmp_face.jpg' % (len(images_map)))
                faceImage.save(tmp_path)
                images_map[image_path] = {'tmp': tmp_path, 'loc': local}
                # break

        # if len(images_map) > 0:
        #
        return images_map

    def opsFaceImages(self, dir, temDir, size=256):
        """
        预处理图片，也就是我们收集到的图片
        :param dir:
        :return:
        """

        if not os.path.exists(temDir):
            os.makedirs(temDir)
        num = 0
        for f in os.listdir(dir):
            if f.find('.jp') > 0:
                image_path = os.path.join(dir, f)
                faceImage, local = self.getFaceImage(image_path)
                if faceImage is None:
                    continue
                tmp_path = os.path.join(temDir, f)
                num += 1
                print('save=', tmp_path, "  num=", num)
                faceImage.resize((size, size)).save(tmp_path)
                # if num > 2:
                #     break

                # if len(images_map) > 0:
                #


if __name__ == '__main__':
    faces = Faces()
    # dir = '/Users/haizhu/Desktop/jiemo/test'
    # dir = '/Users/haizhu/Desktop/jiemo/test_female'

    dir = '/Users/haizhu/Downloads/ml/drive-download-20170824T030713Z-001/part2'
    images_map = faces.opsFaceImages(dir, '/Users/haizhu/Downloads/ml/drive-download-20170824T030713Z-001/all', 256)
    # images = list(v['tmp'] for v in images_map.values())
