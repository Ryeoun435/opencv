import cv2
import os

# 이미지 경로 설정
image_path = './images/'

# 이미지 파일들 로드
cat_ears_image = cv2.imread(os.path.join(image_path,'cat_ears.png'), cv2.IMREAD_UNCHANGED)
bear_ears_image = cv2.imread(os.path.join(image_path,'bear_ears.png'), cv2.IMREAD_UNCHANGED)
cat_nose_image = cv2.imread(os.path.join(image_path,'cat_nose.png'), cv2.IMREAD_UNCHANGED)
bear_nose_image = cv2.imread(os.path.join(image_path,'bear_nose.png'), cv2.IMREAD_UNCHANGED)
see_overlay = cv2.imread(os.path.join(image_path,'see.png'))
work_out_overlay = cv2.imread(os.path.join(image_path,'work.png'))

# 이미지 로드 확인
if cat_ears_image is None:
    print("Error loading cat ears image.")
else:
    print("Cat ears image loaded with shape:", cat_ears_image.shape)

if bear_ears_image is None:
    print("Error loading bear ears image.")
else:
    print("bear ears image loaded with shape:", bear_ears_image.shape)

if cat_nose_image is None:
    print("Error loading cat nose image.")
else:
    print("Cat nose image loaded with shape:", cat_nose_image.shape)

if bear_nose_image is None:
    print("Error loading bear nose image.")
else:
    print("bear nose image loaded with shape:", bear_nose_image.shape)

if see_overlay is None:
    print("Error loading bubble overlay image.")
else:
    print("See overlay image loaded with shape:", see_overlay.shape)

if work_out_overlay is None:
    print("Error loading gym overlay image.")
else:
    print("Work_out overlay image loaded with shape:", work_out_overlay.shape)
