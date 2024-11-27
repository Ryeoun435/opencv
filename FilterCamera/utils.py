import cv2
import numpy as np
from overlays import draw_ears_with_nose,  apply_fisheye_filter
from loadImages import cat_ears_image, cat_nose_image, bear_ears_image, bear_nose_image


# 공통 함수들
def create_mask(overlay):
    """Create a binary mask from the overlay image."""
    gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_overlay, 240, 255, cv2.THRESH_BINARY_INV)
    return mask


def resize_overlay(face_width, overlay):
    """Resize the overlay image proportionally to the face width."""
    scale_factor = face_width / overlay.shape[1]
    new_size = (int(overlay.shape[1] * scale_factor), int(overlay.shape[0] * scale_factor))
    return cv2.resize(overlay, new_size, interpolation=cv2.INTER_AREA)


def overlay_image(img, overlay, x, y, mask):
    """Overlay an image onto the main image at the specified position."""
    h, w = overlay.shape[:2]
    if x + w > img.shape[1] or y + h > img.shape[0] or x < 0 or y < 0:
        return img
    roi = img[y:y+h, x:x+w]
    img[y:y+h, x:x+w] = np.where(mask[:, :, np.newaxis] == 0, roi, overlay)
    return img


# 필터 모드별 함수
def apply_ears(image, overlay, landmarks):
    """Apply ear overlays (e.g., cat or dog ears)."""
    return draw_ears_with_nose(image, overlay, landmarks)


def apply_fisheye(image, face, face_width):
    """Apply a fisheye filter effect centered on the face."""
    center = (face.left() + face_width // 2, face.top() + face_width // 2)
    radius = face_width // 2
    return apply_fisheye_filter(image, center, radius)


def apply_overlay(image, overlay, face, face_width):
    """Apply an image overlay (e.g., filter icons) on the face."""
    resized_overlay = resize_overlay(face_width, overlay)
    mask = create_mask(resized_overlay)

    # 위치 조정
    x = face.right() - 25
    y = face.top() - resized_overlay.shape[0]
    x = min(max(x, 0), image.shape[1] - resized_overlay.shape[1])
    y = max(y, 0)

    return overlay_image(image, resized_overlay, x, y, mask)


# 필터 모드 메인 함수
def apply_filter_mode(image, mode, current_overlay, face, landmarks, face_width):
    """Apply filters and overlays based on the selected mode."""
    if mode in [2, 3, 4, 5]:  # Cat, Dog, Speech Bubble, Fisheye
        if mode == 2:
            image = draw_ears_with_nose(image, cat_ears_image, cat_nose_image, landmarks, face)
        elif mode == 3:
            image = draw_ears_with_nose(image, bear_ears_image, bear_nose_image, landmarks, face)
        elif mode == 4:
            image = apply_fisheye(image, face, face_width)
    elif mode in [8, 9]:
         if current_overlay is not None:
            image = apply_overlay(image, current_overlay, face, face_width)
    return image