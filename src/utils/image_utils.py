import cv2
from src.config import config


def image_processing(image):
    image = median_blur(image)

    return image

def median_blur(image):
    return cv2.medianBlur(image, config.MEDIAN_BLUR_KERNEL_SIZE)

def get_gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)