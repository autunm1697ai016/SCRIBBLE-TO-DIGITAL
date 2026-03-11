import cv2
import numpy as np
import easyocr
import os

# Initialize OCR reader (cached for performance)
reader = None

def get_ocr_reader():
    """Initialize and return the OCR reader"""
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

def enhance_image(image, enhancement_mode="balanced"):
    # Convert to grayscale - handle both RGB and BGR formats
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    if enhancement_mode == "none":
        return gray
    elif enhancement_mode == "binary":
        bright = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        thresh = cv2.threshold(bright, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh
    elif enhancement_mode == "balanced":
        bright = cv2.convertScaleAbs(gray, alpha=1.2, beta=15)
        thresh = cv2.threshold(bright, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh
    else:
        # Default to balanced
        bright = cv2.convertScaleAbs(gray, alpha=1.2, beta=15)
        thresh = cv2.threshold(bright, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh

def extract_text(image):
    """Extract text from image using EasyOCR"""
    reader = get_ocr_reader()
    results = reader.readtext(image)
    
    # Combine all detected text
    text = '\n'.join([result[1] for result in results])
    return text
