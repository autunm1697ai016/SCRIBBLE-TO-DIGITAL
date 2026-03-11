from utils import enhance_image, extract_text

class OCRService:
    def __init__(self, enhancement_mode="balanced"):
        self.enhancement_mode = enhancement_mode
    
    def preprocess(self, image_array):
        """Preprocess the image for better OCR results"""
        return enhance_image(image_array, self.enhancement_mode)
    
    def extract_text(self, image_array):
        """Extract text from the image"""
        return extract_text(image_array)