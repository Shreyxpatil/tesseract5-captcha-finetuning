import pytesseract
import cv2

# Ensure your model is in the current directory or provide the tessdata-dir path
config = '--tessdata-dir model_output -l captcha_final --psm 13'


image = cv2.imread('/home/ca/Projects/captcha_model/image.png')
# (Apply binarization/padding preprocessing here)
text = pytesseract.image_to_string(image, config=config)
print(f"Prediction: {text}")