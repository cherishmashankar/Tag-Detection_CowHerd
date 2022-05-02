
import numpy as np
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
from matplotlib import pyplot as plt


#Read, resize and increase the contrast of the image
image_final = cv2.imread('Photos/ROI_0.png')

#To increase the contrast of the image
image_final = cv2.detailEnhance(image_final, sigma_s=20, sigma_r=0.15)
cv2.imshow("enchan", image_final)


#Convert to Gray scale
gray = cv2.cvtColor(image_final, cv2.COLOR_BGR2GRAY) 
cv2.imshow("gray", gray)
cv2.imshow("gray1", gray)

#To improve the contrast of the image
gray = cv2.equalizeHist(gray)
cv2.imshow("gray2", gray)


#Custom_config selects only digits
custom_config = r'--oem 3 --psm 6 outputbase digits'
print(pytesseract.image_to_string(gray, config=custom_config))


cv2.waitKey(10000)
cv2.destroyAllWindows()