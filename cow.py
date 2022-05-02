import numpy as np
import cv2


#Read the image and resize it
#image = cv2.imread('Photos/cow_tag1.jfif',cv2.IMREAD_COLOR)
image = cv2.imread('Photos/tag3.jfif',cv2.IMREAD_COLOR)
#image = cv2.imread('Photos/tag6.jfif',cv2.IMREAD_COLOR)

#image = cv2.imread('Photos/tag4.jfif',cv2.IMREAD_COLOR)


#resize and copy
image = cv2.resize(image, (600,400) )
original = image.copy()
original1 = image.copy()

#Convert to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 


#Mask the yellow region in the image to get tag region
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([21, 90, 0], dtype="uint8")
upper = np.array([34, 255, 255], dtype="uint8")
mask = cv2.inRange(image, lower, upper)
cv2.imshow('mask', mask)

# Find contours, obtain bounding box, extract and save ROI
ROI_number = 0
cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

#To draw a rectangle and and save that image 
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(original1, (x, y), (x + w, y + h), (36,255,12), 2)
    ROI = original[y:y+h, x:x+w]
    #only image with size greater than 7000 is saved as a new image
    if(ROI.size > 7000):
        cv2.imwrite('Photos/roi/ROI_{}.png'.format(ROI_number), ROI)
        ROI_number += 1


cv2.imshow('image', original1)
ROI_number = ROI_number - 1
number = str(ROI_number)


image_number = "ROI_" + number + ".png"
print(image_number)


cv2.waitKey(20000)
cv2.destroyAllWindows()

