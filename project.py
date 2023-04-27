import matplotlib.pyplot as plt
from scipy import ndimage
import imutils
import cv2
import numpy as np
import csv
import math
import pytesseract
from pytesseract import Output
import re


def main():
    image = cv2.imread('noback.jpg')

    #///// Preprocessing Image /////

    # Source: https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | 
                                            cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)    
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)

    im2 = image.copy()
    file = open("recognized.txt", "w+")
    file.write("")
    file.close()

    print(len(contours))
    for cnt in contours:
        # Source: https://stackoverflow.com/questions/24385714/detect-text-region-in-image-using-opencv 
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)        
        cropped = thresh[y:y + h, x:x + w] # was im2
        file = open("recognized.txt", "a")
        text = pytesseract.image_to_string(cropped)

        ### get title
        '''d = pytesseract.image_to_data(cropped, output_type=Output.DICT)
        n_boxes = len(d['level'])
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('img', cropped)
        cv2.waitKey(0)'''




        newtext = ""
        for i in range(1, len(text)):
            if text[i] == '\n' and text[i - 1] == '\n':
                newtext += " "
            else:
                newtext += text[i]
        regions = re.split("\n{2}", text)
        print(regions)
        
        # then split string by whitespace, get rid of duplicates?
        # actually: just look for date and time, once found, stop searching
        
        file.write(text)
        file.write("\n")        
        file.close()            

main()
