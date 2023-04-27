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


def clean_title(max_val):
    title_words = max_val.split()
    cleaned_title = []
    for word in title_words:
        hasNums = False
        for ch in word:
            if ch.isnumeric():
                hasNums = True
                break
        if not hasNums:
            cleaned_title.append(word)
    new_title = ' '.join(cleaned_title)
    return new_title

# Source: https://stackoverflow.com/questions/20831612/getting-the-bounding-box-of-the-recognized-words-using-python-tesseract/54059166#54059166
def get_sentence_info(cropped): 
    d = pytesseract.image_to_data(cropped, output_type=Output.DICT)
    n_boxes = len(d['level'])
    word_heights = {}
    words = {}
    for i in range(n_boxes):
        if d['text'][i] != '':
            word_heights[i] = d['height'][i]
            words[i] = d['text'][i]
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)
    avg_word_height = sum (word_heights.values()) / len(word_heights)
    sentence = ' '.join(words.values())
    return (avg_word_height, sentence)


def removeWall(image):
    # assumes dark wall
    # works for: templateshapes, templatesale and 2, template_3_bad!!!, withbackground...
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)
    x, y, w, h = cv2.boundingRect(contours[0])
    rect = cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), 2)        
    cropped = thresh[y:y + h, x:x + w] 
    return cropped

# Source: https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/
def main():
    # currently work for title: noback2, salenoback 
    image = cv2.imread('images/templatesale4.jpg')
    croppedImage = removeWall(image)
    inverse = cv2.bitwise_not(croppedImage)
    ksize = 50
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dilation = cv2.dilate(inverse, rect_kernel, iterations = 1)    
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)

    file = open("recognized.txt", "w+")
    file.write("")
    file.close()
    possible_titles = {}
    j = 0

    im2 = image.copy()   
    print(len(contours)) 
    for cnt in contours:
        # Source: https://stackoverflow.com/questions/24385714/detect-text-region-in-image-using-opencv 
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = inverse[y:y + h, x:x + w] # was thresh # was im2
        file = open("recognized.txt", "a")
        text = pytesseract.image_to_string(cropped)

        '''cv2.imshow('image', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        # Get title    
        possible_titles[j] = get_sentence_info(cropped) 
        j += 1

        '''newtext = ""
        for i in range(1, len(text)):
            if text[i] == '\n' and text[i - 1] == '\n':
                newtext += " "
            else:
                newtext += text[i]
        regions = re.split("\n{2}", text)
        print(regions)'''        
        # then split string by whitespace, get rid of duplicates?
        # actually: just look for date and time, once found, stop searching
        
        file.write(text)
        file.write("\n")        
        file.close()   
    
    # title is sentence with max height
    max_val = max(possible_titles.values(), key=lambda sub: sub[0])[1]
    new_title = clean_title(max_val)
    print(new_title)    


main()
