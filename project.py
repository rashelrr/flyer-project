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
from datetime import datetime
from imutils.perspective import four_point_transform


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
    if word_heights:
        avg_word_height = sum(word_heights.values()) / len(word_heights)
        sentence = ' '.join(words.values())
        return (avg_word_height, sentence)
    else:
        return (0, "") # no text in file

def get_date(sentences):
    for text in sentences:
        match = re.search(r'\d{2}/\d{2}/\d{4}', text)
        if match:
            date = datetime.strptime(match.group(), '%m/%d/%Y').strftime("%m/%d/%Y")
            return date

# Adds space between number and time of day
# Adds ":00" if no minutes
def reformat_times(times):
    # times has at most 2 elements
    reformatted = []
    for time in times:
        # extract the number
        t = time.replace("AM", '')
        t = t.replace("PM", '')

        t += ":00 " if ':' not in t else " "
        t += time[-2:]
        reformatted.append(t)
    return reformatted

# Source: https://stackoverflow.com/questions/20437207/using-python-regular-expression-to-match-times
def get_times(sentences):
    times = []
    for text in sentences:
        text_times = re.findall(r'\d{1,2}(?:(?:AM|PM)|(?::\d{1,2})(?:AM|PM)?)', text)
        if text_times: 
            # prevent duplicate times
            [times.append(x) for x in text_times if x not in times]    

    # in case errors w pytesseract occur, use at most first 2 times
    reformatted_times = reformat_times(times[0:2])
    return reformatted_times

def removeWall(image):
    # assumes dark wall
    # works for: templateshapes, templatesale and 2, template_3_bad!!!, withbackground...
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    ret, thresh = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)
    big_contour = max(contours, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(big_contour)
    cropped = thresh[y + 50:y -50 + h, x + 100:x -100 + w] 
    return cropped
    '''
    #cropped = thresh[y:y + h, x:x + w] 
    newcropped = trim(cropped)
    return newcropped'''

# Source: https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
def trim(frame):
    '''if 0 in cropped[:, 0]:
        return trim(cropped[:, 1:])
    else:
        return cropped'''
    if 0 in frame[:, 0]:
        return trim(frame[:, 10:]) # check if leftmost row isn25% filled with black - if less, we're fine, else trim more
    else:
        return frame

def check_bad_elements(title, date, time):
    return (title is None or date is None or time is None
        or title == "" or date == "" or time == [])

def create_calendar_event(title, date, time):
    with open('output.csv', 'w') as f:
        writer = csv.writer(f)
        start_time = time[0]

        if len(time) == 1:
            headers = ['Subject', 'Start date', 'Start time']
            content = [title, date, start_time]
            writer.writerow(headers)
            writer.writerow(content)
        
        if len(time) == 2:
            end_time = time[1]
            headers = ['Subject', 'Start date', 'Start time', 'End time']
            content = [title, date, start_time, end_time]
            writer.writerow(headers)
            writer.writerow(content)

# Source: https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/
def main():
    # currently work for title: noback2, salenoback 
    # working in test: opening, opening2, sale, autumn (not party)
    image = cv2.imread('test/autumn.jpg')
    croppedImage = removeWall(image)
    inverse = cv2.bitwise_not(croppedImage)
    ksize = 100
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dilation = cv2.dilate(inverse, rect_kernel, iterations = 1)    
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)
    
    possible_titles = {}
    sentences = []
    j = 0

    print(len(contours)) 
    for cnt in contours:
        # Source: https://stackoverflow.com/questions/24385714/detect-text-region-in-image-using-opencv 
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = inverse[y:y + h, x:x + w] # was thresh 
        text = pytesseract.image_to_string(cropped)#, config='--psm 6')
        sentences.append(text)

        # Get title    
        possible_titles[j] = get_sentence_info(cropped) 
        j += 1

        '''cv2.imshow('image', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

    
    # Get title
    # title is sentence with max height
    if possible_titles:
        max_val = max(possible_titles.values(), key=lambda sub: sub[0])[1]
        new_title = clean_title(max_val)
    else: 
        new_title = ""

    clean_sentences = []
    for ele in sentences:
        clean_sentences.append(ele.replace("\n", ' '))

    # Get date
    date = get_date(clean_sentences)

    # Get time
    time = get_times(clean_sentences)

    print(new_title, date, time)

    # Check if required elements present
    if check_bad_elements(new_title, date, time):
        print("Some required elements could not be found.")
        print("Failed to create calendar file.")
    else:
        return create_calendar_event(new_title, date, time)

main()
