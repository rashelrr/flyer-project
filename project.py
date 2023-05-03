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


def clean_title(title):
    title_words = title.split()
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
    if word_heights:
        avg_word_height = sum(word_heights.values()) / len(word_heights)
        sentence = ' '.join(words.values())
        return (avg_word_height, sentence)
    else:
        return (0, "") # no text in file

def is_date_valid(date):
    # can assume in format nn/nn/nnnn (n=num)
    mm, dd, year = date.split("/")
    mm, dd, year = int(mm), int(dd), int(year)
    return ((mm in range(1, 13)) and (dd in range(1, 32)) and 
            year in range(23, 26))

def get_date(sentences):
    for text in sentences:
        match = re.search(r'\d{2}/\d{2}/\d{2}', text)
        if match:
            # check within range
            validDate = is_date_valid(match.group())
            if validDate:
                date = datetime.strptime(match.group(), '%m/%d/%y').strftime("%m/%d/%y") # change to y later
                return date
            return None

# Adds space between number and time of day
# Adds ":00" if no minutes
def reformat_times(times):
    reformatted = []
    for time in times:
        # extract the number
        t = time.replace("AM", '')
        t = t.replace("PM", '')

        t += ":00 " if ':' not in t else " "
        t += time[-2:]
        reformatted.append(t)
    return reformatted

def is_time_valid(time):
    # can assume in format n:nn or nn:nn, plus AM/PM (n=num)
    nums = time[:-2]
    x = nums.split(":")
    hour, minute = int(x[0]), int(x[1])

    return ((hour in range(1, 13)) and (minute in range(0, 60)))

# Source: https://stackoverflow.com/questions/20437207/using-python-regular-expression-to-match-times
def get_times(sentences):
    times = []
    for text in sentences:
        text_times = re.findall(r'\d{1,2}(?:(?:AM|PM)|(?::\d{1,2})(?:AM|PM)?)', text)
        if text_times: 
            # prevent duplicate times
            [times.append(x) for x in text_times if x not in times]    

    reformatted_times = reformat_times(times)

    valid_times = []
    for time in reformatted_times:
        if is_time_valid(time):
            valid_times.append(time)

    # in case errors w pytesseract occur, use at most first 2 times
    return valid_times[:2]

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

    cropped = thresh[y:y + h, x:x + w] 
    trim = trimLR(cropped, cropped.shape[0], None) 
    final_trim = trimUD(trim, None, trim.shape[1]) 
    return final_trim

# Source: https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
# if more than 25% black, crop side
def trimLR(frame, rows, cols):
    if np.count_nonzero(frame[:, 0] == 0) > (rows * 0.25): 
        return trimLR(frame[:, 10:], rows, cols) 
    elif np.count_nonzero(frame[:,-1] == 0) > (rows * 0.25): 
        return trimLR(frame[:,:-10], rows, cols) 
    else:
        return frame

def trimUD(frame, rows, cols):
    if np.count_nonzero(frame[0] == 0) > (cols * 0.25):  
        return trimUD(frame[10:], rows, cols) 
    if np.count_nonzero(frame[-1] == 0) > (cols * 0.25):  
        return trimUD(frame[:-10], rows, cols) 
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

# sometimes titles are too big for our kernel
# so add sentences to new_title that are within 10 pixels of max height
def get_title(possible_titles):
    # title: sentence with max height
    if possible_titles:
        max_value = max(possible_titles.values(), key=lambda sub: sub[0])
        max_height = max_value[0]
        reversed_titles = dict(reversed(list(possible_titles.items())))
        large = []
        for key in reversed_titles:
            val = possible_titles[key]
            h = val[0]
            if ((max_height - 11) < h) and ((max_height + 11) > h):
                large.append(val[1])

        new_title = ' '.join(large)
        return clean_title(new_title)
    return ""

def cleanup_sentences(sentences):
    lst = []
    for ele in sentences:
        if ele != '':
            lst.append(ele.replace("\n", ' '))
    return lst

# https://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number
def num_there(s):
    return any(i.isdigit() for i in s)

def fix_date(date, respell, special_respell):
    date = date.replace("|", "/")
    if date.count("/") != 2: # can't do what's below
        return date

    # fix numbers
    lst_date = list(date)
    for i, char in enumerate(lst_date):
        if char in respell:
            lst_date[i] = respell[char]
    new_date = ''.join(lst_date)

    # special cases: ranges
    nums = new_date.split("/")
    mm, dd, year = nums[0], nums[1], nums[2]
    if mm == "17":
        mm = mm.replace('7', special_respell['7'])
    if dd == "37":
        dd = dd.replace('7', special_respell['7'])
    elif dd == "80" or dd == "81":
        dd = dd.replace('8', special_respell['8'])
    if year[-1] == '8':
        year = year[:3] + special_respell['8']
    fixed_date = mm + "/" + dd + "/" + year
    return fixed_date

def fix_time(time, respell, special_respell):
    nums = time[:-2]
    am_pm = time[-2:]

    # fix numbers
    lst_nums = list(nums)
    for i, char in enumerate(lst_nums):
        if char in respell:
            lst_nums[i] = respell[char]
    new_nums = ''.join(lst_nums)

    # special cases
    new_nums = new_nums.replace(".", ":")

    numbers = new_nums.split(":")
    if len(numbers) == 2: # means minutes present
        hour = numbers[0]
        minute = numbers[1]
        for key in special_respell:
            if minute[0] == key:
                minute = special_respell[key] + minute[1]
        new_nums = hour + ":" + minute

    new_nums += am_pm
    return new_nums

def fix_ocr(sentences):
    # sentences: list of strings
    respell = {'A': '4', 'B': '3', 'b': '6', 'D': '0', 'E': '3', 'F': '7', 
               'G': '6', 'g': '9', 'H': '4', 'I': '1', 'i': '1', 'L': '1', 
               'l': '1', 'O': '0', 'q': '9', 'S': '5', 'T': '7', 'U': '0', 
               'Z': '2'}
    special_respell = {'7': '1', '8': '3'}

    # fix possible dates
    for idx, sentence in enumerate(sentences):
        words = sentence.split()
        # fix possible date: assume only 1 word in flyer fits this criteria 
        # since hard to confuse for other non-date strings
        for i, word in enumerate(words):
            if len(word) == 8 and num_there(word):
                words[i] = fix_date(word, respell, special_respell)   
                break
        # fix possible times: assume time ends with am/pm
        for i, word in enumerate(words):
            if (len(word) <= 7 and len(word) >= 3 and 
                (word[-2:] == "AM" or word[-2:] == "PM")):
                words[i] = fix_time(word, respell, special_respell)
        # rejoin words
        new_sentence = ' '.join(words) 
        sentences[idx] = new_sentence
    return sentences

def main():
    files = ["test/autumn.jpg", "test/party.jpg", "test/sale.jpg", 
             "test/green.jpg", "test/opening.jpg", "test/opening2.jpg"]
    files = ["test/green_baddatetime.jpg", "test/green_badtime.jpg"]
    files = ["test/smalleryear.jpg"]
    for f in files:
        process(f)

# Source: https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/
def process(file):
    # currently work for title: noback2, salenoback 
    # working in test: opening, opening2, sale, autumn (not party)
    image = cv2.imread(file)
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

    #print(len(contours)) 
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

    clean_sentences = cleanup_sentences(sentences)
    fixed_sentences = fix_ocr(clean_sentences)
    title = get_title(possible_titles)
    date = get_date(fixed_sentences) 
    time = get_times(fixed_sentences) 

    print(title, date, time)

    # Check if required elements present
    if check_bad_elements(title, date, time):
        print("Some required elements could not be found.")
        print("Failed to create calendar file.")
    else:
        return create_calendar_event(title, date, time)

main()
