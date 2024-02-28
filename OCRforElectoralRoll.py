#Imports
import shutil
from pdf2image import convert_from_path
import cv2
from pathlib import Path
import pytesseract
import json
import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# Set the path for your Tesseract executable
tesseract_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = tesseract_path

if not tesseract_path.exists():
    raise FileNotFoundError(f"Tesseract executable not found at {tesseract_path}")

# Set the path for your poppler-utils bin folder
poppler_path = Path(r"D:\poppler-utils\poppler-24.02.0\Library\bin")

if not poppler_path.exists():
    raise FileNotFoundError(f"Poppler utils folder not found at {poppler_path}")

# Set the path for input and output folders
input_folder = Path(r"D:\OCR\input")
output_folder = Path(r"D:\OCR\output")

if not input_folder.exists():
    raise FileNotFoundError(f"Input Folder not found at {input_folder}")
if not output_folder.exists():
    os.makedirs(output_folder, exist_ok=True)

#Custom Errors
class FolderCreationError(Exception):
    pass

# Required functions
def save_json(page_data, pdf_file):
    try:
        json_output_path = output_folder / f"{pdf_file.stem}.json"
        with open(json_output_path, 'w') as f:
            json.dump(page_data, f)
        print(f"Results saved successfully to {json_output_path}")
    except Exception as e:
        raise RuntimeError(f"Error while saving data: {e}")

def save_to_csv(output_dict, csv_file):
    data_list = []

    for key, data_dict in output_dict.items():
        data_dict["ID"] = key
        data_list.append(data_dict)

    df = pd.DataFrame(data_list)
    df.to_csv(csv_file, index=False, encoding='utf-8')

def make_result_folder(output_folder, pdf_file, output_dict):
    try:
        folder_name = pdf_file.stem
        target_path = output_folder / folder_name
        
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        """
        json_output_path = target_path / f"{pdf_file.stem}.json"
        with open(json_output_path, 'w') as f:
            json.dump(page_data, f)
        """

        csv_output_path = target_path / f"{folder_name}.csv"
        save_to_csv(output_dict, csv_output_path)
        shutil.copy(pdf_file, target_path)
        print(f"Results saved successfully to {csv_output_path}")
        return
    except Exception as e:
        raise FolderCreationError(f"Failed to create result folder for {pdf_file}: {e}")

def skip_file(pdf_file, file_no):
    print(f"Skipping file {pdf_file} with File number-{file_no}")

def preprocess_for_detection(page):
    try:    
        gray_img = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (11,11), 0)
        thresh = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        """
        kernel = np.ones((5,5),np.uint8)
        dilated_img = cv2.dilate(inverted_img, kernel, iterations = 1)
        eroded_img = cv2.erode(dilated_img, kernel, iterations = 1)
        """
        return thresh
    except Exception as e:
        raise RuntimeError(f"Unable to preprocess images:{e}")

def find_id_contours(image):
    try:
        contours,_ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 90000
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        tolerance = 5
        contours = sorted(contours, key=lambda cnt: (round(cv2.boundingRect(cnt)[1]), round(cv2.boundingRect(cnt)[0], tolerance)))
        return contours
    except Exception as e:
        raise RuntimeError(f"Unable to find contours for file number:{e}")
    
def ocr_ids(contours, page_for_extraction):
    try:
        ocr_text = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
                    
            ROI = page_for_extraction[y:y+h, x:x+w]
            text = pytesseract.image_to_string(ROI, config=r"--oem 3 --psm 11") #, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            ocr_text.append(text)
        return ocr_text
    except Exception as e:
        raise RuntimeError(f"Error while performing OCR: {e}")
    
def extract_data(ocr_text, card_counter, output_dict):
    try:
        for card_no, text in enumerate(ocr_text, start=1):

            unique_id_match = re.search(r'\b[A-Z]+\d+\b', text)
            unique_id = unique_id_match.group() if unique_id_match else None
                
            name_match = re.search(r'Name\s*:\s*([A-Z\s.]+)', text)
            name = name_match.group(1).split('\n', 1)[0].strip() if name_match else None

            father_name_match = re.search(r"Father's\s+Name\s*:\s*([A-Z\s.]+)", text)
            father_name = father_name_match.group(1).split('\n', 1)[0].strip() if father_name_match else None
                
            spouse_name_match = re.search(r"Name\s+of\s+Spouse\s*:\s*([A-Z\s.]+)", text)
            spouse_name = spouse_name_match.group(1).split('\n', 1)[0].strip() if spouse_name_match else None

            house_number_match = re.search(r'House\s+Number\s*:\s*(\d+)?', text)
            house_number = house_number_match.group(1).strip() if house_number_match and house_number_match.group(1) else None
                
            age_match = re.search(r'Age\s*:\s*(\d+)', text)
            age = age_match.group(1).strip() if age_match else None
                
            gender_match = re.search(r'Gender\s*:\s*(Male|Female)', text)
            gender = gender_match.group(1).strip() if gender_match else None

            key = f"{card_no+card_counter}"

            data_dict = {
                "unique_id": unique_id,
                "Name": name,
                "Father's Name": father_name,
                "Name of Spouse": spouse_name,
                "House Number": house_number,
                "Age": age,
                "Gender": gender
            }
            output_dict[key] = data_dict

        card_counter += card_no    
        return output_dict, card_counter
        
    except Exception as e:
        raise RuntimeError(f"Error while extracting data: {e}")

#Process files
for file_no, pdf_file in enumerate(input_folder.glob('*.pdf'), start=1):

    if pdf_file.suffix.lower() != '.pdf':
        skip_file(pdf_file, file_no)
        continue

    images = convert_from_path(pdf_file, poppler_path=poppler_path)
    images = images[2:-1]
    np_images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in images]
    output_dict = dict()
    card_counter = 0

    print(f"Processing file no-{file_no}, file name- {pdf_file.stem}")
    
    for page_no, page in enumerate(np_images, start=1):

        page_for_detection = preprocess_for_detection(page)
        contours = find_id_contours(page_for_detection)
        page_for_extraction = page
        ocr_texts = ocr_ids(contours, page_for_extraction)
        output_dict, card_counter = extract_data(ocr_texts, card_counter, output_dict)
    
    make_result_folder(output_folder, pdf_file, output_dict)
    print(f"File number-{file_no} is processed successfully")
     



