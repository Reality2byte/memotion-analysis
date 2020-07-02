import pandas as pd
import numpy as np
import os

#import argparse
#from enum import Enum
import io
import json

from google.cloud import vision
from google.cloud.vision import types
#from PIL import Image, ImageDraw
from google.protobuf.json_format import MessageToJson


df = pd.read_csv('data_7000_new.csv')
img_folder = 'data_7000'

def save_json(response, json_file):
    json_folder = 'ocr_json'
    json_filename = os.path.splitext(json_file)
    json_filename = json_filename[0] + '.json'
    json_path = os.path.join(json_folder, json_filename)
    print('json_path',json_path)
    #my_data = response.json()
    my_data = MessageToJson(response)
    file=my_data.replace('\n','')
    my_data=json.loads(file)
   # my_data = json.loads(my_data)
    try:
        with open(json_path, 'w') as f:
            json.dump(my_data, f)
        return json_filename
    except:
        pass

def get_document_bounds(img_file):
    """Returns document bounds given an image."""
    client = vision.ImageAnnotatorClient.from_service_account_file('C:\\Users\\harsh\\Downloads\\memotion_analysis\\ocr\\ocrenv\\gcred.json')
    imgpath = os.path.join(img_folder, img_file)
    bounds = []
    #text = []
    try:
        with io.open(imgpath, 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)

        response = client.document_text_detection(image=image)
        document = response.full_text_annotation
        texts = response.text_annotations
        
        #print('response ',response)
        #print('document ',document)
        #for text in texts:
        ocr_text = texts[0].description
        print('textsssss\n"{}"'.format(ocr_text))
        json_filename = save_json(response, img_file)
    ##    # Collect specified feature bounds by enumerating all document features
    ##    for page in document.pages:
    ##        for block in page.blocks:
    ##            for paragraph in block.paragraphs:
    ##                for word in paragraph.words:
    ##                    for symbol in word.symbols:
    ##                        if (feature == FeatureType.SYMBOL):
    ##                            print('a')
    ##                            #bounds.append(symbol.bounding_box)
    ##
    ##                    if (feature == FeatureType.WORD):
    ##                        print('a')
    ##                        #bounds.append(word.bounding_box)
    ##
    ##                if (feature == FeatureType.PARA):
    ##                    print('a')
    ##                    #bounds.append(paragraph.bounding_box)
    ##
    ##            if (feature == FeatureType.BLOCK):
    ##                bounds.append(block.bounding_box)
    ##
    ##    # The list `bounds` contains the coordinates of the bounding boxes.
        return ocr_text, json_filename

    except:
        pass


for i in range(len(df)):
    imgname = df['Image_name'][i]
    
    try:
        ocr_text, json_filename = get_document_bounds(imgname)
        df['my_ocr'][i], df['json_name'][i] = ocr_text, json_filename
        if df['Overall_Sentiment'][i] == 'very_positive':
            df['sentiment'][i] = '1'
        if df['Overall_Sentiment'][i] == 'positive':
            df['sentiment'][i] = '2'
        if df['Overall_Sentiment'][i] == 'neutral':
            df['sentiment'][i] = '3'
        if df['Overall_Sentiment'][i] == 'negative':
            df['sentiment'][i] = '4'
        if df['Overall_Sentiment'][i] == 'very_negative':
            df['sentiment'][i] = '5'
    except:
        pass
    
df.to_csv('memotion.csv', index=False)

##get_document_bounds('avengers_1pd1hg.jpg')
