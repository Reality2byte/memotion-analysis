import os
import pandas as pd
import ktrain

import re, string, unicodedata
import nltk
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def tup2dict(tup): 
    di = {}
    for a, b in tup: 
        di[a] = b
    return di

##def answers(pred):
##    taskA = '9'
##    taskB = ['9']*4
##    taskC = ['9']*4
##    p=pred['positive']
##    neg=pred['negative']
##    neu=pred['neutral']
##
##
##    if neg>0.250 :
##        taskA = '-1'
##    elif neu>0.4:
##        taskA = '0'
##    else:
##        taskA = '1'
##
##    if pred['general']>0.70 :
##        taskB[1] = '1'
##        taskC[1] = '1'
##    elif pred['not_sarcastic'] < pred['twisted_meaning'] and pred['very_twisted'] < pred['twisted_meaning']:
##        taskB[1] = '1'
##        taskC[1] = '2'
##    elif pred['very_twisted'] > pred['twisted_meaning'] and pred['very_twisted'] > pred['not_sarcastic']:
##        taskB[1] = '1'
##        taskC[1] = '3'
##    elif pred['not_sarcastic'] > pred['twisted_meaning'] and pred['very_twisted'] < pred['not_sarcastic']:
##        taskB[1] = '0'
##        taskC[1] = '0'
##
##    if pred['general']>0.70 :
##        taskB[1] = '1'
##        taskC[1] = '1'
##
##
##        
####    if pred['positive'] > pred['negative'] and pred['positive'] > pred['neutral']:
####        taskA = '1'
####    elif pred['positive'] < pred['negative'] and pred['negative'] > pred['neutral']:
####        taskA = '-1'
####    elif pred['positive'] < pred['neutral'] and pred['negative'] < pred['neutral']:
####        taskA = '0'
## 
##    if pred['not_funny'] > pred['funny'] and pred['not_funny'] > pred['very_funny'] and pred['hilarious'] < pred['not_funny']:
##        taskB[0] = '0'
##        taskC[0] = '0'
##    elif pred['hilarious'] < pred['funny'] and pred['funny'] > pred['very_funny'] and pred['funny'] > pred['not_funny']:
##        taskB[0] = '1'
##        taskC[0] = '1'
##    elif pred['very_funny'] > pred['funny'] and pred['hilarious'] < pred['very_funny'] and pred['very_funny'] > pred['not_funny']:
##        taskB[0] = '1'
##        taskC[0] = '2'
##    elif pred['hilarious'] > pred['funny'] and pred['hilarious'] > pred['very_funny'] and pred['hilarious'] > pred['not_funny']:
##        taskB[0] = '1'
##        taskC[0] = '3'
##
##    if pred['not_sarcastic'] > pred['general'] and pred['not_sarcastic'] > pred['twisted_meaning'] and pred['very_twisted'] < pred['not_sarcastic']:
##        taskB[1] = '0'
##        taskC[1] = '0'
##    elif pred['not_sarcastic'] < pred['general'] and pred['general'] > pred['twisted_meaning'] and pred['very_twisted'] < pred['general']:
##        taskB[1] = '1'
##        taskC[1] = '1'
##    elif pred['not_sarcastic'] < pred['twisted_meaning'] and pred['general'] < pred['twisted_meaning'] and pred['very_twisted'] < pred['twisted_meaning']:
##        taskB[1] = '1'
##        taskC[1] = '2'
##    elif pred['general'] < pred['very_twisted'] and pred['very_twisted'] > pred['twisted_meaning'] and pred['very_twisted'] > pred['not_sarcastic']:
##        taskB[1] = '1'
##        taskC[1] = '3'
##
##    if pred['not_offensive'] > pred['slight'] and pred['not_offensive'] > pred['very_offensive'] and pred['not_offensive'] > pred['hateful_offensive']:
##        taskB[2] = '0'
##        taskC[2] = '0'
##    elif pred['not_offensive'] < pred['slight'] and pred['slight'] > pred['very_offensive'] and pred['slight'] > pred['hateful_offensive']:
##        taskB[2] = '1'
##        taskC[2] = '1'
##    elif pred['not_offensive'] < pred['very_offensive'] and pred['slight'] < pred['very_offensive'] and pred['very_offensive'] > pred['hateful_offensive']:
##        taskB[2] = '1'
##        taskC[2] = '2'
##    elif pred['not_offensive'] < pred['hateful_offensive'] and pred['hateful_offensive'] > pred['very_offensive'] and pred['slight'] < pred['hateful_offensive']:
##        taskB[2] = '1'
##        taskC[2] = '3'
##
##    if pred['not_motivational'] > pred['motivational']:
##        taskB[3] = '0'
##        taskC[3] = '0'
##    elif pred['not_motivational'] < pred['motivational']:
##        taskB[3] = '1'
##        taskC[3] = '1'
##
##
##    line = taskA + '_' + ''.join(taskB) + '_' + ''.join(taskC) + '\n'
##
##    return line



def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_URL(sample):
    """Remove URLs from a sample string"""
    rurl = re.sub(r"http\S+", "", sample)
    return re.sub(r"\w+[.]com", "", rurl)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def preprocess(sample):
    sample = remove_URL(sample)
    sample = replace_contractions(sample)
    # Tokenize
    words = nltk.word_tokenize(sample)

    # Normalize
    return normalize(words)

def pre_preprocess(xval):
    xval = str(xval.replace('\n',' ').replace('|','').encode("utf-8"))[2:-1]
    xval = xval.lower()
    xval = remove_URL(xval)
    xval = replace_contractions(xval)

    # Tokenize
    words = nltk.word_tokenize(xval)
    #print(words)

    # Normalize
    words = normalize(words)
    #print(words)

    words = [word for word in words if word.isalpha()]
    xval = ' '.join(words)

    return xval



##main


answers_dict = {'hilarious':'3','very_funny':'2','funny':'1','general':'1', 'twisted_meaning':'2', 'very_twisted':'3', 'not_sarcastic':'0','not_funny':'0','motivational':'1','not_motivational':'0', 'positive':'1', 'neutral':'0', 'negative':'-1','not_offensive':'0','very_offensive':'2', 'slight':'1', 'hateful_offensive':'3'}


humourpredictor = ktrain.load_predictor('/home/dgxuser136/ambuje1/senti_humour_ktrainbert.h5')
motivationpredictor = ktrain.load_predictor('/home/dgxuser136/ambuje1/motivation_ktrainbert.h5')
sentipredictor = ktrain.load_predictor('/home/dgxuser136/ambuje1/senti_ktrainbert.h5')
onlyfunnypredictor = ktrain.load_predictor('/home/dgxuser136/ambuje1/senti_onlyfunny_ktrainbert.h5')
offensivepredictor = ktrain.load_predictor('/home/dgxuser136/ambuje1/senti_offensive_ktrainbert.h5')
sarcasticpredictor = ktrain.load_predictor('/home/dgxuser136/ambuje1/senti_sarcastic_ktrainbert.h5')
onlysarcasticpredictor = ktrain.load_predictor('/home/dgxuser136/ambuje1/senti_onlysarcastic_ktrainbert.h5')
onlyoffensivepredictor = ktrain.load_predictor('/home/dgxuser136/ambuje1/senti_onlyoffensive_ktrainbert.h5')

df = pd.read_csv('/home/dgxuser136/ambuje1/2000_testdata.csv')

file1 = open("answer_8models.txt","w")

#x = 'Phrase|Index\n'
#print (tup2dict(tups, dictionary))

for i in range(len(df)):
    #text = str(df['cleaned_ocr'][i])
    text = pre_preprocess(str(df['corrected_text'][i]))

    taskA = '9'
    taskB = ['9']*4
    taskC = ['9']*4
    

    pred = sentipredictor.predict(text)
    print(pred)
    
##    pred = tup2dict(pred)
##    pred = max(pred, key=pred.get)
    taskA = answers_dict[pred]
    
    pred = humourpredictor.predict(text)
    print(pred)
##    pred = tup2dict(pred)
##    pred = max(pred, key=pred.get)
    if pred == 'fun':
        taskB[0] = '1'
    else:
        taskB[0] = '0'
##    taskB[0] = answers_dict[pred]
    if taskB[0] == '1':
        pred = onlyfunnypredictor.predict(text)
        print(pred)
##        pred = tup2dict(pred)
##        pred = max(pred, key=pred.get)
        taskC[0] = answers_dict[pred]
    else:
        taskC[0] = '0'

    pred = sarcasticpredictor.predict(text)
    print(pred)
    if pred == 'sarcastic':
        taskB[1] = '1'
    else:
        taskB[1] = '0'
##    pred = tup2dict(pred)
##    pred = max(pred, key=pred.get)
##    taskB[1] = answers_dict[pred]
    if taskB[1] == '1':
        pred = onlysarcasticpredictor.predict(text)
        print(pred)
##        pred = tup2dict(pred)
##        pred = max(pred, key=pred.get)
        taskC[1] = answers_dict[pred]
    else:
        taskC[1] = '0'

    pred = offensivepredictor.predict(text)
    print(pred)
    if pred == 'offensive':
        taskB[2] = '1'
    else:
        taskB[2] = '0'
##    pred = tup2dict(pred)
##    pred = max(pred, key=pred.get)
##    taskB[2] = answers_dict[pred]
    if taskB[2] == '1':
        pred = onlyoffensivepredictor.predict(text)
        print(pred)
##        pred = tup2dict(pred)
##        pred = max(pred, key=pred.get)
        taskC[2] = answers_dict[pred]
    else:
        taskC[2] = '0'

    pred = motivationpredictor.predict(text)
    print(pred)
##    pred = tup2dict(pred)
##    pred = max(pred, key=pred.get)
    taskB[3] = answers_dict[pred]
    taskC[3] = taskB[3]   

    ans = taskA + '_' + ''.join(taskB) + '_' + ''.join(taskC) + '\n'
    
    

    file1.write(ans)
    

file1.close()

##pred = [('hilarious', 0.10454379), ('not_funny', 0.21206911), ('very_funny', 0.32491603), ('funny', 0.36210373), ('general', 0.55724233), ('not_sarcastic', 0.11089532), ('twisted_meaning', 0.24999025), ('very_twisted', 0.09583253), ('not_offensive', 0.10120422), ('very_offensive', 0.30312324), ('slight', 0.5820346), ('hateful_offensive', 0.04012201), ('not_motivational', 0.9636585), ('motivational', 0.036573086), ('positive', 0.5848341), ('neutral', 0.28037217), ('negative', 0.14398904)]
##pred = tup2dict(pred)
##print(answers(pred))
