import re, string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

## Case Folding
def case_folding(data):
    data = data.lower()
    data = re.sub('@[^\s]+','',data)
    data = ' '.join(re.sub("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",data).split())
    data = re.sub(r"\d+", "", data)
    data = re.sub(r"\n","",data)
    data = re.sub(r"\t","",data)
    data = data.translate(str.maketrans("","",string.punctuation))
    return data

## Remove Stopwords
def remove_stopwords(data):
    sw_indonesia = stopwords.words("indonesian")
    data  = [word for word in data if word not in sw_indonesia]
    data = ' '.join(data)
    return data

## Stemming
def words_stemming(data):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    data = stemmer.stem(data)
    return data     

## Distribusi Frekuensi Kata
def word_frequency(list):
    tokenizedData = []
    for sentence in list:
        t_kata = word_tokenize(sentence)
        tokenizedData += t_kata

    return FreqDist(tokenizedData)