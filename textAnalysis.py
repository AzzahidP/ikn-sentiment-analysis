import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from dataPreparation import word_frequency
from nltk.tokenize import word_tokenize

## Distribusi frekuensi jumlah huruf
def char_dist(data):
    bin_range = np.arange(0, 260, 10)
    data.str.len().hist(bins=bin_range)
    plt.title('Distribusi Frekuensi Jumlah Huruf')
    plt.show()


## Distribusi frekuensi jumlah kata
def word_dist(data):
    bin_range = np.arange(0, 50)
    data.str.split().map(lambda x: len(x)).hist(bins=bin_range)
    plt.title('Distribusi Frekuensi Jumlah Kata')
    plt.show()

## Distribusi frekuensi panjang kata rata-rata
def word_length_avg_dist(data):
    data.str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist()
    plt.title('Distribusi Frekuensi Panjang Kata Rata-Rata')
    plt.show()


## Distribusi frekuensi kata yang sering muncul
def word_freq_dist(data):
    word_frequency(data).plot(10,cumulative=False, title='Distribusi Frekuensi Kata Popular')

## Distribusi Bigram
def bigram_dist(data):
    bigram_data = []
    for title in data:
        a = word_tokenize(title)
        bigram_data += a

    result = pd.Series(nltk.ngrams(bigram_data, 2)).value_counts()[:10]
    print(result)