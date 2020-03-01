import json
from flask import jsonify
import os

# code for extracting bad words

def extract_bad_words(year):
    with open(year, 'r',  encoding="utf-8") as f:
        text = f.read()

    f = open("bad-words.txt", "r")
    words_text = f.read()
    words_text = words_text.split("\n")

    words_text = set(words_text)
    # print(words_text)
    dic = dict()

    for sentence in text.split("\n"):
        words_in_sentence = set(sentence.split())
        intersec = words_in_sentence.intersection(words_text)
        if intersec:
            for word in intersec:
                if word in dic.keys():
                    dic[word] += 1
                else:
                    dic[word] = 1

    return dic.items()


result = {}

decade = 1910
while decade != 2020:
    res = extract_bad_words(os.path.join('decades', str(decade)))
    result[decade] = res
    decade += 10

with open('result.txt', 'w') as f:
    f.write(result)
f.close()

# code for change bad words over genres
