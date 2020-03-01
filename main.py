import json
from flask import jsonify
import os
import csv

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


def generate_bad_words_by_decades():
    decade = 1910
    with open('result-decade.csv', mode='w') as csv_file:

        while decade != 2020:
            result = extract_bad_words(os.path.join('decades', str(decade)))
            fieldnames = ['decade', 'bad word', 'number of appearance']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for res in result:
                writer.writerow({'decade': decade, 'bad word': res[0], 'number of appearance': res[1]})

            decade += 10
    csv_file.close()


def generate_bad_words_by_score():
    score = 0
    with open('result-score.csv', mode='w') as csv_file:

        while score != 9:
            result = extract_bad_words(os.path.join('score', 'score_' + str(score)))
            fieldnames = ['score', 'bad word', 'number of appearance']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for res in result:
                writer.writerow({'score': score, 'bad word': res[0], 'number of appearance': res[1]})

            score += 1
    csv_file.close()


if __name__ == '__main__':
    generate_bad_words_by_decades()
    generate_bad_words_by_score()
