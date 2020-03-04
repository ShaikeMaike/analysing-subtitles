import json
from flask import jsonify
import os
import csv
import collections
import pandas as pd
import matplotlib.pyplot as plt


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

    return dic


def graph_for_number_of_words(number_of_words, kind): 
    word_counter = collections.Counter(number_of_words)
    lst = word_counter.most_common(10)
    df = pd.DataFrame(lst, columns=[kind, 'Number'])
    # df.plot.title(str(decate))
    df.plot.bar(x=kind, y='Number')
    plt.show()


def generate_bad_words_by_decades():
    decade = 1910
    number_of_words = dict()
    with open('result-decade.csv', mode='w') as csv_file:

        while decade != 2020:
            result = extract_bad_words(os.path.join('decades', str(decade)))
            if len(result.keys()) == 0:
                decade += 10
                continue
            print(result)
            fieldnames = ['decade', 'bad word', 'number of appearance']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for res in result:
                writer.writerow({'decade': decade, 'bad word': res[0], 'number of appearance': res[1]})

            number_of_words[str(decade)] = len(result.keys())
            decade += 10

        graph_for_number_of_words(number_of_words, 'Decade')


    csv_file.close()


def generate_bad_words_by_score():
    score = 0
    number_of_words = dict()

    with open('result-score.csv', mode='w') as csv_file:

        while score != 9:
            result = extract_bad_words(os.path.join('score', 'score_' + str(score)))
            fieldnames = ['score', 'bad word', 'number of appearance']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for res in result:
                writer.writerow({'score': score, 'bad word': res[0], 'number of appearance': res[1]})
            number_of_words[str(score)] = len(result.keys())

            score += 1
        graph_for_number_of_words(number_of_words, 'Score')

    csv_file.close()


def generate_bad_words_by_genres():
    number_of_words = dict()

    with open('result-genre.csv', mode='w') as csv_file:

        for genre in ['Western', 'War', 'TV', 'Thriller', 'Science', 'Romance', 'Mystery', 'Music',
                      'Horror', 'History', 'Foreign', 'Fantasy', 'Family', 'Drama', 'Documentary',
                      'Crime', 'Comedy', 'Animation', 'Adventure', 'Action']:
            result = extract_bad_words(os.path.join('generes', genre))
            fieldnames = ['genre', 'bad word', 'number of appearance']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for res in result:
                writer.writerow({'genre': genre, 'bad word': res[0], 'number of appearance': res[1]})

            number_of_words[genre] = len(result.keys())
        graph_for_number_of_words(number_of_words, 'Genere')


    csv_file.close()


def generate_bad_words_score_by_decades():

    with open('result-score_by_decades.csv', mode='w') as csv_file:

        decade = 1910
        while decade != 2020:
            score = 0

            while score != 9:
                path = os.path.join('score_by_decades', str(decade), 'score_' + str(score))
                if not os.path.exists(path):
                    score += 1
                    continue
                result = extract_bad_words(path)
                fieldnames = ['decade', 'score', 'bad word', 'number of appearance']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for res in result:
                    writer.writerow({'decade': decade, 'score': score, 'bad word': res[0], 'number of appearance': res[1]})

                score += 1

            decade += 10

    csv_file.close()


def generate_bad_words_genre_by_decades():
    with open('result-genre_by_decades.csv', mode='w') as csv_file:

        decade = 1910
        while decade != 2020:

            for genre in ['Western', 'War', 'TV', 'Thriller', 'Science', 'Romance', 'Mystery',
                          'Music', 'Horror', 'History', 'Foreign', 'Fantasy', 'Family', 'Drama',
                          'Documentary', 'Crime', 'Comedy', 'Animation', 'Adventure', 'Action']:

                path = os.path.join('genres_by_decades', str(decade), genre)
                if not os.path.exists(path):
                    continue
                result = extract_bad_words(path)

                fieldnames = ['decade', 'genre', 'bad word', 'number of appearance']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for res in result:
                    writer.writerow({'decade': decade, 'genre': genre, 'bad word': res[0],
                                     'number of appearance': res[1]})


            decade += 10

    csv_file.close()



if __name__ == '__main__':
    # generate_bad_words_by_decades()
    # generate_bad_words_by_score()
    generate_bad_words_by_genres()
    # generate_bad_words_score_by_decades()
    # generate_bad_words_genre_by_decades()