import json
from flask import jsonify

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
    res = extract_bad_words(str(decade))
    result[decade] = res
    decade += 10

print(result)
# code for change bad words over years
# _1910 = extract_bad_words("1910")
# result["_1910"] = _1910
# print(_1910)
#
_1920 = extract_bad_words(str(1920))
print(_1920)
#
# _1930 = extract_bad_words("1930")
# print(_1930)
#
# _1940 = extract_bad_words("1940")
# print(_1940)
#
# _1950 = extract_bad_words("1950")
# print(_1950)
#
# _1960 = extract_bad_words("1960")
# print(_1960)
#
# _1960 = extract_bad_words("1970")
# print(_1960)
#
# _1960 = extract_bad_words("1980")
# print(_1960)
#
# _1960 = extract_bad_words("1990")
# print(_1960)
#
# _1960 = extract_bad_words("2000")
# print(_1960)
#
# _1960 = extract_bad_words("2010")
# print(_1960)
#
#
with open('result_path.json', 'w') as f:
    json.dump(jsonify(result), f)
f.close()

# code for change bad words over genres
