import collections
import pandas as pd
import matplotlib.pyplot as plt
import string
import time

wordcount = {}
# file = open('PrideandPrejudice.txt', encoding="utf8")
# Stopwords
stopwords = set(line.strip() for line in open('/Volumes/Treasures/Needle_project/stopwords.txt'))
stopwords = stopwords.union(set(['mr','mrs','said', 'i']))
curse_words = set(line.strip() for line in open('/Volumes/Treasures/Needle_project/bad_words.txt'))
positive_words = set(line.strip() for line in open('/Volumes/Treasures/Needle_project/positive_words.txt'))
# Read input file, note the encoding is specified here
# It may be different in your text file
t0 = time.time()
with open("/Volumes/Treasures/Needle_project/smaller_subs.txt",'r',buffering=100000) as f:
    for line in f:
        # stopwords = set(line.strip() for line in open('stopwords.txt'))
        for word in line.lower().split():
            word = word.replace(".", "")
            word = word.replace(",", "")
            word = word.replace(":", "")
            word = word.replace("\"", "")
            word = word.replace("!", "")
            word = word.replace("-", "")
            word = word.replace("â€œ", "")
            word = word.replace("â€˜", "")
            word = word.replace("*", "")
            word = word.replace("[", "")
            word = word.replace("]", "")
            word = word.replace("(", "")
            word = word.replace(")", "")
            word = word.replace("?", "")
            if word.isspace():
                continue
            if word in positive_words:
                if word not in wordcount:
                    wordcount[word] = 1
                else:
                    wordcount[word] += 1
t1 = time.time()

# Print most common word
print("time it took to process file: %.3f"%(t1-t0))
n_print = int(input("How many most common words to print: "))
print("\nOK. The {} most common words are as follows\n".format(n_print))
word_counter = collections.Counter(wordcount)
for word, count in word_counter.most_common(n_print):
    print(word, ": ", count)
# Close the file
# file.close()
# Create a data frame of the most common words
# Draw a bar chart
lst = word_counter.most_common(n_print)
df = pd.DataFrame(lst, columns = ['Word', 'Count'])
df.plot.bar(x='Word',y='Count')
plt.show()