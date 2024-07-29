""" Creates a cache of cooccurrent relations to each of the remote associates words for use in the soft
    cooccurrence thresholded spreading mechanism for ngrams."""

from google_ngrams import GoogleNGram
import json
import csv
from collections import defaultdict

ngrams = GoogleNGram('~/ngram')

stopwords_link = '/Users/lilygebhart/Documents/GitHub/research/research/nltk_english_stopwords'

with open(stopwords_link, "r") as stopwords_file:
    lines = stopwords_file.readlines()
    stopwords = []
    for l in lines:
        stopwords.append(l[:-1])
stopwords = set(stopwords)

rat_file_link = '/Users/lilygebhart/Documents/GitHub/research/research/RAT/RAT_items.txt'
rat_file = csv.reader(open(rat_file_link))
next(rat_file)

added_words = set()
cooc_rel_dict = defaultdict(set)

counter = 0
for trial in rat_file:
    counter += 1
    print("Trial #", counter)
    row = trial
    context = tuple(row[:3])
    cont = 0
    for word in context:
        cont += 1
        print("context #", cont)
        cooc_rel_dict[word] = set([elem[0] for elem in ngrams.get_max_probability(word)]).difference(stopwords)

for key in cooc_rel_dict.keys():
    cooc_rel_dict[key] = list(cooc_rel_dict[key])
path = "ngrams_simple_RAT_cooccurrence_rels_cache.json"
file = open(path, 'w')
json.dump(cooc_rel_dict, file)
file.close()
