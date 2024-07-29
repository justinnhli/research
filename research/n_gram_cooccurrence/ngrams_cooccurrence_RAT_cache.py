""" Creates a cache of cooccurrent relations to each of the remote associates words for use in the soft
    cooccurrence thresholded spreading mechanism for ngrams."""

from google_ngrams import GoogleNGram
import json
import csv
from collections import defaultdict

ngrams = GoogleNGram('~/ngram_sample')

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
cooc_rels_dict = defaultdict(set)
counter = 0

for trial in rat_file:
    if counter == 1:
        break
    counter += 1
    row = trial
    context = tuple(row[:3])
    for word in context:
        rels_set = set([elem[0] for elem in ngrams.get_max_probability(word)])
        cooc_rels_dict[word] = rels_set
        while rels_set:
            leng = len(rels_set)
            new_rels_set = set()
            for rel in rels_set:
                items = set([elem[0] for elem in ngrams.get_max_probability(rel)]).difference(stopwords)
                cooc_rels_dict[rel] = items
                new_rels_set.update(items)
            added_words.update(rels_set)
            rels_set = new_rels_set.difference(added_words)

path = "ngrams_RAT_cooccurrence_rels_cache.json"
file = open(path, 'w')
json.dump(cooc_rels_dict, file)
file.close()
