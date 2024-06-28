"""
Script to create and save to file a dictionary with all words (and their counts)
 that cooccur with all 3 RAT context words on each trial.
"""

from google_ngrams import GoogleNGram
import json
import csv

cooc_dict_list = []  # To be filled with the words (and their counts) that cooccur with all 3 RAT context words
ngrams = GoogleNGram('~/ngram')

stopwords_link = "/Users/lilygebhart/nltk_data/corpora/stopwords/nltk_english_stopwords"
with open(stopwords_link, "r") as stopwords_file:
    lines = stopwords_file.readlines()
    stopwords = []
    for line in lines:
        stopwords.append(line[:-1])

rat_file_link = '/Users/lilygebhart/Documents/GitHub/research/research/RAT/RAT_items.txt'
rat_file = csv.reader(open(rat_file_link))
next(rat_file)
for trial in rat_file:
    print("trial", trial)
    row = trial
    context = tuple(row[:3])
    cooc_set1 = set([elem[0] for elem in ngrams.get_max_probability(context[0])])
    print("set1")
    cooc_set2 = set([elem[0] for elem in ngrams.get_max_probability(context[1])])
    print("set2")
    cooc_set3 = set([elem[0] for elem in ngrams.get_max_probability(context[2])])
    print("set3")
    joint_cooc_set = cooc_set1 & cooc_set2 & cooc_set3
    print("joint", joint_cooc_set)
    cooc_words_list = []
    for elem in joint_cooc_set:
        if elem.lower() not in stopwords:
            conditional_prob1 = ngrams.get_conditional_probability(base=context[0], target=elem.upper())
            conditional_prob2 = ngrams.get_conditional_probability(base=context[1], target=elem.upper())
            conditional_prob3 = ngrams.get_conditional_probability(base=context[2], target=elem.upper())
            conditional_prob = conditional_prob1 * conditional_prob2 * conditional_prob3
            # Each co-occurring word is represented as a list: [word (str), word joint conditional probability (int)]
            cooc_words_list.append([elem.upper(), conditional_prob])
    cooc_dict_list.append([context, cooc_words_list])
path = "ngrams_cooccurrence_cache.json"
file = open(path, 'w')
json.dump(cooc_dict_list, file)
file.close()






