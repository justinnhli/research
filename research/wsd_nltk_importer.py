import random
from collections import defaultdict
from sentence_long_term_memory import sentenceLTM
from sentence_long_term_memory import SentenceCooccurrenceActivation
import nltk
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn_corpus
from nltk.stem import wordnet as wn
import pandas as pd
import json
import os.path


def extract_sentences(num_sentences=-1):
    """
    Runs the word sense disambiguation task.
    Parameters:
        num_sentences (int): The number of sentences from the corpus to use in the task. The first n sentences
            from the corpus are used and if n=-1, all sentences from the corpus are used.
    Returns:
        list: sentence_list (list of all sentences or the first n sentences of the corpus), word_sense_dict (dictionary with the possible senses of
            each word in the corpus)
    """
    if not (os.path.isfile("./sentence_list.json") or os.path.isfile("./word_sense_dict.json")):
        # Checking that file exists
        sentence_list = []
        word_sense_dict = defaultdict(set)
        if num_sentences == -1:
            semcor_sents = semcor.tagged_sents(tag="sem")
        else:
            semcor_sents = semcor.tagged_sents(tag="sem")[0:num_sentences]
        for sentence in semcor_sents:
            temp_word_sense_dict = defaultdict(set)
            sentence_word_list = []
            for item in sentence:
                if not isinstance(item, nltk.Tree):
                    continue
                if not isinstance(item.label(), nltk.corpus.reader.wordnet.Lemma):
                    continue
                # corpus_word = (item.label(), item.label().synset())
                corpus_word = lemma_to_lemmastring(item)
                sentence_word_list.append(corpus_word)
                # temp_word_sense_dict[corpus_word[0].name()].add(corpus_word)
                temp_word_sense_dict[lemmastring_to_synsetstring(corpus_word)].add(corpus_word)
            if len(temp_word_sense_dict) > 1:
                for word, senses in temp_word_sense_dict.items():
                    word_sense_dict[word] = set(word_sense_dict[word])
                    word_sense_dict[word] |= senses
                    word_sense_dict[word] = list(word_sense_dict[word])
                sentence_list.append(sentence_word_list)
        print(word_sense_dict.items())
        sent_list_file = open("./sentence_list.json", 'w')
        json.dump(sentence_list, sent_list_file)
        sent_list_file.close()
        word_sense_dict_file = open("./word_sense_dict.json", 'w')
        json.dump(word_sense_dict, word_sense_dict_file, indent=4)
        word_sense_dict_file.close()
    sentence_list = json.load(open("./sentence_list.json"))
    word_sense_dict = json.load(open("./word_sense_dict.json"))
    return sentence_list, word_sense_dict


def lemma_to_lemmastring(lemma):
    """
    Converts Lemma object to string tuple compatible with all python versions
    """
    if isinstance(lemma.label(), nltk.corpus.reader.wordnet.Lemma):
        lemma_string = lemma.label().name() + " " + lemma.label().synset().name()
        return lemma_string
    else:
        raise ValueError("Lemma object could not be converted into string. Check Lemma type.")


def lemmastring_to_synsetstring(lemmastring):
    return lemmastring[:lemmastring.find(' ')]


# Testing---------------------------------------------------------------------------------------------------------------

sentence_list, word_sense_dict = extract_sentences()
