from collections import defaultdict
import nltk
from nltk.corpus import semcor
import json
import os.path


class CorpusUtilities:
    """ A small library of functions that assist in working with the Semcor corpus"""
    def __init__(self, num_sentences=-1, partition=1):
        """
        Parameters:
            num_sentences (int): The number of sentences from the corpus to use in the task. The first n sentences
                from the corpus are used and if n=-1, all sentences from the corpus are used.
            partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking
                at sentences 10000 - 14999.
        """
        self.num_sentences = num_sentences
        self.partition = partition
    def lemma_to_tuple(self, lemma):
        """
        Converts lemmas to tuples to prevent usage of the nltk corpus
        Parameters:
            lemma (lemma object) a lemma object from the nltk package
        Returns:
            (tuple) a tuple containing the sense and synset of the word originally in lemma format.
        """
        lemma_word = lemma.name()
        synset_string = lemma.synset().name()
        lemma_tuple = (lemma_word, synset_string)
        return lemma_tuple

    def get_sentence_list(self):
        """
        Gets sentence list from semcor corpus in nltk python module
        Parameters:
            num_sentences (int): The number of sentences from the corpus to use in the task. The first n sentences
                from the corpus are used and if n=-1, all sentences from the corpus are used.
            partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking at
                sentences 10000 - 14999.
        Returns:
            (list) sentence_list (list of all sentences or the first n sentences of the corpus)
        """
        if self.num_sentences == -1:
            sentence_list_path = "./sentence_list/sentence_list.json"
        elif self.num_sentences != -1 and self.partition == 1:
            sentence_list_path = "./sentence_list/sentence_list_" + str(self.num_sentences) + ".json"
        else:
            sentence_list_path = "./sentence_list/sentence_list_" + str(self.num_sentences) + "_partition_" + str(
                self.partition) + ".json"
        if not os.path.isfile(sentence_list_path):
            # Checking that file exists
            sentence_list = []
            if self.num_sentences == -1:
                semcor_sents = semcor.tagged_sents(tag="sem")
            else:
                if self.partition == 1:
                    semcor_sents = semcor.tagged_sents(tag="sem")[0:self.num_sentences]
                elif self.partition * self.num_sentences > 30195:
                    raise ValueError(self.partition, self.num_sentences)
                else:
                    semcor_sents = semcor.tagged_sents(tag="sem")[
                                   (self.num_sentences * (self.partition - 1)):(self.num_sentences * self.partition)]
            for sentence in semcor_sents:
                sentence_word_list = []
                for item in sentence:
                    if not isinstance(item, nltk.Tree):
                        continue
                    if not isinstance(item.label(), nltk.corpus.reader.wordnet.Lemma):
                        continue
                    corpus_word = self.lemma_to_tuple(item.label())
                    sentence_word_list.append(corpus_word)
                if len(sentence_word_list) > 1:
                    sentence_list.append(sentence_word_list)
            sent_list_file = open(sentence_list_path, 'w')
            json.dump(sentence_list, sent_list_file)
            sent_list_file.close()
        else:
            # Getting json file containing the sentence list and converting the words stored as strings into tuples
            sentence_list = json.load(open(sentence_list_path))
            for sentence_index in range(len(sentence_list)):
                for word_index in range(len(sentence_list[sentence_index])):
                    word = sentence_list[sentence_index][word_index]
                    sentence_list[sentence_index][word_index] = tuple(word)
        return sentence_list

    def get_word_counts(self):
        """ Gets the number of times each word (encompassing all senses) occurs in the sentence list"""
        word_counts = defaultdict(int)
        sentence_list = self.get_sentence_list()
        for sentence in sentence_list:
            for sense in sentence:
                word_counts[sense[0]] += 1
        return word_counts

    def get_sense_counts(self):
        """ Gets the number of times each sense-specific word occurs in the sentence list"""
        sense_counts = defaultdict(int)
        sentence_list = self.get_sentence_list()
        for sentence in sentence_list:
            for sense in sentence:
                sense_counts[sense] += 1
        return sense_counts

    def get_word_word_cooccurrences(self):
        """ Creates a symmetric dictionary keys as word/word tuples and values the number of times each occur in the
         same sentence"""
        word_word_cooccurrences = defaultdict(int)
        sentence_list = self.get_sentence_list()
        for sentence in sentence_list:
            for target_index in range(len(sentence)):
                target_sense = sentence[target_index]
                target_word = target_sense[0]
                for other_index in range(len(sentence)):
                    if target_index != other_index:
                        other_sense = sentence[other_index]
                        other_word = other_sense[0]
                        word_word_cooccurrences[(target_word, other_word)] += 1
        return word_word_cooccurrences

    def get_sense_sense_cooccurrences(self):
        """ Creates a symmetric dictionary keys as sense/sense tuples and values the number of times each occur in the
                 same sentence"""
        sense_sense_cooccurrences = defaultdict(int)
        sentence_list = self.get_sentence_list()
        for sentence in sentence_list:
            for target_index in range(len(sentence)):
                target_sense = sentence[target_index]
                for other_index in range(len(sentence)):
                    if target_index != other_index:
                        other_sense = sentence[other_index]
                        sense_sense_cooccurrences[(target_sense, other_sense)] += 1
        return sense_sense_cooccurrences

    def get_sense_word_cooccurrences(self):
        """ Creates a symmetric dictionary keys as sense/word tuples and values the number of times each occur in the
                 same sentence"""
        sense_word_cooccurrences = defaultdict(int)
        sentence_list = self.get_sentence_list()
        for sentence in sentence_list:
            for target_index in range(len(sentence)):
                target_sense = sentence[target_index]
                for other_index in range(len(sentence)):
                    if target_index != other_index:
                        other_sense = sentence[other_index]
                        other_word = other_sense[0]
                        sense_word_cooccurrences[(target_sense, other_word)] += 1
        return sense_word_cooccurrences

    def get_word_sense_dict(self):
        """
        Makes a dictionary with each senseless word the key, and each of its senses the values.
        Returns:
             (dict) dictionary with the possible senses of each word in the corpus
            """
        word_sense_dict = defaultdict(set)
        sentence_list = self.get_sentence_list()
        for sentence in sentence_list:
            temp_word_sense_dict = defaultdict(set)
            for word in sentence:
                temp_word_sense_dict[word[0]].add(word)
            if len(temp_word_sense_dict) > 1:
                for word, senses in temp_word_sense_dict.items():
                    word_sense_dict[word] = set(word_sense_dict[word])
                    word_sense_dict[word] |= senses
                    word_sense_dict[word] = list(word_sense_dict[word])
        return word_sense_dict
