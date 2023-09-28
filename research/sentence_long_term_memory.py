import math
from research import ActivationDynamics
from research import NaiveDictLTM
from collections import defaultdict
from research.sentence_cooccurrence_activation import SentenceCooccurrenceActivation


class sentenceLTM(NaiveDictLTM):

    def __init__(self, activation_cls=SentenceCooccurrenceActivation, **kwargs):
        #super.__init__(**kwargs)
        super().__init__(**kwargs)
        self.activation = activation_cls(self)
        self.cooccurrent_elements = {}
    def activate_sentence(self, sentence_words, time=0):
        self.activation.activate_sentence(sentence_words, time)
        return None

    def guess_word_sense(self, word_to_guess, sentence_words):
        word_sense_cooccurrence = self.activation.get_word_sense_cooccurrence(word_to_guess, sentence_words)
        max_cooccurrence_count = -1
        guess = ""
        for key in word_sense_cooccurrence.keys():
            if word_sense_cooccurrence[key] > max_cooccurrence_count:
                max_cooccurrence_count = word_sense_cooccurrence[key]
                guess = key
        return guess
