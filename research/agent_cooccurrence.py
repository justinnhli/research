""" Implements cooccurrence agent(s) and nltk_converter classes. """

from collections import defaultdict
from n_gram_cooccurrence.google_ngrams import *

class AgentCooccurrence:
    """ General cooccurrence agent superclass. """

    def __init__(self):
        pass

    def get_count(self, *events):
        raise NotImplementedError()

    def get_conditional_probability(self, target, base):
        joint_count = self.get_count(base, target)
        base_count = self.get_count(base)
        return joint_count / base_count

    def do_wsd(self, target_index, sentence):
        raise NotImplementedError()

    def do_rat(self, context1, context2, context3):
        raise NotImplementedError()


class AgentCooccurrenceCorpus(AgentCooccurrence):
    """ Cooccurrence Agent for a corpus cooccurrence source. """

    def __init__(self, num_sentences, partition, corpus_utilities, context_type):
        "cooc_source is a nested list of sentences, which are themselves lists containing sense-specified words"
        super().__init__()
        self.num_sentences = num_sentences
        self.partition = partition
        self.corpus_utilities = corpus_utilities
        self.context_type = context_type
        self.sentence_list = corpus_utilities.get_sentence_list()
        self.word_sense_dict = corpus_utilities.get_word_sense_dict()
        self.sense_counts = corpus_utilities.get_sense_counts()
        self.word_counts = corpus_utilities.get_word_counts()
        self.sense_sense_cooccurrences = corpus_utilities.get_sense_sense_cooccurrences()
        self.sense_word_cooccurrences = corpus_utilities.get_sense_word_cooccurrences()
        self.word_word_cooccurrences = corpus_utilities.get_word_word_cooccurrences()

    def do_wsd(self, target_index, sentence):
        """ Gets a guess for the WSD task, assuming that either sense-specific information is known about the context
        (context_type = "sense") or not (context_type = "word")"""
        max_score = -float("inf")
        max_senses = None
        target_sense = sentence[target_index]
        if self.context_type == "word":
            sentence = [word[0] for word in sentence]
        for target_sense_candidate in self.word_sense_dict[target_sense[0]]:
            candidate_conditional_probability = 0
            for context_index in range(len(sentence)):
                if context_index != target_index:
                    context_word = sentence[context_index]
                    candidate_conditional_probability += self.get_conditional_probability(target_sense_candidate,
                                                                                          context_word)
            if candidate_conditional_probability > max_score:
                max_score = candidate_conditional_probability
                max_senses = [target_sense_candidate]
            elif candidate_conditional_probability == max_score:
                max_senses.append(target_sense_candidate)
        return max_senses

    def get_count(self, *events):
        """ Gets the counts of a single returned element, or two different elements for computing the conditional
        probability"""
        if len(events) == 0:
            raise ValueError(events)
        if len(events) == 1:
            event = events[0]
            if type(event) == tuple:  # Context type = sense
                return self.sense_counts[event]
            else:  # Context type = word
                return self.word_counts[event]
        elif len(events) == 2:
            event1 = events[0]
            event2 = events[1]
            if type(event1) == tuple and type(event2) == tuple:  # Context type = sense
                if (event1, event2) not in self.sense_sense_cooccurrences.keys():
                    return 0
                return self.sense_sense_cooccurrences[(event1, event2)]
            elif type(event1) == tuple and type(event2) != tuple:
                if (event1, event2) not in self.sense_word_cooccurrences.keys():
                    return 0
                return self.sense_word_cooccurrences[(event1, event2)]
            elif type(event1) != tuple and type(event2) == tuple:
                if (event2, event1) not in self.sense_word_cooccurrences.keys():
                    return 0
                return self.sense_word_cooccurrences[(event2, event1)]
            else:
                if (event1, event2) not in self.word_word_cooccurrences.keys():
                    return 0
                return self.word_word_cooccurrences[(event1, event2)]
        else:
            raise ValueError(events)

    def get_conditional_probability(self, target, base):
        """ Assumes that target is a sense (aka is formatted as a (word, sense) tuple"""
        joint_count = self.get_count(base, target)
        base_count = self.get_count(base, target[0])
        if base_count == 0:
            return 0
        return joint_count / base_count


class AgentCooccurrenceNGrams(AgentCooccurrence):
    """ Cooccurrence agent for a ngrams cooccurrence source. """

    def __init__(self, stopwords, ngrams=GoogleNGram('~/ngram')):
        """
        stopwords is a python list of stopwords from a downloaded file.
        """
        super().__init__()
        self.ngrams = ngrams
        self.stopwords = stopwords

    def get_word_counts(self, word):
        """ Returns the number of times the word occurred in the ngrams corpus.
            Assumes merge_variants (whether different capilizations should be considered the same word) to be true"""
        return self.ngrams.get_ngram_counts(word)[word]

    def get_all_word_cooccurrences(self, word):
        """ Returns an ordered list (most to least # of times word occurs) of tuples formatted as
         (word, # times word occcurred) for words that cooccur with the input word"""
        return self.ngrams.get_max_probability(word)

    def get_conditional_probability(self, word, context):
        """ Assumes that the context is only one word (acting as the base parameter for the Google ngrams class)"""
        return self.ngrams.get_conditional_probability(base=context, target=word)

    def do_rat(self, context1, context2, context3):
        cooc_set1 = set([elem[0] for elem in self.ngrams.get_max_probability(context1)])
        cooc_set2 = set([elem[0] for elem in self.ngrams.get_max_probability(context2)])
        cooc_set3 = set([elem[0] for elem in self.ngrams.get_max_probability(context3)])
        joint_cooc_set = cooc_set1 & cooc_set2 & cooc_set3
        if len(joint_cooc_set) == 0:
            return None
        elif len(joint_cooc_set) == 1:
            return joint_cooc_set.pop()
        else:
            max_cond_prob = -float("inf")
            max_elems = []
            for elem in list(joint_cooc_set):
                if elem.lower() in self.stopwords:
                    continue
                cond_prob1 = self.ngrams.get_conditional_probability(base=context1, target=elem)
                cond_prob2 = self.ngrams.get_conditional_probability(base=context2, target=elem)
                cond_prob3 = self.ngrams.get_conditional_probability(base=context3, target=elem)
                joint_cond_prob = cond_prob1 * cond_prob2 * cond_prob3
                if joint_cond_prob > max_cond_prob:
                    max_cond_prob = joint_cond_prob
                    max_elems = [elem]
                elif joint_cond_prob == max_cond_prob:
                    max_elems.append(elem)
            return max_elems

# Testing ... ----------------------------------------------------------------------------------------------------------

