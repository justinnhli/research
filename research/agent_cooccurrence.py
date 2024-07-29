""" Implements cooccurrence agent(s) and nltk_converter classes. """

from collections import defaultdict
from n_gram_cooccurrence.google_ngrams import *
import json

class AgentCooccurrence:
    """ General cooccurrence agent superclass. """

    def __init__(self):
        pass

    def get_count(self, *events):
        """ Gets the number of times each word occurred, or several words occurred together in context"""
        raise NotImplementedError()

    def get_conditional_probability(self, target, base):
        """
        Gets the conditional probability of seeing a particular word given the context.
        Parameters:
            target (varies): The word of interest.
            base (varies): The context.
        Returns:
            (float) decimal conditional probability.
        """
        joint_count = self.get_count(base, target)
        base_count = self.get_count(base)
        return joint_count / base_count

    def do_wsd(self, target_index, sentence):
        """ Completes the WSD task """
        raise NotImplementedError()

    def do_rat(self, context1, context2, context3):
        """ Completes the RAT task. """
        raise NotImplementedError()


class AgentCooccurrenceCorpus(AgentCooccurrence):
    """ Cooccurrence Agent for a corpus cooccurrence source. """

    def __init__(self, num_sentences, partition, corpus_utilities, context_type):
        """
        Parameters:
            num_sentences (int): The number of sentences from the corpus to use in the task. The first n sentences
                from the corpus are used and if n=-1, all sentences from the corpus are used.
            partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking
                at sentences 10000 - 14999.
            corpus_utilities (class): A class of functions useful for corpus mechanisms, specific to the partition of
                the Semcor corpus used
            context_type (string): Indicates for cooccurrence dependent corpus mechanisms, whether we know the sense of
                the context words ("sense") or not ("word")
        """
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

    def get_count(self, *events):
        """
        Gets the counts of a single returned element, or two different elements for computing the conditional
        probability.
        Parameters:
            events (list): events to get the counts of.
        Returns:
            (int) counts of the events
        """
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
            if type(event1) == tuple and type(event2) == tuple:  # Two senses
                if (event1, event2) not in self.sense_sense_cooccurrences.keys():
                    return 0
                return self.sense_sense_cooccurrences[(event1, event2)]
            elif type(event1) == tuple and type(event2) != tuple:  # event1 is a sense, event2 is a word
                if (event1, event2) not in self.sense_word_cooccurrences.keys():
                    return 0
                return self.sense_word_cooccurrences[(event1, event2)]
            elif type(event1) != tuple and type(event2) == tuple:  # event1 is a word, event2 is a sense
                if (event2, event1) not in self.sense_word_cooccurrences.keys():
                    return 0
                return self.sense_word_cooccurrences[(event2, event1)]
            else:  # event1 and event2 are words
                if (event1, event2) not in self.word_word_cooccurrences.keys():
                    return 0
                return self.word_word_cooccurrences[(event1, event2)]
        else:
            raise ValueError(events)

    def get_conditional_probability(self, target, base):
        """
        Gets conditional probability
        Parameters:
            target (tuple): Word of interest. Assumes that target is a sense (aka is formatted as a (word, sense) tuple)
            base (tuple or string): Context, can be a sense (described above) or a word (just a string, no sense
                information).
        Returns:
            (float) Decimal conditional probability
        """
        joint_count = self.get_count(base, target)  # How many times target SENSE & context cooccur
        base_count = self.get_count(base, target[0])  # How many times target WORD & context cooccur
        if base_count == 0:
            return 0
        return joint_count / base_count

    def do_wsd(self, target_index, sentence):
        """
        Completes the WSD task.
        Parameters:
            target_index (int): Integer >= 0 corresponding to the index of the list of sentence words where the target
                sense can be found.
            sentence (list): List of words in the current sentence from the SemCor corpus.
        Returns:
            (list) A list of word sense disambiguation sense guesses.
        """
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


class AgentCooccurrenceNGrams(AgentCooccurrence):
    """ Cooccurrence agent for a ngrams cooccurrence source. """

    def __init__(self, stopwords, ngrams=GoogleNGram('~/ngram')):
        """
        Parameters:
            stopwords (list): A list of stopwords - common words to not include semantic relations to.
            ngrams (class): Instance of the GoogleNGram class.
        """
        super().__init__()
        self.ngrams = ngrams
        self.stopwords = stopwords
        self.cooc_cache = self.get_cooccurrence_cache_dict()


    def get_cooccurrence_cache_dict(self):
        cooc_cache = json.load(open("./n_gram_cooccurrence/ngrams_cooccurrence_cache.json"))
        cooc_cache_dict = defaultdict(list)
        for entry in cooc_cache:
            key = tuple([entry[0][0].upper(), entry[0][1].upper(), entry[0][2].upper()])
            cooc_elements = entry[1]
            vals = []
            for elem in cooc_elements:
                val = elem[0]
                counts = elem[1]
                vals.append([val, counts])
            cooc_cache_dict[key] = vals
        return cooc_cache_dict

    def get_word_counts(self, word):
        """
        Returns the number of times the word occurred in the ngrams corpus.
        Assumes merge_variants (whether different capitilizations should be considered the same word) to be true.
        Parameters:
            word (string): Word to get the counts of.
        Returns:
            (int): counts
        """
        return self.ngrams.get_ngram_counts(word)[word]

    def get_all_word_cooccurrences(self, word):
        """
        Finds all words that cooccur with a word of interest.
        Parameters:
            word (string): word of interest.
        Returns:
            (list) ordered list (most to least # of times word occurs) of tuples formatted as
        (word, # times word occcurred) for words that cooccur with the input word.
         """
        return self.ngrams.get_max_probability(word)

    def get_conditional_probability(self, word, context):
        """
        Gets the conditional probability of seeing a particular word given the context.
        Parameters:
            target (varies): The word of interest.
            base (varies): The context.
        Returns:
            (float) decimal conditional probability.
        """
        return self.ngrams.get_conditional_probability(base=context, target=word)

    def do_rat(self, context1, context2, context3):
        """
        Completes one trial of the RAT.
        Parameters:
            context1, context2, context3 (string): Context words to be used in the RAT task.
        Returns:
            A list of RAT guesses. Returns [] if there are no viable guesses.
        """
        joint_cooc_set = self.cooc_cache[tuple([context1.upper(), context2.upper(), context3.upper()])]
        if len(joint_cooc_set) == 0:
            return []
        elif len(joint_cooc_set) == 1:
            return list(joint_cooc_set)[0][0]
        else:
            max_cond_prob = -float("inf")
            max_elems = []
            for elem in list(joint_cooc_set):
                cooc_word = elem[0]
                if cooc_word.lower() in self.stopwords:
                    continue
                joint_cond_prob = elem[1]
                if joint_cond_prob > max_cond_prob:
                    max_cond_prob = joint_cond_prob
                    max_elems = [cooc_word]
                elif joint_cond_prob == max_cond_prob:
                    max_elems.append(cooc_word)
            return max_elems

# Testing ... ----------------------------------------------------------------------------------------------------------

