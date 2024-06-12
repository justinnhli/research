""" Implements the "all knowing" and "all-mechanism" oracle agent."""

from agent_cooccurrence import *
from agent_spreading import *
from n_gram_cooccurrence.google_ngrams import *


class AgentOracle:
    """ Implements the "upper bound" oracle agent. """

    def __init__(self, corpus_utilities):
        """
        Parameters:
            corpus_utilities (class): A class of functions useful for corpus mechanisms, specific to the partition of the
                Semcor corpus used
        """
        self.corpus_utilities = corpus_utilities
        self.num_sentences = corpus_utilities.num_sentences
        self.partition = corpus_utilities.partition

    def do_wsd(self, target_index, sentence, timer_word, timer_sentence, timer_never):
        """
        Completes a trial of the WSD.
        Parameters:
            target_index (int): The index of the "target" word in the sentence given in the sentence parameter list.
            sentence (list): A list of lemma/synset tuples referring to all words in the sentence (including the target
                sense)
            timer_word (int): Timer for the network that clears after every word.
            timer_sentence (int): Timer for the network that clears after every sentence.
            timer_never (int): Timer for the network that never clears.
        """
        raise NotImplementedError

    def do_rat(self):
        """ Runs a trial of the RAT."""
        raise NotImplementedError


class AgentOracleCorpus(AgentOracle):

    def __init__(self, corpus_utilities, outside_corpus=False, activation_base=2,
                 decay_parameter=0.05, constant_offset=0):
        """
        Parameters:
            corpus_utilities (class): A class of functions useful for corpus mechanisms, specific to the partition of the
                Semcor corpus used
            outside_corpus (bool): True if semantic relations can be considered outside the corpus and False if semantic
            activation_base (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            constant_offset (float): A parameter in the activation equation.
        """
        super().__init__(corpus_utilities)
        self.num_sentences = corpus_utilities.num_sentences
        self.partition = corpus_utilities.partition
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter
        self.constant_offset = constant_offset
        self.sentence_list = corpus_utilities.get_sentence_list()
        self.word_sense_dict = corpus_utilities.get_word_sense_dict()
        self.outside_corpus = outside_corpus
        self.sem_nospread_agent = AgentSpreadingCorpus(corpus_utilities=corpus_utilities, outside_corpus=outside_corpus,
                                                       spreading=False,
                                                       clear="never", activation_base=activation_base,
                                                       decay_parameter=decay_parameter, constant_offset=constant_offset)
        self.sem_nospread_network = self.sem_nospread_agent.create_sem_network()
        self.sem_never_agent = AgentSpreadingCorpus(corpus_utilities=corpus_utilities, outside_corpus=outside_corpus,
                                                    spreading=True, clear="never",
                                                    activation_base=activation_base, decay_parameter=decay_parameter,
                                                    constant_offset=constant_offset)
        self.sem_never_network = self.sem_never_agent.create_sem_network()
        self.sem_sentence_agent = AgentSpreadingCorpus(corpus_utilities=corpus_utilities, outside_corpus=outside_corpus,
                                                       spreading=True,
                                                       clear="sentence", activation_base=activation_base,
                                                       decay_parameter=decay_parameter, constant_offset=constant_offset)
        self.sem_sentence_network = self.sem_sentence_agent.create_sem_network()
        self.sem_word_agent = AgentSpreadingCorpus(corpus_utilities=corpus_utilities, outside_corpus=outside_corpus,
                                                   spreading=True, clear="word",
                                                   activation_base=activation_base, decay_parameter=decay_parameter,
                                                   constant_offset=constant_offset)
        self.sem_word_network = self.sem_word_agent.create_sem_network()
        self.cooc_word_agent = AgentCooccurrenceCorpus(num_sentences=corpus_utilities.num_sentences,
                                                       partition=corpus_utilities.partition,
                                                       corpus_utilities=corpus_utilities,
                                                       context_type="word")
        self.cooc_sense_agent = AgentCooccurrenceCorpus(num_sentences=corpus_utilities.num_sentences,
                                                        partition=corpus_utilities.partition,
                                                        corpus_utilities=corpus_utilities,
                                                        context_type="sense")

    def do_wsd(self, target_index, sentence, timer_word, timer_sentence, timer_never):
        """
        Completes a trial of the WSD.
        Parameters:
            target_index (int): The index of the "target" word in the sentence given in the sentence parameter list.
            sentence (list): A list of lemma/synset tuples referring to all words in the sentence (including the target
                sense)
            timer_word (int): Timer for the network that clears after every word.
            timer_sentence (int): Timer for the network that clears after every sentence.
            timer_never (int): Timer for the network that never clears.
        """
        word = sentence[target_index]
        word_senses = self.word_sense_dict[word[0]]
        correct_sense = sentence[target_index]
        cooc_word_guess = self.cooc_word_agent.do_wsd(target_index, sentence)
        if correct_sense in cooc_word_guess:
            return [correct_sense]
        cooc_sense_guess = self.cooc_sense_agent.do_wsd(target_index, sentence)
        if correct_sense in cooc_sense_guess:
            return [correct_sense]
        sem_guess_no_spread = self.sem_nospread_agent.do_wsd(word=word, context=word_senses, time=timer_never,
                                                             network=self.sem_nospread_network)
        if correct_sense in sem_guess_no_spread:
            return [correct_sense]
        sem_guess_never = self.sem_never_agent.do_wsd(word=word, context=word_senses, time=timer_never,
                                                      network=self.sem_never_network)
        if correct_sense in sem_guess_never:
            return [correct_sense]
        sem_guess_sentence = self.sem_sentence_agent.do_wsd(word=word, context=word_senses, time=timer_never,
                                                            network=self.sem_sentence_network)
        if correct_sense in sem_guess_sentence:
            return [correct_sense]
        sem_guess_word = self.sem_word_agent.do_wsd(word=word, context=word_senses, time=timer_never,
                                                    network=self.sem_word_network)
        if correct_sense in sem_guess_word:
            return [correct_sense]
        else:
            return [None]
