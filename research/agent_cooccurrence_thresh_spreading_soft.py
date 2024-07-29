from corpus_utilities import CorpusUtilities
from sentence_cooccurrence_activation import SentenceCooccurrenceActivation
from agent_spreading import AgentSpreadingCorpus, AgentSpreadingNGrams
from agent_cooccurrence import AgentCooccurrenceCorpus
from sentence_long_term_memory import sentenceLTM
from n_gram_cooccurrence.google_ngrams import GoogleNGram
import os
import json
import math
from collections import defaultdict


class AgentSoftCoocThreshSpreadingCorpus(AgentSpreadingCorpus):
    """
    Implements the soft cooccurrence thresholded spreading agent for WSD on SemCor corpus.
    Allows spreading to occur to elements that are cooccurrently and semantically related
     to the target word.
    """

    def __init__(self, context_type, corpus_utilities, outside_corpus, spreading=True, activation_base=2,
                 decay_parameter=0.05, constant_offset=0):
        super().__init__(corpus_utilities, outside_corpus, spreading, activation_base, decay_parameter, constant_offset)
        self.context_type = context_type
        self.sentence_list = self.corpus_utilities.get_sentence_list()
        self.word_sense_dict = self.corpus_utilities.get_word_sense_dict()
        self.cooc_network = self.create_cooc_network()


    def get_cooc_relations_dict(self):
        """
        Creates a dictionary of cooccurrence relations for every word in the corpus.
        Parameters:
              sentence_list (nested list): A list of sentences from the Semcor corpus.
              context_type (string): Indicates for cooccurrence dependent corpus mechanisms, whether we know the sense
                of the context words ("sense") or not ("word")
        Returns:
            (dict) Dictionary where each word in the corpus is a key and each of the other words it cooccurs with
                (are in the same sentence as our target word) are values (in a set).
        """
        cooc_rel_dict = defaultdict(set)
        length = len(self.sentence_list)
        counter = 0
        for sent in self.sentence_list:
            counter += 1
            if counter % 50 == 0:
                print(counter, "out of", length, "in coocreldict")
            for index in range(len(sent)):
                for context_index in range(len(sent)):
                    if index != context_index:
                        target_sense = sent[index]
                        context_sense = sent[context_index]
                        if self.context_type == "sense":
                            cooc_rel_dict[target_sense].update([context_sense])
                        else:
                            context_word = context_sense[0]
                            cooc_rel_dict[target_sense].update([context_word])
        return cooc_rel_dict

    def create_cooc_network(self):
        """Creates a network based on cooccurrences, with each """
        network = sentenceLTM(
            activation_cls=(lambda ltm:
                            SentenceCooccurrenceActivation(
                                ltm,
                                activation_base=self.activation_base,
                                constant_offset=self.constant_offset,
                                decay_parameter=self.decay_parameter
                            )))
        cooc_rel_dict = self.get_cooc_relations_dict()
        rel_keys = list(cooc_rel_dict.keys())
        length = len(rel_keys)
        counter = 0
        for key in rel_keys:
            counter += 1
            if counter % 50 == 0:
                print(counter, "out of", length, "stored")
            rels = cooc_rel_dict[key]
            if self.context_type == "word":
                # Adjusts the relations dictionary to get all senses for each word.
                new_rels = []  # List to store all senses that fit each word
                for rel in rels:
                    new_rels.extend(self.word_sense_dict[rel])
                rels = new_rels
            network.store(mem_id=key,
                          time=1,
                          spread_depth=1,
                          assocs=list(rels))
        return network



    def clear_sem_network(self, start_time=1):
        """
            Clears the semantic network by resetting activations to a certain "starting time".
            Parameters:
                sem_network (sentenceLTM): Network to clear.
                start_time (int): The network will be reset so that activations only at the starting time and before the
                    starting time remain.
            Returns:
                sentenceLTM: Cleared semantic network.
        """
        activations = self.network.activation.activations
        cooc_activations = self.cooc_network.activation.activations
        if start_time > 0:
            activated_words = activations.keys()
            cooc_activated_words = cooc_activations.keys()
            for word in activated_words:
                activations[word] = [act for act in activations[word] if act[0] <= start_time]
            for word in cooc_activated_words:
                cooc_activations[word] = [act for act in cooc_activations[word] if act[0] <= start_time]
            self.network.activation.activations = activations
            self.cooc_network.activation.activations = cooc_activations
        elif start_time == 0:
            self.network.activation.activations = defaultdict(list)
            self.cooc_network.activation.activations = defaultdict(list)
        else:
            raise ValueError(start_time)

    def get_sem_cooc_activation(self, elem, time):
        """ Gets the summed activation from the cooccurrence and spreading networks."""
        return math.exp(self.network.get_activation(elem, time)) + math.exp(
            self.cooc_network.get_activation(elem, time))

    def do_wsd(self, word, context, time):
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        max_act = float('-inf')
        max_guess = []
        for candidate in context:
            candidate_act = self.get_sem_cooc_activation(word, time)
            if candidate_act > max_act:
                max_act = candidate_act
                max_guess = [candidate]
            elif candidate_act == max_act:
                max_guess.append(candidate)
        self.network.store(mem_id=word, time=time, spread_depth=spread_depth)
        self.cooc_network.store(mem_id=word, time=time, spread_depth=1)
        for elem in max_guess:
            self.network.store(mem_id=elem, time=time, spread_depth=spread_depth)
            self.cooc_network.store(mem_id=elem, time=time, spread_depth=1)
        return max_guess


class AgentSoftCooccurrenceThreshSpreadingNGrams(AgentSpreadingNGrams):
    """
    Implements the soft cooccurrence thresholded spreading agent for RAT on NGrams corpus.
    Allows spreading to occur to elements that are cooccurrently and semantically related
     to the target word.
    """

    def __init__(self, stopwords, source, ngrams=GoogleNGram('~/ngram'), spreading=True, clear="never",
                 activation_base=2.0, decay_parameter=0.05, constant_offset=0.0):
        super().__init__(stopwords=stopwords, source=source, spreading=spreading, clear=clear,
                         activation_base=activation_base, decay_parameter=decay_parameter, constant_offset=constant_offset)
        self.ngrams = ngrams
        self.cooc_network = self.create_cooc_network()

    def clear_sem_network(self, start_time=1):
        """
            Clears the semantic network by resetting activations to a certain "starting time".
            Parameters:
                sem_network (sentenceLTM): Network to clear.
                start_time (int): The network will be reset so that activations only at the starting time and before the
                    starting time remain.
            Returns:
                sentenceLTM: Cleared semantic network.
        """
        activations = self.network.activation.activations
        cooc_activations = self.cooc_network.activation.activations
        if start_time > 0:
            activated_words = activations.keys()
            cooc_activated_words = cooc_activations.keys()
            for word in activated_words:
                activations[word] = [act for act in activations[word] if act[0] <= start_time]
            for word in cooc_activated_words:
                cooc_activations[word] = [act for act in cooc_activations[word] if act[0] <= start_time]
            self.network.activation.activations = activations
            self.cooc_network.activation.activations = cooc_activations
        elif start_time == 0:
            self.network.activation.activations = defaultdict(list)
            self.cooc_network.activation.activations = defaultdict(list)
        else:
            raise ValueError(start_time)

    def create_cooc_network(self):
        """ Creates cooccurrence relations network from a cooccurrence relations cache stored from google ngrams. """
        cooc_rel_dict = json.load(open("./n_gram_cooccurrence/ngrams_simple_RAT_cooccurrence_rels_cache.json"))  # FIXME
        network = sentenceLTM(
            activation_cls=(lambda ltm:
                            SentenceCooccurrenceActivation(
                                ltm,
                                activation_base=self.activation_base,
                                constant_offset=self.constant_offset,
                                decay_parameter=self.decay_parameter
                            )))
        for word in cooc_rel_dict.keys():
            network.store(mem_id=word.upper(),
                          time=1,
                          activate=False,
                          assocs=cooc_rel_dict[word])
        return network

    def get_sem_cooc_activation(self, word, time):
        """ Gets the summed activation from the cooccurrence and spreading networks."""
        spread_act = self.network.get_activation(word, time)
        cooc_act = self.cooc_network.get_activation(word, time)
        if spread_act is None and cooc_act is None:
            return 0
        elif spread_act is None:
            return math.exp(cooc_act)
        elif cooc_act is None:
            return math.exp(spread_act)
        return math.exp(spread_act) + math.exp(cooc_act)

    def do_rat(self, context1, context2, context3):
        """
        Completes one trial of the RAT.
        Parameters:
            context1, context2, context3 (string): Context words to be used in the RAT task.
        Returns:
            A list of RAT guesses. Returns [] if there are no viable guesses.
        """
        if self.spreading:
            spread_depth = -1
        else:
            spread_depth = 0
        context_list = [context1.upper(), context2.upper(), context3.upper()]
        for context in context_list:
            self.network.store(mem_id=context.upper(), time=2, spread_depth=spread_depth)
            self.cooc_network.store(mem_id=context.upper(), time=2, spread_depth=1)
        max_act = -float("inf")
        guesses = []
        elements = sorted(set(self.network.activation.activations.keys()))
        for elem in elements:
            if elem in context_list:
                continue
            elem_act = self.get_sem_cooc_activation(elem.upper(), 3)
            if elem_act is None:
                continue
            elif elem_act > max_act:
                max_act = elem_act
                guesses = [elem]
            elif elem_act == max_act:
                guesses.append(elem)
        return guesses
