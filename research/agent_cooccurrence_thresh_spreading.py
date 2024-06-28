from corpus_utilities import CorpusUtilities
from sentence_cooccurrence_activation import SentenceCooccurrenceActivation
from agent_spreading import AgentSpreadingCorpus, AgentSpreadingNGrams
from agent_cooccurrence import AgentCooccurrenceCorpus
from sentence_long_term_memory import sentenceLTM
from n_gram_cooccurrence.google_ngrams import *
import time
import json


class AgentCoocThreshSpreadingCorpus(AgentSpreadingCorpus, AgentCooccurrenceCorpus):
    """
    Implements cooccurrence thresholded spreading agent.
    Only allows spreading to occur for elements that also cooccur.
     """

    def __init__(self, context_type, whole_corpus, corpus_utilities, outside_corpus, spreading=True, clear="never",
                 activation_base=2, decay_parameter=0.05, constant_offset=0, threshold=0.0):
        """
        Parameters:
            context_type (string): Indicates for cooccurrence dependent corpus mechanisms, whether we know the sense of the
                context words ("sense") or not ("word")
            whole_corpus (bool): For cooccurrence dependent corpus mechanisms, whether to include cooccurrent relations from
                the whole corpus (True) or not (False).
            corpus_utilities (class): A class of functions useful for corpus mechanisms, specific to the partition of the
                Semcor corpus used
            outside_corpus (bool): True if semantic relations can be considered outside the corpus and False if semantic
            activation_base (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            constant_offset (float): A parameter in the activation equation.
            spreading (bool): Whether to include the effects of spreading in the semantic network.
        """
        super().__init__(corpus_utilities, outside_corpus, spreading, clear, activation_base, decay_parameter,
                         constant_offset)
        self.sense_word_cooccurrences = corpus_utilities.get_sense_word_cooccurrences()
        self.sense_sense_cooccurrences = corpus_utilities.get_sense_sense_cooccurrences()
        self.word_counts = corpus_utilities.get_word_counts()
        self.sense_counts = corpus_utilities.get_sense_counts()
        self.context_type = context_type
        self.whole_corpus = whole_corpus
        self.threshold = threshold

    def adjust_sem_rel_dict(self):
        """
        Adjusts the semantic relations dictionary to only include words that also cooccur with each word (as key).
        Parameters:
            sem_rel_dict (dict): A nested dictionary with each sense-specific word the key, and values the different
                semantic categories (synonyms, hyponyms, etc.) that the various sense-specific semantically related
                words are included in.
        Returns:
            (dict) Cooccurrence adjusted semantic relations dictionary.
        """
        link = "./semantic_relations_lists/semantic_relations_list"
        sent_list = self.corpus_utilities.get_sentence_list()
        if self.whole_corpus:
            whole_cu = CorpusUtilities()
            all_sents_list = whole_cu.get_sentence_list()
            cooc_rel_dict = self.create_cooc_relations_dict(all_sents_list, context_type=self.context_type)
        else:
            cooc_rel_dict = self.create_cooc_relations_dict(sent_list, context_type=self.context_type)
        for word_key in sem_rel_dict.keys():
            word_rel_dict = sem_rel_dict[word_key]  # has all different relations to target word
            for cat in word_rel_dict.keys():  # looping through each relation category
                rels = word_rel_dict[cat]  # getting the relations in that category
                new_rels = []
                for rel in rels:  # going through words corresponding to each relation
                    if self.context_type == "sense":
                        base = rel
                    else:
                        base = rel[0]
                    if base in list(cooc_rel_dict[word_key]):
                        if self.threshold == 0:
                            new_rels.append(rel)
                        if self.get_conditional_probability(target=word_key, base=base) > self.threshold:
                            new_rels.append(rel)
                sem_rel_dict[word_key][cat] = new_rels
        return sem_rel_dict

    def create_cooc_relations_dict(self, sentence_list, context_type):
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
        for sent in sentence_list:
            for index in range(len(sent)):
                for context_index in range(len(sent)):
                    if index != context_index:
                        target_sense = sent[index]
                        context_sense = sent[context_index]
                        if context_type == "sense":
                            cooc_rel_dict[target_sense].update([context_sense])
                        else:
                            context_word = context_sense[0]
                            cooc_rel_dict[target_sense].update([context_word])
        return cooc_rel_dict

    def create_sem_network(self):
        """ Builds corpus semantic network. """
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        semantic_relations_dict = self.get_semantic_relations_dict(self.outside_corpus)
        semantic_relations_dict = self.adjust_sem_rel_dict(semantic_relations_dict)
        network = sentenceLTM(
            activation_cls=(lambda ltm:
                            SentenceCooccurrenceActivation(
                                ltm,
                                activation_base=self.activation_base,
                                constant_offset=self.constant_offset,
                                decay_parameter=self.decay_parameter
                            )))
        relations_keys = sorted(list(set(semantic_relations_dict.keys())))
        for word_index in range(len(relations_keys)):
            word_key = relations_keys[word_index]
            val_dict = semantic_relations_dict[word_key]
            network.store(mem_id=word_key,
                          time=1,
                          spread_depth=spread_depth,
                          synonyms=val_dict['synonyms'],
                          hypernyms=val_dict['hypernyms'],
                          hyponyms=val_dict['hyponyms'],
                          holynyms=val_dict['holonyms'],
                          meronyms=val_dict['meronyms'],
                          attributes=val_dict['attributes'],
                          entailments=val_dict['entailments'],
                          causes=val_dict['causes'],
                          also_sees=val_dict['also_sees'],
                          verb_groups=val_dict['verb_groups'],
                          similar_tos=val_dict['similar_tos'])
        return network


class AgentCoocThreshSpreadingNGrams(AgentSpreadingNGrams):
    """ Agent to implement cooccurrence thresholded spreading on google ngrams """

    def __init__(self, sem_rel_dict, stopwords, ngrams=GoogleNGram('~/ngram'), spreading=True, clear="never",
                 activation_base=2,
                 decay_parameter=0.05,
                 constant_offset=0):
        """
        Parameters:
            sem_rel_dict (dictionary): A dictionary containing all semantic relations (the values) for each word
                (the keys) from the SWOWEN, SFFAN, (or both) databases.
            stopwords (list): A list of stopwords - common words to not include semantic relations to.
            ngrams (class): Instance of the GoogleNGram class.
            spreading (bool): Whether to include the effects of spreading in creating the semantic network.
            clear (string): How often to clear the network. Possible values are "never", "trial",
                indicating that the network is never cleared, or cleared after each RAT trial.
            activation_base (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            constant_offset (float): A parameter in the activation equation.
        """
        super().__init__(sem_rel_dict=sem_rel_dict, spreading=spreading, clear=clear, activation_base=activation_base,
                         stopwords=stopwords,
                         decay_parameter=decay_parameter, constant_offset=constant_offset)
        self.ngrams = ngrams
        # self.sem_rel_dict = self.adjust_sem_rel_dict(sem_rel_dict)
        self.sem_rel_dict = sem_rel_dict

    def get_cooccurring_words(self, word):
        """
        Gets the words that cooccur with a given input word in google ngrams
        Parameters:
            word (string): word of interest
        Returns:
             (list) ordered list (most to least # of times word occurs) of tuples formatted as
             (word, # times word occcurred) for words that cooccur with the input word"""
        cooc_words_counts = self.ngrams.get_max_probability(word)
        cooc_words = [word[0] for word in cooc_words_counts if word not in self.stopwords]
        return cooc_words


    def adjust_sem_rel_dict(self, sem_rel_dict, save_dict=None):
        """
        Adjusts the semantic relations dictionary to only include words that also cooccur with each word (as key).
        Parameters:
            sem_rel_dict (dict): A dictionary with all semantic relations of interest - from SWOWEN, SFFAN, or both.
            save_dict (str): A link to save the adjusted dictionary to. If none, nothing is saved.
        Returns:
            (dict) Cooccurrence adjusted semantic relations dictionary.
        """
        thresh_sem_rel_dict = defaultdict(list)
        total = len(list(sem_rel_dict.keys()))
        counter = 0
        for word_key in sorted(list(sem_rel_dict.keys())):
            counter += 1
            # if counter % 50 == 0:
            #     print(counter, "out of", total)
            if word_key.lower() in self.stopwords or word_key.count(" ") > 0:
                # Making sure we're only looking at bigrams
                continue
            rels = sem_rel_dict[word_key]  # has all different relations to target word
            new_rels = []
            for rel in rels:  # going through words corresponding to each relation
                if rel.lower() in self.stopwords or rel.count(" ") != 0:
                    # Making sure we're only looking at bigrams
                    continue
                counts = self.ngrams.get_ngram_counts(word_key + " " + rel)[word_key + " " + rel] + \
                         self.ngrams.get_ngram_counts(rel + " " + word_key)[rel + " " + word_key]
                if counts != 0:
                    new_rels.append(rel.upper())
            if new_rels:
                thresh_sem_rel_dict[word_key.upper()] = new_rels
        if save_dict:
            file = open(save_dict, 'w')
            json.dump(thresh_sem_rel_dict, file)
            file.close()
        return thresh_sem_rel_dict

    def create_sem_network(self):
        """ Builds a semantic network. """
        if self.spreading:
            spread_depth = -1
        else:
            spread_depth = 0
        network = sentenceLTM(
            activation_cls=(lambda ltm:
                            SentenceCooccurrenceActivation(
                                ltm,
                                activation_base=self.activation_base,
                                constant_offset=self.constant_offset,
                                decay_parameter=self.decay_parameter
                            )))
        keys = sorted(list(self.sem_rel_dict.keys()))
        key_length = len(keys)
        counter = 0
        for word in keys:
            counter += 1
            # if counter % 50 == 0:
            #     print(counter, "out of", key_length)
            assocs = self.sem_rel_dict[word]
            network.store(mem_id=word,
                          time=1,
                          spread_depth=spread_depth,
                          activate=False,
                          assocs=assocs)
        return network
