"""
Implements cooccurrence thresholded spreading agent.
Only allows spreading to occur for elements that also cooccur.
 """
from collections import defaultdict
from corpus_utilities import CorpusUtilities
from sentence_cooccurrence_activation import SentenceCooccurrenceActivation
from agent_spreading import AgentSpreadingCorpus, AgentSpreadingNGrams
from sentence_long_term_memory import sentenceLTM
from n_gram_cooccurrence.google_ngrams import *

class AgentCoocThreshSpreadingCorpus(AgentSpreadingCorpus):
    def __init__(self, context_type, whole_corpus, corpus_utilities, outside_corpus, spreading=True, clear="never", activation_base=2,
                 decay_parameter=0.05, constant_offset=0):
        super().__init__(corpus_utilities, outside_corpus, spreading, clear, activation_base, decay_parameter,
                         constant_offset)
        self.context_type = context_type
        self.whole_corpus = whole_corpus

    def adjust_sem_rel_dict(self, sem_rel_dict):
        """ This is for cooccurrence thresholded spreading """
        sent_list = self.corpus_utilities.get_sentence_list()
        if self.whole_corpus:
            whole_cu = CorpusUtilities()
            all_sents_list = whole_cu.get_sentence_list()
            cooc_rel_dict = self.create_cooc_relations_dict(all_sents_list,context_type=self.context_type)
        else:
            cooc_rel_dict = self.create_cooc_relations_dict(sent_list, context_type=self.context_type)
        for word_key in sem_rel_dict.keys():
            word_rel_dict = sem_rel_dict[word_key]  # has all different relations to target word
            for cat in word_rel_dict.keys():  # looping through each relation category
                rels = word_rel_dict[cat]  # getting the relations in that category
                for rel in rels:  # going through words corresponding to each relation
                    if self.context_type == "sense":
                        if rel not in cooc_rel_dict[word_key]:
                            sem_rel_dict[word_key][cat].remove(rel)  # removing un-cooccurrenced relation
                    else:
                        if rel[0] not in cooc_rel_dict[word_key]:
                            sem_rel_dict[word_key][cat].remove(rel)
        return sem_rel_dict

    def create_cooc_relations_dict(self, sentence_list, context_type):
        """
            Creates a dictionary where each word in the corpus is a key and each of the other words it cooccurs with (are in
            the same sentence as our target word) are values (in a set).
            For cooccurrence thresholded spreading
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
        """
        Builds a semantic network with each word in the Semcor corpus and its corresponding synonyms, hypernyms, hyponyms,
            holonyms, meronyms, attributes, entailments, causes, also_sees, verb_groups, and similar_tos. Note that all words
            are stored at time 1.
        Parameters:
            sentence_list (Nested String List): A list of the sentences or the first n sentences in the Semcor corpus
                with each word represented by a tuple: (lemma, lemma synset).
            spreading (bool): Whether to include the effects of spreading in creating the semantic network.
            activation_base (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            constant_offset (float): A parameter in the activation equation.
            partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking at
                sentences 10000 - 14999.
        Returns:
            network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
        """
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
        relations_keys = list(semantic_relations_dict.keys())
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

    def __init__(self, sem_rel_dict, ngrams=GoogleNGram('~/ngram'), spreading=True, clear="never", activation_base=2,
                 decay_parameter=0.05, constant_offset=0):
        """sem_rel_dict has all semantic relations we're interested in - from either SWOWEN or SFFAN, or both!"""
        super().__init__(spreading, clear, activation_base, decay_parameter, constant_offset, ngrams)
        self.sem_rel_dict = self.adjust_sem_rel_dict(sem_rel_dict)

    def get_cooccurring_words(self, word):
        """ Returns an ordered list (most to least # of times word occurs) of tuples formatted as
             (word, # times word occcurred) for words that cooccur with the input word"""
        cooc_words_counts = self.ngrams.get_max_probability(word)
        cooc_words = [word[0] for word in cooc_words_counts]
        return cooc_words

    def adjust_sem_rel_dict(self, sem_rel_dict):
        """ This is for cooccurrence thresholded spreading """
        for word_key in sem_rel_dict.keys():
            cooc_words = self.get_cooccurring_words(word_key.upper())
            word_rel_dict = sem_rel_dict[word_key]  # has all different relations to target word
            for cat in word_rel_dict.keys():  # looping through each relation category
                rels = word_rel_dict[cat]  # getting the relations in that category
                for rel in rels:  # going through words corresponding to each relation
                    if rel.upper() not in cooc_words:
                        sem_rel_dict[word_key][cat].remove(rel)
        return sem_rel_dict

    def create_sem_network(self):
        """
        Builds a semantic network with each key word in the SWOWEN and South Florida Free Association Norms (SFFAN).
            Note that all words are stored at time 1.
        Parameters:
            SWOWEN_link (string): link to the SWOWEN preprocessed dictionary
            SFFAN_link (string): link to the SFFAN preprocessed dictionary
            spreading (bool): Whether to include the effects of spreading in creating the semantic network.
            activation_base (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            constant_offset (float): A parameter in the activation equation.
            partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking at
                sentences 10000 - 14999.
        Returns:
            network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
        """
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
        keys = list(self.sem_rel_dict.keys())
        for word in keys:
            assocs = self.sem_rel_dict[word]
            network.store(mem_id=word,
                          time=1,
                          spread_depth=spread_depth,
                          assocs=assocs)
        return network




