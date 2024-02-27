import math
import nltk
from long_term_memory import NaiveDictLTM
from uuid import uuid4 as uuid
from data_structures import AVLTree
from rl_environments import AttrVal
from sentence_cooccurrence_activation import SentenceCooccurrenceActivation


class sentenceLTM(NaiveDictLTM):
    " A child class of NaiveDictLTM for the word sense disambiguation task."
    def __init__(self, activation_cls=SentenceCooccurrenceActivation, **kwargs):
        """
        Initialize the sentenceLTM.
        Parameters:
            activation_cls: Must be SentenceCooccurrenceActivation.
        """
        #super.__init__(**kwargs)
        super().__init__(**kwargs)
        self.activation = activation_cls(self)
        self.cooccurrent_elements = {}
        self.knowledge = {}


    def get_cooccurrence(self, word1, word2):
        """
        Gets the cooccurrence of two elements based on information stored in the cooccurrent_elements dict.
        Parameters:
            word1 (tuple): A tuple of the form (lemma, lemma synset) that corresponds to a sense-specific word.
            word2 (tuple):  A tuple of the form (lemma, lemma synset) that corresponds to a sense-specific word.
        Returns:
            (Any): The number of times the elements have cooccurred if they have previously and
            False if they have not cooccurred.
        """
        if word1[0] < word2[0] or (word1[0] == word2[0] and word1[1] <= word2[1]):
            key = (word1, word2)
        else:
            key = (word2, word1)
        if key in self.cooccurrent_elements.keys():
            return self.cooccurrent_elements[key]
        else:
            return False


    def activate_cooccur_pair(self, word1, word2, time=0):
        """
        Activates two words occurring together and updates their co-occurrence in dict cooccurrent_elements.
        Parameters:
            word1 (tuple): A tuple of the form (lemma, lemma synset) that corresponds to a sense-specific word.
            word2 (tuple): A tuple of the form (lemma, lemma synset) that corresponds to a sense-specific word.
            time (float): The time of activation (optional).
        Returns:
            None
        """
        if word1[0] < word2[0] or (word1[0] == word2[0] and word1[1] <= word2[1]):
            key = (word1, word2)
        else:
            key = (word2, word1)
        if key in self.cooccurrent_elements:
            curr_cooccurrence_value = self.cooccurrent_elements.get(key) + 1
            self.cooccurrent_elements[key] = curr_cooccurrence_value
            word1_ratio = 1 + (curr_cooccurrence_value / (len(self.activation.activations[word1]) + 1))
            word2_ratio = 1 + (curr_cooccurrence_value / (len(self.activation.activations[word2]) + 1))
        else:
            self.cooccurrent_elements[key] = 1
            word1_ratio = 1 + (1 / (len(self.activation.activations[word1]) + 1))
            word2_ratio = 1 + (1 / (len(self.activation.activations[word2]) + 1))
        self.store(key, time)
        self.store(word1, time, word1_ratio)
        self.store(word2, time, word2_ratio)

    def store(self, mem_id=None, time=0, element_pair_ratio=1, spread_depth=-1, **kwargs):
        """
        Stores (or activates) an element added to the network.
        Parameters:
            mem_id (any): The ID of the desired element.
            time (float): The time of retrieval (for activation). (optional)
            element_pair_ratio (float): The ratio of the number of times a pair of elements have cooccurred over the number
                of times the element of interest has been activated. (optional)
            spread_depth (int): How deep to spread activation after an element has been activated (optional). Default is
                spread_depth = -1: full spreading to all connected nodes.
        Returns:
            (Boolean): True if completed.
        """
        if mem_id is None:
            mem_id = uuid()
        if mem_id not in self.knowledge:
            self.knowledge[mem_id] = AVLTree()
        for attr, val in kwargs.items():
            if isinstance(val, list):
                for v in val:
                    if v not in self.knowledge:
                      self.knowledge[v] = AVLTree()
                    self.knowledge[mem_id].add(AttrVal(attr, v))
            elif val not in self.knowledge:
                self.knowledge[val] = AVLTree()
                self.knowledge[mem_id].add(AttrVal(attr, val))
        self.activation.simple_activate(mem_id=mem_id, time=time, element_pair_ratio=element_pair_ratio,
                                        spread_depth=spread_depth)
        return True


    def get_activation(self, mem_id, time=0):
        return self.activation.get_activation(mem_id, time)

    def sense_query(self, word, time=0):
        # Gets the activation for each sense of a word given and returns them in a list.
        candidates = []
        for mem_id in self.knowledge.keys():
            #if isinstance(mem_id[0], nltk.corpus.reader.wordnet.Lemma):
            if mem_id[0] == word:
                candidates.append(mem_id)
        candidate_dict = {}
        for candidate in candidates:
            candidate_dict[candidate] = self.activation.get_activation(mem_id=candidate, time=time)
        return candidate_dict