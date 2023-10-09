import math
from research import NaiveDictLTM
from uuid import uuid4 as uuid
from research.data_structures import AVLTree
from research.rl_environments import AttrVal
from research.sentence_cooccurrence_activation import SentenceCooccurrenceActivation


class sentenceLTM(NaiveDictLTM):

    def __init__(self, activation_cls=SentenceCooccurrenceActivation, **kwargs):
        #super.__init__(**kwargs)
        super().__init__(**kwargs)
        self.activation = activation_cls(self)
        self.cooccurrent_elements = {}

    def get_cooccurrence(self, word1, word2):
        if word1[0] < word2[0]:
            key = (word1, word2)
        elif word1[0] == word2[0] and word1[1]<word2[1]:
            key = (word1, word2)
        else:
            key = (word2, word1)
        if key in self.cooccurrent_elements:
            return self.cooccurrent_elements[key]
        else:
            return False

    def activate_cooccur_pair(self, word1, word2, time=0):
        if word1[0] < word2[0]:
            key = (word1, word2)
        elif word1[0] == word2[0] and word1[1]<word2[1]:
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
        self.store(key, time) # FIXME Little confused about where this comes in... bc I won't activate it later
        if word1 in self.knowledge:
            self.activation.simple_activate(word1, time, word1_ratio)
        else:
            self.store(word1, time, word1_ratio)
        if word2 in self.knowledge:
            self.activation.simple_activate(word2, time, word2_ratio)
        else:
            self.store(word2, time, word2_ratio)


    def store(self, mem_id=None, time=0, element_pair_ratio = 1, **kwargs): # noqa: D102
        # type: (Hashable, int, int, **Any) -> bool
        if mem_id is None:
            mem_id = uuid()
        if mem_id not in self.knowledge:
            self.knowledge[mem_id] = AVLTree()
        else:
            self.activation.simple_activate(mem_id, time, element_pair_ratio)
        for attr, val in kwargs.items():
            if val not in self.knowledge:
                self.knowledge[val] = AVLTree()
            self.knowledge[mem_id].add(AttrVal(attr, val))
        return True

