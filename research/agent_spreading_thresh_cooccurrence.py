""""
    Implements the spreading thresholded cooccurrence agent. Only allows as valid "context" or cooccurrently related
    items, those that are semantically related as well. Does not necessarily use spreading, but rather uses the
    relations embedded in the semantic relations dictionary/semantic network to determine what can and cannot be spread
    to.
"""
from agent_cooccurrence import AgentCooccurrenceCorpus, AgentCooccurrenceNGrams
from agent_spreading import AgentSpreadingCorpus, AgentSpreadingNGrams
from n_gram_cooccurrence.google_ngrams import *


class AgentSpreadingThreshCoocCorpus(AgentCooccurrenceCorpus, AgentSpreadingCorpus):

    def __init__(self, num_sentences, partition, corpus_utilities, context_type, outside_corpus):
        """
        Whole corpus is whether semantic relations should be required to cooccur with the target word over the whole
        corpus (True) or only in the partition of interest.
        Parameters:
            num_sentences (int): The number of sentences from the corpus to use in the task. The first n sentences
                from the corpus are used and if n=-1, all sentences from the corpus are used.
            partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking
                at sentences 10000 - 14999.
            corpus_utilities (class): A class of functions useful for corpus mechanisms, specific to the partition of the
                Semcor corpus used
            context_type (string): Indicates for cooccurrence dependent corpus mechanisms, whether we know the sense of the
                context words ("sense") or not ("word")
            whole_corpus (bool): For cooccurrence dependent corpus mechanisms, whether to include cooccurrent relations from
                the whole corpus (True) or not (False).
        """
        super().__init__(num_sentences=num_sentences, partition=partition,
                         corpus_utilities=corpus_utilities, context_type=context_type)
        self.outside_corpus = outside_corpus
        self.spreading_agent = AgentSpreadingCorpus(corpus_utilities, outside_corpus)
        self.sem_rel_dict = self.get_semantic_relations_dict(False)
        if context_type == "word":
            self.sem_rel_dict = self.get_word_adjusted_sem_rel_dict(self.sem_rel_dict)
        self.sense_sense_cooccurrences = corpus_utilities.get_sense_sense_cooccurrences()
        self.sense_word_cooccurrences = corpus_utilities.get_sense_word_cooccurrences()
        self.word_word_cooccurrences = corpus_utilities.get_word_word_cooccurrences()
        self.word_sense_dict = corpus_utilities.get_word_sense_dict()



    def get_word_adjusted_sem_rel_dict(self, sem_rel_dict):
        """
         Creates a word-based semantic relations dictionary (assuming we don't care about the sense of each
         semantically-related word).
         Parameters:
            sem_rel_dict (dict): A nested dictionary with each sense-specific word the key, and values the different
                semantic categories (synonyms, hyponyms, etc.) that the various sense-specific semantically related
                words are included in.
         Returns: (dict) Altered semantic relations dict that assumes only the sense of each semantically related word
            is not known.
        """
        keys = sem_rel_dict.keys()
        for word in keys:
            rels = sem_rel_dict[word]
            for rel in rels.keys():
                new_rel_list = []
                rel_words = rels[rel]
                for rel_word in rel_words:
                    new_rel_list.append(rel_word[0])
                sem_rel_dict[word][rel] = new_rel_list
        return sem_rel_dict


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
        for target_sense_candidate in self.word_sense_dict[target_sense[0]]:
            candidate_conditional_probability = 0
            for context_index in range(len(sentence)):
                if context_index == target_index:
                    continue
                context_sense = sentence[context_index]
                if self.context_type == "sense":
                    base = context_sense
                    organized_target_rels = self.sem_rel_dict[target_sense_candidate]
                    target_rels = sum(list(organized_target_rels.values()), [])
                    if context_sense not in target_rels:
                        continue
                else:  # context == "word"
                    context_word = context_sense[0]
                    base = context_word
                    organized_target_rels = self.sem_rel_dict[target_sense_candidate]
                    target_rels = sum(list(organized_target_rels.values()), [])
                    if context_word not in target_rels:
                        continue
                candidate_conditional_probability += self.get_conditional_probability(target=target_sense_candidate,
                                                                                      base=base)
            if candidate_conditional_probability > max_score:
                max_score = candidate_conditional_probability
                max_senses = [target_sense_candidate]
            elif candidate_conditional_probability == max_score:
                max_senses.append(target_sense_candidate)
        if max_score == -float("inf") or max_score == 0:
            return []
        return max_senses


class AgentSpreadingThreshCoocNGrams(AgentCooccurrenceNGrams, AgentSpreadingNGrams):

    def __init__(self, stopwords, sem_rel_dict, ngrams=GoogleNGram('~/ngram')):
        """
        Parameters:
            stopwords (list): A list of stopwords - common words to not include semantic relations to.
            sem_rel_dict (dict): A dictionary containing all semantic relations (the values) for each word
                (the keys) from the SWOWEN, SFFAN, (or both) databases.
            ngrams (class): The google ngrams class.
        """
        super().__init__(stopwords, ngrams)
        self.sem_rel_dict = self.filter_sem_rel_dict(sem_rel_dict)


    def do_rat(self, context1, context2, context3):
        """
        Completes one round of the RAT task.
        Parameters:
            context1, context2, context3 (string): Context words to be used in the RAT task.
        Returns:
            A list of RAT guesses. Returns [] if there are no viable guesses.
        """
        cooc_set1 = set([elem[0].upper() for elem in self.ngrams.get_max_probability(context1)])
        cooc_set2 = set([elem[0].upper() for elem in self.ngrams.get_max_probability(context2)])
        cooc_set3 = set([elem[0].upper() for elem in self.ngrams.get_max_probability(context3)])
        # Now threshold based on semantic relations as well
        sem_rel_set1 = set()
        sem_rel_set2 = set()
        sem_rel_set3 = set()
        if context1.lower() in list(self.sem_rel_dict.keys()):
            sem_rel_set1 = set([elem.upper() for elem in self.sem_rel_dict[context1]])
        if context2.lower() in list(self.sem_rel_dict.keys()):
            sem_rel_set2 = set([elem.upper() for elem in self.sem_rel_dict[context2]])
        if context3.lower() in list(self.sem_rel_dict.keys()):
            sem_rel_set3 = set([elem.upper() for elem in self.sem_rel_dict[context3]])
        joint_cooc_set = cooc_set1 & cooc_set2 & cooc_set3 & sem_rel_set1 & sem_rel_set2 & sem_rel_set3
        if len(joint_cooc_set) == 0:
            return None
        elif len(joint_cooc_set) == 1:
            return joint_cooc_set.pop()
        else:
            max_cond_prob = -float("inf")
            max_elems = []
            for elem in list(joint_cooc_set):
                # threshold based on stopwords
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
