from agent_cooccurrence import AgentCooccurrenceNGrams, AgentCooccurrenceCorpus, AgentCooccurrence
from agent_spreading import AgentSpreadingNGrams, AgentSpreadingCorpus, AgentSpreading
from collections import defaultdict
from n_gram_cooccurrence.google_ngrams import GoogleNGram
import math

class AgentJointProbability:
    """ Implements a general integrated joint probability mechanism. """
    def __init__(self, spreading=True, clear="never", activation_base=2, decay_parameter=0.05, constant_offset=0):
        self.spreading = spreading
        self.clear = clear
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter
        self.constant_offset = constant_offset
        self.spreading_agent = AgentSpreading(spreading, clear, activation_base, decay_parameter, constant_offset)
        self.cooc_agent = AgentCooccurrence()

    def get_conditional_probability(self, word, context):
        """ Gets conditional probability of a word given context from the cooccurrence agent mechanism.
        """
        raise NotImplementedError

    def get_semantic_activation(self, word, time):
        """ Gets activation from spreading agent mechanism"""
        raise NotImplementedError

    def get_joint_probabilities(self, cooc_candidates, spreading_candidates, context, time):
        """
        Gets the joint probability of conditional probabilities for cooccurrence
        and activations for semantic spreading
         """
        conditional_probabilities = defaultdict(float)
        semantic_activations = defaultdict(float)
        sem_sum = 0
        cooc_sum = 0
        for candidate in spreading_candidates:
            semantic_activations[candidate] = self.get_semantic_activation(candidate, time)
            sem_sum += semantic_activations[candidate]
        for candidate in cooc_candidates:
            conditional_probabilities[candidate] = self.get_conditional_probability(candidate, context)
            cooc_sum += conditional_probabilities[candidate]
        joint_candidates = set(cooc_candidates) & set(spreading_candidates)
        joint_probabilities = defaultdict(float)
        for candidate in joint_candidates:
            joint_probabilities[candidate] = (conditional_probabilities[candidate] * semantic_activations[
                candidate]) / (sem_sum * cooc_sum)
        return joint_probabilities

    def do_wsd(self, target_index, sentence, time):
        raise NotImplementedError


class AgentJointProbabilityCorpus(AgentJointProbability):
    """Implements joint probability integrated mechanism. """

    def __init__(self, num_sentences, partition, corpus_utilities, context_type, outside_corpus, spreading=True,
                 clear="never", activation_base=2.0, decay_parameter=0.05, constant_offset=0.0):
        super().__init__(spreading, clear, activation_base, decay_parameter, constant_offset)
        self.num_sentences = num_sentences
        self.partition = partition
        self.activation_base = activation_base
        self.corpus_utilities = corpus_utilities
        self.context_type = context_type
        self.outside_corpus = outside_corpus
        self.cooc_agent = AgentCooccurrenceCorpus(num_sentences, partition, corpus_utilities, context_type)
        self.spreading_agent = AgentSpreadingCorpus(corpus_utilities, outside_corpus, spreading, clear, activation_base,
                                                    decay_parameter, constant_offset)
        self.network = self.spreading_agent.create_sem_network()
        self.word_sense_dict = self.corpus_utilities.get_word_sense_dict()

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
        self.network = self.spreading_agent.clear_sem_network(self.network, start_time)

    def get_conditional_probability(self, word, context):
        """ Gets conditional probability for word across whole sentence (context)"""
        conditional_prob = 0
        for con in context:
            if self.context_type == "word":
                conditional_prob += self.cooc_agent.get_conditional_probability(target=word, base=con[0])
            else:
                conditional_prob += self.cooc_agent.get_conditional_probability(target=word, base=con)
        return conditional_prob

    def get_semantic_activation(self, word, time):
        """ Gets semantic activation"""
        return math.exp(self.network.get_activation(mem_id=word, time=time))

    def do_wsd(self, target_index, sentence, time):
        """ Does one trial of the WSD task."""
        word = sentence[target_index]
        edited_sentence = []
        for index in range(len(sentence)):
            if index != target_index:
                edited_sentence.append(sentence[index])
        senses = self.word_sense_dict[word[0]]
        joint_probs = self.get_joint_probabilities(cooc_candidates=senses, spreading_candidates=senses,
                                                   context=edited_sentence, time=time)
        max_joint = 0
        guesses = []
        for key in list(joint_probs.keys()):
            joint = joint_probs[key]
            if joint > max_joint:
                guesses = [key]
            if joint == max_joint:
                guesses.append(key)
                max_joint = joint
        for guess in guesses:
            self.network.store(guess, time)
        self.network.store(word, time)
        return guesses


class AgentJointProbabilityNGrams(AgentJointProbability):
    """ Implements the integrated joint probability mechanism for the NGrams corpus."""
    def __init__(self, sem_rel_dict, stopwords, ngrams=GoogleNGram('~/ngram'), spreading=True, clear="never",
                 activation_base=2, decay_parameter=0.05, constant_offset=0):
        super().__init__(spreading, clear, activation_base, decay_parameter, constant_offset)
        self.sem_rel_dict = sem_rel_dict
        self.stopwords = stopwords
        self.ngrams = ngrams
        self.cooc_agent = AgentCooccurrenceNGrams(stopwords, ngrams)
        self.spread_agent = AgentSpreadingNGrams(sem_rel_dict, stopwords, ngrams, spreading, clear, activation_base,
                 decay_parameter, constant_offset)
        self.network = self.spread_agent.create_semantic_network()

    def get_conditional_probability(self, word, context):
        """ Gets conditional probability for ngrams directly from the cooccurrence class.
        Context is the 3 RAT context words given on each trial.
        """
        cond_prob1 = self.ngrams.get_conditional_probability(base=context[0], target=word)
        cond_prob2 = self.ngrams.get_conditional_probability(base=context[1], target=word)
        cond_prob3 = self.ngrams.get_conditional_probability(base=context[2], target=word)
        joint_cond_prob = cond_prob1 * cond_prob2 * cond_prob3
        return joint_cond_prob

    def get_semantic_activation(self, word, time):
        """ Gets semantic activation for a word at a specific time """
        return self.network.get_activation(mem_id=word, time=time)

    def do_rat(self, context1, context2, context3):
        """ Does one trial of the RAT task"""
        cooc_set1 = set([elem[0] for elem in self.ngrams.get_max_probability(context1)])
        cooc_set2 = set([elem[0] for elem in self.ngrams.get_max_probability(context2)])
        cooc_set3 = set([elem[0] for elem in self.ngrams.get_max_probability(context3)])
        joint_cooc_list = list(cooc_set1 & cooc_set2 & cooc_set3)
        if len(joint_cooc_list) == 0:
            return []
        if self.spreading:
            spread_depth = -1
        else:
            spread_depth = 0
        for context in [context1, context2, context3]:
            self.network.store(mem_id=context, time=2, spread_depth=spread_depth)
        network_elements = sorted(set(self.network.knowledge.keys()))
        joint_probs = self.get_joint_probabilities(spreading_candidates=network_elements,
                                                   cooc_candidates=joint_cooc_list,
                                                   context=[context1, context2, context3],
                                                   time=3)  # Since everything is stored at 2, activate on 3.
        joint_candidates = list(joint_probs.keys())
        if len(joint_candidates) == 0:
            return []
        elif len(joint_candidates) == 1:
            return joint_candidates[0]
        else:
            max_act = -float("inf")
            guesses = []
            for candidate in joint_candidates:
                act = joint_probs[candidate]
                if act > max_act:
                    max_act = act
                    guesses = [candidate]
                elif act == max_act:
                    guesses.append(candidate)
            return guesses



