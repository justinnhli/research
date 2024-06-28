from agent_cooccurrence import AgentCooccurrenceNGrams, AgentCooccurrenceCorpus, AgentCooccurrence
from agent_spreading import AgentSpreadingNGrams, AgentSpreadingCorpus, AgentSpreading
from collections import defaultdict
from n_gram_cooccurrence.google_ngrams import GoogleNGram
import math
import numpy


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
        joint_candidates = set(spreading_candidates) & set(cooc_candidates) # Elems not in the intersection will have zero joint probability
        for candidate in joint_candidates:
            semantic_activations[candidate] = self.get_semantic_activation(candidate, time)
            if semantic_activations[candidate] is not None:
                sem_sum += semantic_activations[candidate]
            conditional_probabilities[candidate] = self.get_conditional_probability(candidate, context)
            if conditional_probabilities[candidate] is not None:
                cooc_sum += conditional_probabilities[candidate]
        joint_probabilities = defaultdict(float)
        if sem_sum == 0 or cooc_sum == 0:
            return joint_probabilities
        for candidate in joint_candidates:
            if conditional_probabilities[candidate] is None or semantic_activations[candidate] is None:
                joint_probabilities[candidate] = 0
            else:
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
                max_joint = joint
            if joint == max_joint:
                guesses.append(key)
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
        self.joint_cond_prob_dict = self.get_joint_conditional_probability_dict()
        self.spread_agent = AgentSpreadingNGrams(sem_rel_dict, stopwords, spreading, clear, activation_base,
                                                 decay_parameter, constant_offset)
        self.network = self.spread_agent.create_sem_network()

    def get_joint_conditional_probability_dict(self):
        """ Easier to access conditional probabilities for each word that
            is jointly cooccurrent with the 3 RAT context words"""
        joint_conditional_prob_dict = defaultdict(dict)
        for elem in self.cooc_agent.cooc_cache.keys():
            joint_cooc_rels = self.cooc_agent.cooc_cache[elem]
            sub_dict = defaultdict(int)
            for entry in joint_cooc_rels:
                sub_dict[entry[0]] = entry[1]
            joint_conditional_prob_dict[elem] = sub_dict
        return joint_conditional_prob_dict


    def get_conditional_probability(self, word, context):
        """ Gets conditional probability for ngrams directly from the cooccurrence class.
        Context is the 3 RAT context words given on each trial.
        """
        return self.joint_cond_prob_dict[tuple(context)][word]

    def get_semantic_activation(self, word, time):
        """ Gets semantic activation for a word at a specific time """
        act = self.network.get_activation(mem_id=word.upper(), time=time)
        if act is not None:
            return math.exp(act)
        return act

    def do_rat(self, context1, context2, context3):
        """ Does one trial of the RAT task"""
        context_words = [context1.upper(), context2.upper(), context3.upper()]
        joint_cooc_list = [elem[0] for elem in self.cooc_agent.cooc_cache[tuple(context_words)]]
        # Just gets the words, not conditional probabilities
        if len(joint_cooc_list) == 0:
            return []
        if self.spreading:
            spread_depth = -1
        else:
            spread_depth = 0
        for context in context_words:
            self.network.store(mem_id=context, time=2, spread_depth=spread_depth)
        network_elements = sorted([elem.upper() for elem in set(self.network.knowledge.keys())])
        joint_probs = self.get_joint_probabilities(spreading_candidates=network_elements,
                                                   cooc_candidates=joint_cooc_list,
                                                   context=context_words,
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


class AgentJointVariance(AgentJointProbability):
    """Modifies the implementation of AgentJointProbabilityCorpus to take into account the spread/variance of the
         distribution (via std deviation) for each choice of network clearing and context types."""

    def __init__(self, var_type="stdev", activation_base=2.0, decay_parameter=0.05, constant_offset=0.0):
        super().__init__(activation_base=activation_base, decay_parameter=decay_parameter,
                        constant_offset=constant_offset)
        self.var_type = var_type
        self.cooc_agent = AgentCooccurrence()
        self.spreading_agent = AgentSpreading(spreading=True, clear="never", activation_base=activation_base,
                                              decay_parameter=decay_parameter, constant_offset=constant_offset)

    def get_conditional_distribution(self, *args):
        raise NotImplementedError

    def get_spreading_distribution(self, *args):
        raise NotImplementedError

    def get_distribution_variance(self, distribution):
        """ Gets the spread of the distribution via standard deviation.
            Possible values of var_type:
                "stdev" --> Returns the standard deviation of the distribution.
                "maxdiff" --> Returns the difference between the highest probability and 2nd highest probability items
        """
        vals = sorted(list(distribution.values()))
        if self.var_type == "stdev":
            return numpy.std(vals)
        elif self.var_type == "maxdiff":
            if len(vals) < 2:
                return 0
            return vals[-1] - vals[-2]
        else:
            raise ValueError(self.var_type)


    def create_joint_distribution(self, cooccurrence_dist, spreading_dist):
        """ Creates a joint distribution, assuming the cooccurrence and spreading distributions are already known"""
        words = set()
        words.update(list(cooccurrence_dist.keys()))
        words.update(list(spreading_dist.keys()))
        joint_dist = defaultdict(float)
        for word in words:
            if word not in cooccurrence_dist.keys() or word not in spreading_dist.keys():
                joint_dist[word] = 0
                continue
            joint_dist[word] = cooccurrence_dist[word] * spreading_dist[word]
        return joint_dist

    def do_wsd(self, target_index, sentence, timer):
        raise NotImplementedError

    def do_rat(self, context1, context2, context3):
        raise NotImplementedError

    def get_guesses(self, dist):
        """ Makes guesses based on pre-calculated distributions. """
        max_prob = -float("inf")
        max_guesses = []
        for key in dist.keys():
            prob = dist[key]
            if prob == max_prob:
                max_guesses.append(key)
            elif prob > max_prob:
                max_prob = prob
                max_guesses = [key]
        return max_guesses


class AgentJointVarianceCorpus(AgentJointVariance):
    """Modifies the implementation of AgentJointProbabilityCorpus to take into account the spread/variance of the
     distribution (via std deviation) for each choice of network clearing and context types for the Semcor corpus."""

    def __init__(self, num_sentences, partition, corpus_utilities, outside_corpus, context_type, clear,
                 var_type="stdev", activation_base=2.0, decay_parameter=0.05, constant_offset=0.0):
        super().__init__(var_type=var_type, activation_base=activation_base, decay_parameter=decay_parameter,
                         constant_offset=constant_offset)
        self.num_sentences = num_sentences
        self.partition = partition
        self.corpus_utilities = corpus_utilities
        self.outside_corpus = outside_corpus
        self.word_sense_dict = self.corpus_utilities.get_word_sense_dict()
        self.context_type = context_type
        self.clear = clear
        self.cooc_agent = AgentCooccurrenceCorpus(num_sentences, partition, corpus_utilities, context_type=context_type)
        self.spreading_agent = AgentSpreadingCorpus(corpus_utilities, outside_corpus, spreading=True,
                                                          clear=clear, activation_base=activation_base,
                                                          decay_parameter=decay_parameter,
                                                          constant_offset=constant_offset)
        self.network = self.spreading_agent.create_sem_network()


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

    def get_conditional_distribution(self, word, context, context_type):
        """ Creates a dictionary of the conditional distribution for every possible sense of a certain word in the
         sentence given.
         Assumes word is not in the context. """
        prob_sum = 0
        conditional_probs = defaultdict(float)
        senses = self.word_sense_dict[word[0]]
        for sense in senses:
            conditional_prob = 0
            for con in context:
                if context_type == "word":
                    conditional_prob += self.cooc_agent.get_conditional_probability(target=sense, base=con[0])
                else:
                    conditional_prob += self.cooc_agent.get_conditional_probability(target=sense, base=con)
            conditional_probs[sense] = conditional_prob
            prob_sum += conditional_prob
        for key in conditional_probs.keys():
            conditional_probs[key] = conditional_probs[key] / prob_sum
        return conditional_probs

    def get_spreading_distribution(self, word, time):
        """ Gets distribution of semantic activation values for all senses of a word given.
            Options for clear are "never", "sentence", "word" """
        act_sum = 0
        senses = self.word_sense_dict[word[0]]
        sense_acts = defaultdict(float)
        for sense in senses:
            sense_acts[sense] = math.exp(self.network.get_activation(sense, time))
            act_sum += sense_acts[sense]
        if act_sum == 0:
            return {}
        for sense in senses:
            sense_acts[sense] = sense_acts[sense] / act_sum
        return sense_acts



    def do_wsd(self, target_index, sentence, timer):
        """Implements somewhat of an oracle agent across all clearing modalities and context types to determine which
        offers the most evidence for the correct guess.
        Chooses the "peakiest" cooccurrence and spreading distributions to combine into a joint distribution based on
        standard deviation"""
        word = sentence[target_index]
        edited_sentence = []
        for index in range(len(sentence)):
            if index != target_index:
                edited_sentence.append(sentence[index])
        cooc_dist = self.get_conditional_distribution(word, edited_sentence, self.context_type)
        spread_dist = self.get_spreading_distribution(word, time=timer)
        cooc_dist_variance = self.get_distribution_variance(cooc_dist)
        spread_dist_variance = self.get_distribution_variance(spread_dist)
        if cooc_dist_variance < spread_dist_variance:
            guesses = self.spreading_agent.do_wsd(word, self.word_sense_dict[word[0]], timer, self.network)
        else:
            guesses = self.cooc_agent.do_wsd(target_index, sentence)
        return guesses

    def get_avg_variance(self, var_type):
        """ Gets quantile information for the variance of cooccurrence and spreading data. """
        sentence_list = self.corpus_utilities.get_sentence_list()
        timer = 2
        cooc_variances = []
        spread_variances = []
        for sentence in sentence_list:
            for target_index in range(len(sentence)):
                word = sentence[target_index]
                edited_sentence = []
                for index in range(len(sentence)):
                    if index != target_index:
                        edited_sentence.append(sentence[index])
                cooc_dist = self.get_conditional_distribution(word, edited_sentence, self.context_type)
                spread_dist = self.get_spreading_distribution(word, time=timer)
                timer += 1
                cooc_variances.append(self.get_distribution_variance(cooc_dist, var_type))
                spread_variances.append(self.get_distribution_variance(spread_dist, var_type))
        print("cooc:", "min:", numpy.quantile(cooc_variances, 0), "0.25:", numpy.quantile(cooc_variances, 0.25),
              "median:", numpy.quantile(cooc_variances, 0.5), "0.75:", numpy.quantile(cooc_variances, 0.75), "max:",
              numpy.quantile(cooc_variances, 1))
        print("spread:", "min:", numpy.quantile(spread_variances, 0), "0.25:", numpy.quantile(spread_variances, 0.25),
              "median:", numpy.quantile(spread_variances, 0.5), "0.75:", numpy.quantile(spread_variances, 0.75), "max:",
              numpy.quantile(spread_variances, 1))

class AgentJointVarianceNGrams(AgentJointVariance):
    """ Implementation of joint-based oracle for the Google ngrams corpus. """

    def __init__(self, sem_rel_dict, stopwords, ngrams=GoogleNGram('~/ngram'), var_type="stdev", spreading=True,
                 clear="never", activation_base=2, decay_parameter=0.05, constant_offset=0):
        super().__init__(var_type=var_type, activation_base=activation_base, decay_parameter=decay_parameter,
                         constant_offset=constant_offset)
        self.sem_rel_dict = sem_rel_dict
        self.stopwords = stopwords
        self.ngrams = ngrams
        self.cooc_agent = AgentCooccurrenceNGrams(stopwords, ngrams)
        self.joint_cond_prob_dict = self.get_joint_conditional_probability_dict()
        self.spread_agent = AgentSpreadingNGrams(sem_rel_dict, stopwords, spreading, clear, activation_base,
                                                 decay_parameter, constant_offset)
        self.network = self.spread_agent.create_sem_network()

    def get_joint_conditional_probability_dict(self):
        """ Easier to access conditional probabilities for each word that
            is jointly cooccurrent with the 3 RAT context words"""
        joint_conditional_prob_dict = defaultdict(dict)
        for elem in self.cooc_agent.cooc_cache.keys():
            joint_cooc_rels = self.cooc_agent.cooc_cache[elem]
            sub_dict = defaultdict(int)
            for entry in joint_cooc_rels:
                sub_dict[entry[0]] = entry[1]
            joint_conditional_prob_dict[elem] = sub_dict
        return joint_conditional_prob_dict

    def get_conditional_distribution(self, context1, context2, context3):
        """ Getting a conditional distribution dictionary with keys the words jointly related to the 3 context
        words, and values the normalized conditional probabilities. """
        cond_probs = self.joint_cond_prob_dict[tuple([context1.upper(), context2.upper(), context3.upper()])]
        # Normalizing the probabilities
        total = sum(list(cond_probs.values()))
        for key in cond_probs.keys():
            cond_probs[key] = cond_probs[key]/total
        return cond_probs

    def get_spreading_distribution(self, context1, context2, context3):
        """ Gets normalized distribution of activations resulting from activating the 3 context words in the spreading
         mechanism on the RAT"""
        act_sum = 0
        spread_dist = defaultdict(float)
        self.network.store(context1.upper(), 1)
        self.network.store(context2.upper(), 1)
        self.network.store(context3.upper(), 1)
        for elem in self.network.activation.activations.keys():
            act = math.exp(self.network.get_activation(elem, 2))
            if act is not None and act != 0:
                spread_dist[elem] = act
                act_sum += act
        for elem in spread_dist.keys():
            spread_dist[elem] = spread_dist[elem] / act_sum
        self.spreading_agent.clear_sem_network(self.network, 0) # reset the network.
        return spread_dist

    def do_rat(self, context1, context2, context3):
        """ Does the RAT test in an oracle manner based on whichever method has the most variance"""
        cooc_dist = self.get_conditional_distribution(context1, context2, context3)
        spread_dist = self.get_spreading_distribution(context1, context2, context3)
        cooc_dist_variance = self.get_distribution_variance(cooc_dist)
        spread_dist_variance = self.get_distribution_variance(spread_dist)
        if cooc_dist_variance < spread_dist_variance:
            guesses = self.get_guesses(spread_dist)
        else:
            guesses = self.get_guesses(cooc_dist)
        return guesses
