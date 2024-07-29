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

    def get_activation_probability(self, word_activation, other_activations, tau, s=0.25):
        """ Gets activation probability for a given element with a specified activation """
        num = math.exp(word_activation / s)
        denom = math.exp(tau / s) + sum(math.exp(act / s) for act in other_activations)
        return num / denom

    def get_spreading_distribution(self, *args):
        raise NotImplementedError

    def get_cooccurrence_distribution(self, *args):
        raise NotImplementedError

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

    def do_wsd(self, target_index, sentence, time):
        raise NotImplementedError

    def do_rat(self, target_index, sentence, time):
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
        self.spreading_agent.clear_sem_network(start_time)

    def get_conditional_probability(self, word, context):
        """ Gets conditional probability for word across whole sentence (context)"""
        conditional_prob = 0
        for con in context:
            if self.context_type == "word":
                conditional_prob += self.cooc_agent.get_conditional_probability(target=word, base=con[0])
            else:
                conditional_prob += self.cooc_agent.get_conditional_probability(target=word, base=con)
        return conditional_prob

    def get_cooccurrence_distribution(self, target_index, word_senses, sentence, context_type):
        """ Creates a dictionary of the conditional distribution for every possible sense of a certain word in the
         sentence given.
         Assumes word is not in the context. """
        conditional_probs = defaultdict(float)
        cooc_sum = 0
        for sense in word_senses:
            conditional_prob = 0
            for con_index in range(len(sentence)):
                if con_index == target_index:
                    continue
                con = sentence[con_index]
                if context_type == "word":
                    conditional_prob += self.cooc_agent.get_conditional_probability(target=sense, base=con[0])
                else:
                    conditional_prob += self.cooc_agent.get_conditional_probability(target=sense, base=con)
            conditional_probs[sense] = conditional_prob
            cooc_sum += conditional_prob
        for key in conditional_probs.keys():
            conditional_probs[key] = conditional_probs[key] / cooc_sum
        return conditional_probs

    def get_spreading_distribution(self, word_senses, time, tau=-float("inf"), s=0.25):
        """ Gets distribution of semantic activation values for all senses of a word given.
            Options for clear are "never", "sentence", "word" """
        sense_acts = defaultdict(float)
        for sense in word_senses:
            sense_acts[sense] = self.spreading_agent.network.get_activation(sense, time)
        sense_act_probs = defaultdict(float)
        for sense in word_senses:
            other_acts = list(sense_acts.values())
            sense_act_probs[sense] = self.get_activation_probability(word_activation=sense_acts[sense],
                                                                     other_activations=other_acts,
                                                                     tau=tau, s=s)
        return sense_act_probs

    def do_wsd(self, target_index, sentence, time):
        """ Does one trial of the WSD task."""
        word = sentence[target_index]
        edited_sentence = []
        for index in range(len(sentence)):
            if index != target_index:
                edited_sentence.append(sentence[index])
        senses = self.word_sense_dict[word[0]]
        spread_dist = self.get_spreading_distribution(senses, time)
        cooc_dist = self.get_cooccurrence_distribution(context_type=self.context_type, word_senses=senses,
                                                       target_index=target_index, sentence=sentence)
        joint_probs = self.create_joint_distribution(spreading_dist=spread_dist, cooccurrence_dist=cooc_dist)
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
            self.spreading_agent.network.store(guess, time)
        self.spreading_agent.network.store(word, time)
        return guesses


class AgentJointProbabilityNGrams(AgentJointProbability):
    """ Implements the integrated joint probability mechanism for the NGrams corpus."""

    def __init__(self, stopwords, source, ngrams=GoogleNGram('~/ngram'), spreading=True, clear="never",
                 activation_base=2, decay_parameter=0.05, constant_offset=0):
        super().__init__(spreading, clear, activation_base, decay_parameter, constant_offset)
        self.source = source
        self.stopwords = stopwords
        self.ngrams = ngrams
        self.cooc_agent = AgentCooccurrenceNGrams(stopwords, ngrams)
        self.joint_cond_prob_dict = self.get_joint_conditional_probability_dict()
        self.spreading_agent = AgentSpreadingNGrams(stopwords, source, spreading, clear, activation_base,
                                                 decay_parameter, constant_offset)

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

    def get_cooccurrence_distribution(self, context1, context2, context3):
        """ Getting a conditional distribution dictionary with keys the words jointly related to the 3 context
        words, and values the normalized conditional probabilities. """
        cond_probs = self.joint_cond_prob_dict[tuple([context1.upper(), context2.upper(), context3.upper()])]
        # Normalizing the probabilities
        total = sum(list(cond_probs.values()))
        for key in cond_probs.keys():
            cond_probs[key] = cond_probs[key] / total
        return cond_probs

    def get_spreading_distribution(self, context1, context2, context3):
        """ Gets normalized distribution of activations resulting from activating the 3 context words in the spreading
         mechanism on the RAT"""
        self.spreading_agent.network.store(context1.upper(), 1)
        self.spreading_agent.network.store(context2.upper(), 1)
        self.spreading_agent.network.store(context3.upper(), 1)
        acts = defaultdict(float)
        for elem in self.spreading_agent.network.activation.activations.keys():
            act = self.spreading_agent.network.get_activation(elem, 2)
            if act is not None and act != 0:
                acts[elem] = act
        act_prob_dist = defaultdict(float)
        act_sum = 0
        other_acts = list(acts.values())
        for elem in acts.keys():
            act_prob_dist[elem] = self.get_activation_probability(acts[elem], other_acts, math.log(3/8))
            act_sum += act_prob_dist[elem]
        for elem in act_prob_dist.keys():
            act_prob_dist[elem] = act_prob_dist[elem] / act_sum
        self.spreading_agent.clear_sem_network(0)  # reset the network.
        return act_prob_dist

    def do_rat(self, context1, context2, context3):
        """ Does one trial of the RAT task"""
        context_words = [context1.upper(), context2.upper(), context3.upper()]
        cooc_dist = self.get_cooccurrence_distribution(context_words[0], context_words[1], context_words[2])
        spread_dist = self.get_spreading_distribution(context_words[0], context_words[1], context_words[2])
        joint_probs = self.create_joint_distribution(spreading_dist=spread_dist, cooccurrence_dist=cooc_dist)
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

    def __init__(self, corpus_utilities, outside_corpus, context_type, clear,
                 var_type="stdev", activation_base=2.0, decay_parameter=0.05, constant_offset=0.0):
        super().__init__(var_type=var_type, activation_base=activation_base, decay_parameter=decay_parameter,
                         constant_offset=constant_offset)
        self.num_sentences = corpus_utilities.num_sentences
        self.partition = corpus_utilities.partition
        self.corpus_utilities = corpus_utilities
        self.outside_corpus = outside_corpus
        self.word_sense_dict = self.corpus_utilities.get_word_sense_dict()
        self.context_type = context_type
        self.clear = clear
        self.cooc_agent = AgentCooccurrenceCorpus(self.num_sentences, self.partition, corpus_utilities, context_type=context_type)
        self.spreading_agent = AgentSpreadingCorpus(corpus_utilities, outside_corpus, spreading=True,
                                                    clear=clear, activation_base=activation_base,
                                                    decay_parameter=decay_parameter,
                                                    constant_offset=constant_offset)

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
        self.spreading_agent.clear_sem_network(start_time)

    def get_cooccurrence_distribution(self, target_index, word_senses, sentence, context_type):
        """ Creates a dictionary of the conditional distribution for every possible sense of a certain word in the
         sentence given.
         Assumes word is not in the context. """
        conditional_probs = defaultdict(float)
        cooc_sum = 0
        for sense in word_senses:
            conditional_prob = 0
            for con_index in range(len(sentence)):
                if con_index == target_index:
                    continue
                con = sentence[con_index]
                if context_type == "word":
                    conditional_prob += self.cooc_agent.get_conditional_probability(target=sense, base=con[0])
                else:
                    conditional_prob += self.cooc_agent.get_conditional_probability(target=sense, base=con)
            conditional_probs[sense] = conditional_prob
            cooc_sum += conditional_prob
        for key in conditional_probs.keys():
            conditional_probs[key] = conditional_probs[key] / cooc_sum
        return conditional_probs

    def get_spreading_distribution(self, word_senses, time, tau=-float("inf"), s=0.25):
        """ Gets distribution of semantic activation values for all senses of a word given.
            Options for clear are "never", "sentence", "word" """
        sense_acts = defaultdict(float)
        for sense in word_senses:
            sense_acts[sense] = self.spreading_agent.network.get_activation(sense, time)
        sense_act_probs = defaultdict(float)
        for sense in word_senses:
            other_acts = list(sense_acts.values())
            sense_act_probs[sense] = self.get_activation_probability(word_activation=sense_acts[sense],
                                                                     other_activations=other_acts,
                                                                     tau=tau, s=s)
        return sense_act_probs

    def do_wsd(self, target_index, sentence, time):
        """ Does one trial of the WSD task."""
        word = sentence[target_index]
        edited_sentence = []
        for index in range(len(sentence)):
            if index != target_index:
                edited_sentence.append(sentence[index])
        senses = self.word_sense_dict[word[0]]
        spread_dist = self.get_spreading_distribution(senses, time)
        spread_dist_var = self.get_distribution_variance(spread_dist)
        cooc_dist = self.get_cooccurrence_distribution(context_type=self.context_type, word_senses=senses,
                                                       target_index=target_index, sentence=sentence)
        cooc_dist_var = self.get_distribution_variance(cooc_dist)
        if cooc_dist_var >= spread_dist_var:
            guesses = self.get_guesses(cooc_dist)
        else:
            guesses = self.get_guesses(spread_dist)
        for guess in guesses:
            self.spreading_agent.network.store(guess, time)
        self.spreading_agent.network.store(word, time)
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

    def __init__(self, source, stopwords, ngrams=GoogleNGram('~/ngram'), var_type="stdev", spreading=True,
                 clear="never", activation_base=2, decay_parameter=0.05, constant_offset=0):
        super().__init__(var_type=var_type, activation_base=activation_base, decay_parameter=decay_parameter,
                         constant_offset=constant_offset)
        self.stopwords = stopwords
        self.ngrams = ngrams
        self.cooc_agent = AgentCooccurrenceNGrams(stopwords, ngrams)
        self.joint_cond_prob_dict = self.get_joint_conditional_probability_dict()
        self.spreading_agent = AgentSpreadingNGrams(source=source, stopwords=stopwords, spreading=spreading, clear=clear,
                                                 activation_base=activation_base,
                                                 decay_parameter=decay_parameter, constant_offset=constant_offset)

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

    def get_cooccurrence_distribution(self, context1, context2, context3):
        """ Getting a conditional distribution dictionary with keys the words jointly related to the 3 context
        words, and values the normalized conditional probabilities. """
        cond_probs = self.joint_cond_prob_dict[tuple([context1.upper(), context2.upper(), context3.upper()])]
        # Normalizing the probabilities
        total = sum(list(cond_probs.values()))
        for key in cond_probs.keys():
            cond_probs[key] = cond_probs[key] / total
        return cond_probs

    def get_spreading_distribution(self, context1, context2, context3):
        """ Gets normalized distribution of activations resulting from activating the 3 context words in the spreading
         mechanism on the RAT"""
        self.spreading_agent.network.store(context1.upper(), 1)
        self.spreading_agent.network.store(context2.upper(), 1)
        self.spreading_agent.network.store(context3.upper(), 1)
        acts = defaultdict(float)
        for elem in self.spreading_agent.network.activation.activations.keys():
            act = self.spreading_agent.network.get_activation(elem, 2)
            if act is not None and act != 0:
                acts[elem] = act
        act_prob_dist = defaultdict(float)
        act_sum = 0
        other_acts = list(acts.values())
        for elem in acts.keys():
            act_prob_dist[elem] = self.get_activation_probability(acts[elem], other_acts, math.log(3/8))
            act_sum += act_prob_dist[elem]
        for elem in act_prob_dist.keys():
            act_prob_dist[elem] = act_prob_dist[elem] / act_sum
        self.spreading_agent.clear_sem_network(0)  # reset the network.
        return act_prob_dist


    def do_rat(self, context1, context2, context3):
        """ Does the RAT test in an oracle manner based on whichever method has the most variance"""
        context_words = [context1.upper(), context2.upper(), context3.upper()]
        cooc_dist = self.get_cooccurrence_distribution(context_words[0], context_words[1], context_words[2])
        spread_dist = self.get_spreading_distribution(context_words[0], context_words[1], context_words[2])
        cooc_dist_variance = self.get_distribution_variance(cooc_dist)
        spread_dist_variance = self.get_distribution_variance(spread_dist)
        if cooc_dist_variance < spread_dist_variance:
            guesses = self.get_guesses(spread_dist)
        else:
            guesses = self.get_guesses(cooc_dist)
        return guesses


class AgentAdditiveProbabilityCorpus(AgentJointProbabilityCorpus):
    """ Implements "Additive Probability" mechanism for WSD, which adds together the semantic spreading probability &
        cooccurrence conditional probability distributions, a small alteration from AgentJointProbabilityCorpus."""
    def __init__(self, num_sentences, partition, corpus_utilities, context_type, outside_corpus, spreading=True,
                 clear="never", activation_base=2.0, decay_parameter=0.05, constant_offset=0.0):
        super().__init__(num_sentences, partition, corpus_utilities, context_type, outside_corpus, spreading,
                         clear, activation_base, decay_parameter, constant_offset)

    def create_joint_distribution(self, cooccurrence_dist, spreading_dist):
        """ Creates a joint distribution, assuming the cooccurrence and spreading distributions are already known.
        Overrides joint distribution function from AgentJointProbability (gets the multiplicative joint probability
         distribution) to get the "additive" joint distribution"""
        words = set()
        words.update(list(cooccurrence_dist.keys()))
        words.update(list(spreading_dist.keys()))
        joint_dist = defaultdict(float)
        for word in words:
            if word not in cooccurrence_dist.keys() and word not in spreading_dist.keys():
                joint_dist[word] = 0
            elif word not in cooccurrence_dist.keys():
                joint_dist[word] = spreading_dist[word]
            elif word not in spreading_dist.keys():
                joint_dist[word] = cooccurrence_dist[word]
            else:
                joint_dist[word] = cooccurrence_dist[word] + spreading_dist[word]
        return joint_dist

class AgentAdditiveProbabilityNGrams(AgentJointProbabilityNGrams):
    """ Implements "Additive Probability" mechanism for RAT, which adds together the semantic spreading probability &
            cooccurrence conditional probability distributions, a small alteration from AgentJointProbabilityNGrams."""

    def __init__(self, stopwords, source, ngrams=GoogleNGram('~/ngram'), spreading=True, clear="never",
                 activation_base=2, decay_parameter=0.05, constant_offset=0):
        super().__init__(stopwords, source, ngrams, spreading, clear, activation_base, decay_parameter, constant_offset)

    def create_joint_distribution(self, cooccurrence_dist, spreading_dist):
        """ Creates a joint distribution, assuming the cooccurrence and spreading distributions are already known.
        Overrides joint distribution function from AgentJointProbability (gets the multiplicative joint probability
         distribution) to get the "additive" joint distribution"""
        words = set()
        words.update(list(cooccurrence_dist.keys()))
        words.update(list(spreading_dist.keys()))
        joint_dist = defaultdict(float)
        for word in words:
            if word not in cooccurrence_dist.keys() and word not in spreading_dist.keys():
                joint_dist[word] = 0
            elif word not in cooccurrence_dist.keys():
                joint_dist[word] = spreading_dist[word]
            elif word not in spreading_dist.keys():
                joint_dist[word] = cooccurrence_dist[word]
            else:
                joint_dist[word] = cooccurrence_dist[word] + spreading_dist[word]
        return joint_dist