from sentence_long_term_memory import sentenceLTM
from nltk.corpus import wordnet as wn_corpus
from corpus_utilities import *
from sentence_cooccurrence_activation import SentenceCooccurrenceActivation
from n_gram_cooccurrence.google_ngrams import *


class AgentSpreading:
    """ Implements the spreading agent. """

    def __init__(self, spreading=True, clear="never", activation_base=2, decay_parameter=0.05, constant_offset=0):
        self.spreading = spreading
        self.clear = clear
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter
        self.constant_offset = constant_offset

    def clear_sem_network(self, network, start_time=1):
        """
            Clears the semantic network by resetting activations to a certain "starting time".
            Parameters:
                sem_network (sentenceLTM): Network to clear.
                start_time (int): The network will be reset so that activations only at the starting time and before the
                    starting time remain.
            Returns:
                sentenceLTM: Cleared semantic network.
            """
        activations = network.activation.activations
        activated_words = activations.keys()
        for word in activated_words:
            activations[word] = [act for act in activations[word] if act[0] <= start_time]
        return network

    def do_wsd(self, word, context, time, network):
        """ Finds guesses for the WSD task."""
        raise NotImplementedError

    def do_rat(self, context1, context2, context3, network):
        """ Finds guesses for the RAT test."""
        raise NotImplementedError


class AgentSpreadingCorpus(AgentSpreading):

    def __init__(self, corpus_utilities, outside_corpus, spreading=True, clear="never", activation_base=2.0,
                 decay_parameter=0.05, constant_offset=0.0):
        """
        Parameters:
            corpus_utilities (class): A class of functions useful for corpus mechanisms, specific to the partition of the
                Semcor corpus used.
            outside_corpus (bool): True if semantic relations can be considered outside the corpus and False if semantic
            spreading (bool): Whether to include the effects of spreading in creating the semantic network.
            clear (string): How often to clear the network. Possible values are "never", "sentence", or "word",
                indicating that the network is never cleared, cleared after each sentence, or cleared after each word.
            activation_base (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            constant_offset (float): A parameter in the activation equation.
        """
        super().__init__(spreading, clear, activation_base, decay_parameter, constant_offset)
        self.corpus_utilities = corpus_utilities
        self.sentence_list = corpus_utilities.get_sentence_list()
        self.outside_corpus = outside_corpus

    def get_semantic_relations_dict(self, outside_corpus):
        """
            Gets the words related to each word in sentence_list and builds a dictionary to make the semantic network
            Parameters:
                sentence_list (list): list of all sentences or a partition of n sentences in the corpus
                partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking
                    at sentences 10000 - 14999.
                outside_corpus (bool): True if semantic relations can be considered outside the corpus and False if semantic
                    relations are only considered from words inside the corpus.
            Returns:
                (dict) A dictionary with the semantic relations for every unique word in sentence_list
        """
        sem_rel_path = "./semantic_relations/semantic_relations_list"
        if not outside_corpus:
            sem_rel_path = sem_rel_path + "_inside_corpus"
        if len(self.sentence_list) == 30195:
            sem_rel_path = sem_rel_path + ".json"
        elif self.corpus_utilities.partition == 1:
            sem_rel_path = sem_rel_path + "_" + str(len(self.sentence_list)) + ".json"
        else:
            sem_rel_path = sem_rel_path + "_" + str(len(self.sentence_list)) + "_partition_" + str(
                self.corpus_utilities.partition) + ".json"
        if not os.path.isfile(sem_rel_path):
            corpus_utilities = CorpusUtilities()
            semantic_relations_list = []
            # These are all the words in the corpus.
            semcor_words = set(sum(self.sentence_list, []))
            counter = 0
            for word in semcor_words:
                counter += 1
                syn = wn_corpus.synset(word[1])
                synonyms = [corpus_utilities.lemma_to_tuple(synon) for synon in syn.lemmas() if
                            corpus_utilities.lemma_to_tuple(synon) != word]
                # These are all synsets.
                synset_relations = [syn.hypernyms(), syn.hyponyms(),
                                    syn.member_holonyms() + syn.substance_holonyms() + syn.part_holonyms(),
                                    syn.member_meronyms() + syn.substance_meronyms() + syn.part_meronyms(),
                                    syn.attributes(), syn.entailments(), syn.causes(), syn.also_sees(),
                                    syn.verb_groups(), syn.similar_tos()]
                lemma_relations = []
                for relation in range(len(synset_relations)):
                    lemma_relations.append([])
                    # Getting each of the synsets above in synset_relations.
                    for syn in range(len(synset_relations[relation])):
                        # Getting the lemmas in each of the synset_relations synsets.
                        syn_lemmas = synset_relations[relation][syn].lemmas()
                        # Checking that lemmas from relations are in the corpus if outside_corpus=False
                        if not outside_corpus:
                            syn_lemmas = [lemma for lemma in syn_lemmas if lemma in semcor_words]
                        # Adding each lemma to the list
                        for syn_lemma in syn_lemmas:
                            lemma_tuple = corpus_utilities.lemma_to_tuple(syn_lemma)
                            if word != lemma_tuple:
                                lemma_relations[relation].append(lemma_tuple)
                word_sem_rel_subdict = self.create_word_sem_rel_dict(synonyms=synonyms,
                                                                     hypernyms=lemma_relations[0],
                                                                     hyponyms=lemma_relations[1],
                                                                     holonyms=lemma_relations[2],
                                                                     meronyms=lemma_relations[3],
                                                                     attributes=lemma_relations[4],
                                                                     entailments=lemma_relations[5],
                                                                     causes=lemma_relations[6],
                                                                     also_sees=lemma_relations[7],
                                                                     verb_groups=lemma_relations[8],
                                                                     similar_tos=lemma_relations[9])
                # Adding pairs of word & the dictionary containing its relations to the big json list (since json doesn't let lists be keys)
                # But we can still keep the word_sem_rel_subdict intact since its keys are strings
                semantic_relations_list.append([word, word_sem_rel_subdict])
            sem_rel_file = open(sem_rel_path, 'w')
            json.dump(semantic_relations_list, sem_rel_file)
            sem_rel_file.close()
        semantic_relations_list = json.load(open(sem_rel_path))
        semantic_relations_dict = {}
        for pair in semantic_relations_list:
            key = tuple(pair[0])
            val_dict = pair[1]
            for val_key in ["synonyms", "hypernyms", "hyponyms", "holonyms", "meronyms", "attributes",
                            "entailments",
                            "causes", "also_sees", "verb_groups", "similar_tos"]:
                list_val_vals = val_dict[val_key]
                tuple_val_vals = []
                for val_val in list_val_vals:
                    tuple_val_vals.append(tuple(val_val))
                val_dict[val_key] = tuple_val_vals
            semantic_relations_dict[key] = val_dict
        return semantic_relations_dict

    def create_word_sem_rel_dict(self, synonyms, hypernyms, hyponyms, holonyms, meronyms, attributes,
                                 entailments, causes, also_sees, verb_groups, similar_tos):
        """
        Creates a semantic relations dictionary with given semantic relations for a word.
        Parameters:
            synonyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            hypernyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            hyponyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            holonyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            meronyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            attributes (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            entailments (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            causes (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            also_sees (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            verb_groups (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            similar_tos (list) A list of word relations drawn from the synset a word belongs to from the nltk package
        Returns: A dictionary with the semantic relations for one word in the corpus.
        """
        sem_rel_dict = {"synonyms": set(synonyms), "hypernyms": set(hypernyms), "hyponyms": set(hyponyms),
                        "holonyms": set(holonyms), "meronyms": set(meronyms), "attributes": set(attributes),
                        "entailments": set(entailments), "causes": set(causes), "also_sees": set(also_sees),
                        "verb_groups": set(verb_groups), "similar_tos": set(similar_tos)}
        for rel in sem_rel_dict.keys():
            vals = sem_rel_dict[rel]
            string_vals = []
            for val in vals:
                string_vals.append(list(val))
            sem_rel_dict[rel] = string_vals
        return sem_rel_dict

    def create_sem_network(self):
        """
        Builds a semantic network with each word in the Semcor corpus and its corresponding synonyms, hypernyms,
            hyponyms, holonyms, meronyms, attributes, entailments, causes, also_sees, verb_groups, and similar_tos.
             Note that all words are stored at time 1.
        Returns:
            network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
        """
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        semantic_relations_dict = self.get_semantic_relations_dict(self.outside_corpus)
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

    def do_wsd(self, word, context, time, network):
        """
        Gets guesses for a trial of the WSD.
        Parameters:
            word (sense-word tuple): The word to guess the sense of, the "target word" (should not have sense-identifying
                information).
            context (list): A list of all possible senses of the target word, often obtained from the word sense
                dictionary.
            time (int): The time to calculate activations at.
            network (sentenceLTM): Semantic network
        """
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        max_act = float('-inf')
        max_guess = []
        for candidate in context:
            candidate_act = network.get_activation(mem_id=candidate, time=time)
            if candidate_act > max_act:
                max_act = candidate_act
                max_guess = [candidate]
            elif candidate_act == max_act:
                max_guess.append(candidate)
        network.store(mem_id=word, time=time, spread_depth=spread_depth)
        for elem in max_guess:
            network.store(mem_id=elem, time=time, spread_depth=spread_depth)
        return max_guess


class AgentSpreadingNGrams(AgentSpreading):
    """ Implements spreading on google ngrams. """
    def __init__(self, sem_rel_dict, stopwords, ngrams=GoogleNGram('~/ngram'), spreading=True, clear="never", activation_base=2,
                 decay_parameter=0.05, constant_offset=0):
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
        super().__init__(spreading, clear, activation_base, decay_parameter, constant_offset)
        self.ngrams = ngrams
        self.sem_rel_dict = self.filter_sem_rel_dict(sem_rel_dict)
        self.stopwords = stopwords


    def filter_sem_rel_dict(self, sem_rel_dict):
        """
        Filters the sem rel dict for stopwords to ensure that all words are valid.
        Parameters:
            sem_rel_dict (dictionary): A dictionary containing all semantic relations (the values) for each word
                (the keys) from the SWOWEN, SFFAN, (or both) databases.
        Returns:
            (dict) filtered semantic relations dictionary.
        """
        keys = sem_rel_dict.keys()
        for key in keys:
            rels = sem_rel_dict[key]
            words = [word for word in rels if word.lower() not in self.stopwords]
            sem_rel_dict[key] = words
        return sem_rel_dict


    def create_sem_network(self):
        """ Creates a semantic network from the semantic relations list. """
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

    def do_rat(self, context1, context2, context3, network):
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
        for context in [context1, context2, context3]:
            network.store(mem_id=context, time=2, spread_depth=spread_depth)
        max_act = -float("inf")
        guesses = []
        elements = sorted(set(network.knowledge.keys()))
        for elem in elements:
            if elem in context:
                continue
            elem_act = network.get_activation(mem_id=elem, time=3)
            if elem_act is None:
                continue
            elif elem_act > max_act:
                max_act = elem_act
                guesses = [elem]
            elif elem_act == max_act:
                guesses.append(elem)
        return guesses
