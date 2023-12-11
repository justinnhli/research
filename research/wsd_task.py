import random
from collections import defaultdict
import nltk
import math
from sentence_long_term_memory import sentenceLTM
from sentence_long_term_memory import SentenceCooccurrenceActivation
from nltk.corpus import wordnet
from nltk.corpus import semcor


def run_wsd(guess_method, activation_base=2, decay_parameter=0.05, constant_offset=0, iterations=1, num_sentences=-1):
    """
    Runs the word sense disambiguation task over the Semcor corpus.
    Parameters:
        guess_method (string): The function used to guess the sense of each word in each sentence. Possibilities are:
            "context_word", "context_sense", "frequency", "naive_sem", "naive_sem_spreading".
        activation_base (float): A parameter in the activation equation.
        decay_parameter (float): A parameter in the activation equation.
        constant_offset (float): A parameter in the activation equation.
        iterations (int): The number of times to run through the corpus (for semantic effects only).
    Returns:
        (float): The raw percent accuracy of the guesses of context_sense_predict_word_sense.
    """
    sentence_list, word_sense_dict = extract_sentences(num_sentences=num_sentences)
    if guess_method == "context_word":
        # The guess_dict is a dictionary with keys the sense of each word in the corpus and values a list of boolean
        # values indicating whether the sense was guessed correctly each time it appears in the corpus.
        guess_dict = get_corpus_accuracy(guess_method="context_word",
                                         sentence_list=sentence_list,
                                         word_sense_dict=word_sense_dict)
    elif guess_method == "context_sense":
        guess_dict = get_corpus_accuracy(guess_method="context_sense",
                                                   sentence_list=sentence_list,
                                                   word_sense_dict=word_sense_dict)
    elif guess_method == "frequency":
        guess_dict = get_corpus_accuracy(guess_method="frequency",
                                         sentence_list=sentence_list,
                                         word_sense_dict=word_sense_dict)
    elif guess_method == "naive_sem":
        sem_network = create_sem_network(sentence_list, spreading=False, time=True, activation_base=activation_base,
                                         decay_parameter=decay_parameter, constant_offset=constant_offset)
        guess_dict = naive_semantic_predict_word_sense(sem_network[0], sentence_list, word_sense_dict, iterations,
                                                       sem_network[1])
    elif guess_method == "naive_sem_spreading":
        sem_network = create_sem_network(sentence_list, spreading=True, time=True, activation_base=activation_base,
                                         decay_parameter=decay_parameter, constant_offset=constant_offset)
        guess_dict = naive_semantic_predict_word_sense(sem_network[0], sentence_list, word_sense_dict, iterations,
                                                       sem_network[1])
    else:
        raise ValueError(guess_method)
    raw_truths = sum(guess_dict.values(), [])
    accuracy = raw_truths.count(True) / len(raw_truths)
    return accuracy


def extract_sentences(num_sentences = -1):
    """
    Runs the word sense disambiguation task.
    Parameters:
        directory_list (list): A list of directories where the tag files from the semcor corpus can be found.
    Returns:
        list: sentence_list (list of sentences in the corpus), word_sense_dict (dictionary with the possible senses of
            each word in the corpus)
    """
    sentence_list = []
    word_sense_dict = defaultdict(set)
    if num_sentences == -1:
        semcor_sents = semcor.tagged_sents(tag="sem")
    else:
        semcor_sents = semcor.tagged_sents(tag="sem")[0:num_sentences]
    for sentence in semcor_sents:
        temp_word_sense_dict = defaultdict(set)
        sentence_word_list = []
        for item in sentence:
            if not isinstance(item, nltk.Tree):
                continue
            if not isinstance(item.label(), nltk.corpus.reader.wordnet.Lemma):
                continue
            corpus_word = (item.label(), item.label().synset())
            sentence_word_list.append(corpus_word)
            temp_word_sense_dict[corpus_word[0].name()].add(corpus_word)
        if len(temp_word_sense_dict) > 1:
            for word, senses in temp_word_sense_dict.items():
                word_sense_dict[word] |= senses
            sentence_list.append(sentence_word_list)
    return sentence_list, word_sense_dict

def create_sem_network(sentence_list, spreading=True, time=False, activation_base=2, decay_parameter=0.05,
                       constant_offset=0):
    """
    Builds a semantic network with each word in the Semcor corpus and its corresponding synonyms, hypernyms, hyponyms,
        holonyms, meronyms, attributes, entailments, causes, also_sees, verb_groups, and similar_tos.
    Parameters:
        sentence_list (Nested String List): A list of the sentences in the Semcor corpus with each word represented by
            a tuple: (lemma, lemma synset).
        spreading (bool): Whether to include the effects of spreading in creating the semantic network.
        activation_base (float): A parameter in the activation equation.
        decay_parameter (float): A parameter in the activation equation.
        constant_offset (float): A parameter in the activation equation.
    Returns:
        network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
    """
    network = sentenceLTM(
        activation_cls=(lambda ltm:
                        SentenceCooccurrenceActivation(
                            ltm,
                            activation_base=activation_base,
                            constant_offset=constant_offset,
                            decay_parameter=decay_parameter
                        )))
    timer = 1
    semcor_words = sum(sentence_list, [])
    if spreading:
        for sentence in sentence_list:
            for word in sentence:
                # if not network.retrievable(word):
                syn = word[1]
                lemma = word[0]
                synonyms = [(synon, syn) for synon in syn.lemmas() if (synon, syn) in semcor_words and synon != lemma]
                # These are all synsets.
                synset_relations = [syn.hypernyms(), syn.hyponyms(),
                                  syn.member_holonyms() + syn.substance_holonyms() + syn.part_holonyms(),
                                  syn.member_meronyms() + syn.substance_meronyms() + syn.part_meronyms(),
                                  syn.attributes(), syn.entailments(), syn.causes(), syn.also_sees(), syn.verb_groups(),
                                  syn.similar_tos()]
                lemma_relations = []
                for ii in range(len(synset_relations)):
                    lemma_relations.append([])
                    for jj in range(len(synset_relations[ii])):
                        syn_lemmas = synset_relations[ii][jj].lemmas()
                        for lemma in syn_lemmas:
                            lemma_relations[ii].append((lemma, synset_relations[ii][jj]))
                for ii in range(len(lemma_relations)):
                    lemma_relations[ii] = [word_tuple for word_tuple in set(lemma_relations[ii]) if word_tuple in
                                           semcor_words and word_tuple != word]
                print(lemma_relations)
                network.store(mem_id=word,
                              time=timer,
                              synonyms=synonyms,
                              hypernyms=lemma_relations[0],
                              hyponyms=lemma_relations[1],
                              holynyms=lemma_relations[2],
                              meronyms=lemma_relations[3],
                              attributes=lemma_relations[4],
                              entailments=lemma_relations[5],
                              causes=lemma_relations[6],
                              also_sees=lemma_relations[7],
                              verb_groups=lemma_relations[8],
                              similar_tos=lemma_relations[9])
                timer += 1
    else:
        for sentence in sentence_list:
            for word in sentence:
                network.store(mem_id=word, time=timer)
            timer += 1
    if time:
        return network, timer
    else:
        return network


def precompute_word_sense(sentence_list):
    word_counts = defaultdict(int)
    sense_counts = defaultdict(int)
    for sentence in sentence_list:
        for sense in sentence:
            word_counts[sense[0].name()] += 1
            sense_counts[sense] += 1
    return word_counts, sense_counts


def precompute_cooccurrences(sentence_list):
    """
    Precomputes word-word, sense-word, sense-sense cooccurrences and the number of times each sense shows up in the
    corpus.
    Parameters:
        sentence_list (list): List of all sentences in the corpus (from extract_sentences function).
    Returns:
        list: word_word_cooccurrences (a dictionary with keys as word/word pairs and values the cooccurrences associated)
            sense_word_cooccurrences (a dictionary with keys as sense/word pairs and values the cooccurrences associated)
            sense_sense_cooccurrences (a dictionary with keys as sense/sense pairs and values the cooccurrences associated)
            sense_frequencies (a dictionary with keys a word sense and values the occurrences of that sense in the corpus)
    """
    word_word_cooccurrences = defaultdict(int)
    sense_word_cooccurrences = defaultdict(int)
    sense_sense_cooccurrences = defaultdict(int)
    sense_frequencies = defaultdict(int)
    for sentence in sentence_list:
        for target_index in range(len(sentence)):
            target_sense = sentence[target_index]
            target_word = target_sense[0].name()
            sense_frequencies[target_sense] += 1
            for other_index in range(len(sentence)):
                if target_index != other_index:
                    other_sense = sentence[other_index]
                    other_word = other_sense[0].name()
                    word_word_cooccurrences[(target_word, other_word)] += 1
                    sense_word_cooccurrences[(target_sense, other_word)] += 1
                    sense_sense_cooccurrences[(target_sense, other_sense)] += 1
    return word_word_cooccurrences, sense_word_cooccurrences, sense_sense_cooccurrences, sense_frequencies

def guess_word_sense_context_word(target_index, sentence, word_sense_dict, sense_word_cooccurrences, word_word_cooccurrences):
    """
    Guesses the sense of a "target" word based on how often it cooccurs with other words in the sentence
    Parameters:
        target_sense (tuple): The "target" word in the sentence: a lemma, synset tuple.
        sentence (list): A list of lemma/synset tuples referring to all words in the sentence (including the target sense)
        word_sense_dict (dict): dictionary with the possible senses of each word in the corpus
        sense_word_cooccurrences (dict): a dictionary with keys as sense/word pairs and values the cooccurrences associated
        word_word_cooccurrences (dict): a dictionary with keys as word/word pairs and values the cooccurrences associated
    Returns:
        bool: Returns true if the correct word sense for the target word is guessed, and false if incorrect.
    """
    max_score = -float("inf")
    max_sense = None
    target_sense = sentence[target_index]
    target_word = target_sense[0].name()
    for target_sense_candidate in word_sense_dict[target_sense[0].name()]:
        aggregate = 0
        for other_index in range(len(sentence)):
            if other_index != target_index:
                other_word = sentence[other_index][0].name()
                if (target_sense_candidate, other_word) not in sense_word_cooccurrences:
                    numerator = 0
                else:
                    numerator = sense_word_cooccurrences[(target_sense_candidate, other_word)]
                denominator = word_word_cooccurrences[(target_word, other_word)]
                aggregate += numerator/denominator
        if aggregate > max_score:
            max_score = aggregate
            max_sense = target_sense_candidate
    return max_sense


def guess_word_sense_context_sense(target_index, sentence, word_sense_dict, sense_word_cooccurrences, sense_sense_cooccurrences):
    """
        Guesses the sense of a "target" word based on how often it cooccurs with other sense-specific words in the sentence
        Parameters:
            target_sense (tuple): The "target" word in the sentence: a lemma, synset tuple.
            sentence (list): A list of lemma/synset tuples referring to all words in the sentence (including the target sense)
            word_sense_dict (dict): dictionary with the possible senses of each word in the corpus
          .  sense_word_cooccurrences (dict): a dictionary with keys as sense/word pairs and values the cooccurrences associated
            sense_sense_cooccurrences (dict): a dictionary with keys as sense/sense pairs and values the cooccurrences associated
        Returns:
            bool: Returns true if the correct word sense for the target word is guessed, and false if incorrect.
        """
    max_score = -float("inf")
    max_sense = None
    target_sense = sentence[target_index]
    target_word = target_sense[0].name()
    for target_sense_candidate in word_sense_dict[target_sense[0].name()]:
        aggregate = 0
        for other_index in range(len(sentence)):
            if other_index != target_index:
                other_sense = sentence[other_index]
                if (target_sense_candidate, other_sense) not in sense_sense_cooccurrences:
                    numerator = 0
                else:
                    numerator = sense_sense_cooccurrences[(target_sense_candidate, other_sense)]
                denominator = sense_word_cooccurrences[(other_sense, target_word)]
                aggregate += numerator/denominator
        if aggregate > max_score:
            max_score = aggregate
            max_sense = target_sense_candidate
    return max_sense

def guess_word_sense_frequency(target_index, sentence, word_sense_dict, sense_frequencies):
    """
    Guesses the sense of a target word based on the most frequent sense of that word in the corpus.
    Parameters:
        target_sense (tuple): The "target" word in the sentence: a lemma, synset tuple.
        word_sense_dict (dict): dictionary with the possible senses of each word in the corpus.
        sense_frequencies (dict): dictionary with keys a word sense and values the occurrences of that sense in the corpus
    Returns:
        bool: Returns true if the correct word sense for the target word is guessed, and false if incorrect.
    """
    target_sense = sentence[target_index]
    target_word = target_sense[0].name()
    max_score = -float("inf")
    max_sense = None
    for target_sense_candidate in word_sense_dict[target_word]:
        if sense_frequencies[target_sense_candidate] > max_score:
            max_score = sense_frequencies[target_sense_candidate]
            max_sense = target_sense_candidate
    return max_sense


def get_corpus_accuracy(guess_method, sentence_list, word_sense_dict):
    """
    Guesses the word sense for every word in the corpus based on a specified guess method.
    Parameters:
        guess_method (string): Which method to use when guessing the sense of each word. Possibilities are "context_word",
            "context_sense", and "frequency".
        sentence_list (list): A nested list of all sentences in the corpus with each word referenced by a lemma/synset
            tuple.
        word_sense_dict (dict): dictionary with the possible senses of each word in the corpus
    Returns:
        list: A list of how accurate guesses were based on each sense-specific word in the corpus, and the overall accuracy
            of the guessing method.
    """
    word_word_cooccurrences, sense_word_cooccurrences, sense_sense_cooccurrences, sense_frequencies = precompute_cooccurrences(sentence_list)
    accuracy_dict = defaultdict(list)
    for sentence in sentence_list:
        for target_index in range(len(sentence)):
            if guess_method == "context_word":
                guess = guess_word_sense_context_word(target_index,
                                                               sentence,
                                                               word_sense_dict,
                                                               sense_word_cooccurrences,
                                                               word_word_cooccurrences)
            elif guess_method == "context_sense":
                guess = guess_word_sense_context_sense(target_index,
                                                                sentence,
                                                                word_sense_dict,
                                                                sense_word_cooccurrences,
                                                                sense_sense_cooccurrences)
            elif guess_method == "frequency":
                guess = guess_word_sense_frequency(target_index,
                                                            sentence,
                                                            word_sense_dict,
                                                            sense_frequencies)
            target_sense = sentence[target_index]
            if target_sense not in accuracy_dict:
                accuracy_dict[target_sense] = []
            if target_sense == guess:
                accuracy_dict[target_sense].append(True)
            else:
                accuracy_dict[target_sense].append(False)
    return accuracy_dict




def naive_semantic_predict_word_sense(sem_network, sentence_list, word_sense_dict, iterations, time):
    """
    Predicts the word sense by choosing the sense with the highest activation (with spreading optional)
    Parameters:
        sem_network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
        sentence_list (list): A nested list of all sentences in the corpus with each word referenced by a lemma/synset
            tuple.
        word_sense_dict (dict): dictionary with the possible senses of each word in the corpus
        iterations (int): The number of times to run through the corpus and guess the sense of each word.
        time (float): The starting time
    Returns:
        dict: A dictionary of each word in the corpus and whether guesses for that word sense were correct or incorrect.
    """
    timer = time
    guess_dict = {}
    for iter in range(iterations):
        for sentence in sentence_list:
            for word in sentence:
                # senses = sem_network.sense_query(word=word[0], time=timer)
                senses = word_sense_dict[word[0].name()]
                max_act = -1
                for sense in senses:
                    sense_act = sem_network.get_activation(mem_id=sense, time=timer)
                    if sense_act > max_act:
                        max_act = sense_act
                        sense_guess = sense
                if word not in guess_dict.keys():
                    guess_dict[word] = []
                if sense_guess == word:
                    guess_dict[word].append(True)
                else:
                    guess_dict[word].append(False)
                sem_network.store(mem_id=word, time=timer)
                sem_network.store(mem_id=sense_guess, time=timer)
                timer += 1
    return guess_dict


def dummy_predict_word_sense(sentence_list):
    """
    Dummy function to predict the word sense of all of the words in a sentence.
    Parameters:
        sentence_list (String list): Formatted like [[word, part of speech, correct word sense], ...] to take in
            information about each word in a sentence.
    Returns:
        accuracy_list (Boolean list): A list that indicates whether the sense of each word in the sentence was determined
            correctly or not.
    """
    accuracy_list = []
    for word in sentence_list:
        guess_sense = random.randint(1, 3)
        if guess_sense == int(word[2]):
            accuracy_list.append(True)
        else:
            accuracy_list.append(False)
    return accuracy_list


# Testing...
#print(run_wsd(guess_method="context_word"))
#print(run_wsd(guess_method="context_sense"))
print(run_wsd(guess_method="naive_sem_spreading", iterations=3, num_sentences=300))
