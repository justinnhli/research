import os.path
from collections import defaultdict
import datetime
import json
import random
from sentence_long_term_memory import sentenceLTM
import agent_cooccurrence
from corpus_utilities import CorpusUtilities
from agent_oracle import AgentOracleCorpus
from agent_cooccurrence import AgentCooccurrenceCorpus
from agent_spreading_thresh_cooccurrence import AgentSpreadingThreshCoocCorpus
from agent_cooccurrence_thresh_spreading import AgentCoocThreshSpreadingCorpus
from agent_spreading import AgentSpreadingCorpus


def run_wsd(guess_method, activation_base=2, decay_parameter=0.05, constant_offset=0, iterations=1, num_sentences=-1,
            partition=1, outside_corpus=False, clear_network="never", context_type="word", whole_corpus=True):
    """
    Runs the word sense disambiguation task over the Semcor corpus (or a subset of it).
    Parameters:
        guess_method (string): The function used to guess the sense of each word in each sentence. Possibilities are:
            "cooc", "frequency", "naive_semantic", "naive_semantic_spreading", "oracle". Integrated mechanism
            possibilities include: "sem_thresh_cooc", "cooc_thresh_sem"
        activation_base (float): A parameter in the activation equation.
        decay_parameter (float): A parameter in the activation equation.
        constant_offset (float): A parameter in the activation equation.
        iterations (int): The number of times to run through the corpus (for semantic effects only).
        num_sentences (int): The number of sentences from the corpus to use in the task. The first n sentences
            from the corpus are used and if n=-1, all sentences from the corpus are used.
        partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking at
            sentences 10000 - 14999.
        outside_corpus (bool): True if semantic relations can be considered outside the corpus and False if semantic
            relations are only considered from words inside the corpus.
        clear_network (string): How often to clear the network. Possible values are "never", "sentence", or "word",
            indicating that the network is never cleared, cleared after each sentence, or cleared after each word.
        context_type (string): Indicates for cooccurrence dependent corpus mechanisms, whether we know the sense of the
            context words ("sense") or not ("word")
        whole_corpus (bool): For cooccurrence dependent corpus mechanisms, whether to include cooccurrent relations from
            the whole corpus (True) or not (False).
    Returns:
        (float): The raw percent accuracy of the guesses of context_sense_predict_word_sense.
    """
    corpus_utilities = CorpusUtilities(num_sentences, partition)
    if guess_method == "cooc" and context_type == "word":
        # The guess_dict is a dictionary with keys the sense of each word in the corpus and values a list of boolean
        # values indicating whether the sense was guessed correctly each time it appears in the corpus.
        guess_dicts = get_corpus_accuracy(guess_method="cooc",
                                          context_type="word",
                                          corpus_utilities=corpus_utilities,
                                          iterations=iterations,
                                          whole_corpus=whole_corpus)
    elif guess_method == "cooc" and context_type == "sense":
        guess_dicts = get_corpus_accuracy(guess_method="cooc",
                                          context_type="sense",
                                          corpus_utilities=corpus_utilities,
                                          iterations=iterations,
                                          whole_corpus=whole_corpus)
    elif guess_method == "frequency":
        guess_dicts = get_corpus_accuracy(guess_method="frequency",
                                          corpus_utilities=corpus_utilities,
                                          iterations=iterations)
    elif guess_method == "naive_semantic":
        guess_dicts = get_corpus_accuracy("naive_semantic",
                                          corpus_utilities=corpus_utilities,
                                          clear_network=clear_network,
                                          activation_base=activation_base,
                                          decay_parameter=decay_parameter,
                                          constant_offset=constant_offset,
                                          iterations=iterations,
                                          outside_corpus=outside_corpus)
    elif guess_method == "naive_semantic_spreading":
        guess_dicts = get_corpus_accuracy("naive_semantic_spreading",
                                          corpus_utilities=corpus_utilities,
                                          clear_network=clear_network,
                                          activation_base=activation_base,
                                          decay_parameter=decay_parameter,
                                          constant_offset=constant_offset,
                                          iterations=iterations,
                                          outside_corpus=outside_corpus)
    elif guess_method == "oracle":
        guess_dicts = get_corpus_accuracy("oracle",
                                          context_type=context_type,
                                          corpus_utilities=corpus_utilities,
                                          clear_network=clear_network,
                                          activation_base=activation_base,
                                          decay_parameter=decay_parameter,
                                          constant_offset=constant_offset,
                                          iterations=iterations,
                                          outside_corpus=outside_corpus)
    elif guess_method == "sem_thresh_cooc":
        guess_dicts = get_corpus_accuracy("sem_thresh_cooc",
                                          corpus_utilities=corpus_utilities,
                                          iterations=iterations,
                                          context_type=context_type)
    elif guess_method == "cooc_thresh_sem":
        guess_dicts = get_corpus_accuracy("cooc_thresh_sem",
                                          corpus_utilities=corpus_utilities,
                                          clear_network=clear_network,
                                          activation_base=activation_base,
                                          decay_parameter=decay_parameter,
                                          constant_offset=constant_offset,
                                          iterations=iterations,
                                          outside_corpus=outside_corpus,
                                          context_type=context_type,
                                          whole_corpus=whole_corpus)
    else:
        raise ValueError(guess_method)
    accuracies = []
    # Calculating the upper and lower accuracy bounds for each iteration (normally only one)
    for guess_dict in guess_dicts:
        guesses = sum(list(guess_dict.values()), [])
        upper_accuracies = 0
        lower_accuracies = 0
        total = len(guesses)
        for guess_list in guesses:
            if len(set(guess_list)) == 2:
                upper_accuracies += 1
            elif len(set(guess_list)) == 1 and guess_list[0]:
                upper_accuracies += 1
                lower_accuracies += 1
        accuracies.append([lower_accuracies / total, upper_accuracies / total])
    return accuracies


def get_corpus_accuracy(guess_method, corpus_utilities, clear_network="never", activation_base=2,
                        decay_parameter=0.05, constant_offset=0, iterations=1, outside_corpus=False,
                        context_type="word", whole_corpus=False, index_info=False):
    """
    Guesses the word sense for every word in the corpus based on a specified guess method.
    Parameters:
        guess_method (string): Which method to use when guessing the sense of each word. Possibilities are "cooc",
            "naive_semantic_spreading", "naive_semantic", "oracle", and "frequency". Integrated mechanisms include
            "sem_thresh_cooc", "cooc_thresh_sem".
        corpus_utilities (class): A class of functions useful for corpus mechanisms, specific to the partition of the
            Semcor corpus used
        clear_network (string): How often to clear the network. Possible values are "never", "sentence", or "word",
            indicating that the network is never cleared, cleared after each sentence, or cleared after each word.
        activation_base (float): A parameter in the activation equation.
        decay_parameter (float): A parameter in the activation equation.
        constant_offset (float): A parameter in the activation equation.
        iterations (int): The number of times to run through the corpus (for semantic effects only).
        outside_corpus (bool): True if semantic relations can be considered outside the corpus and False if semantic
            relations are only considered from words inside the corpus.
        context_type (string): Indicates for cooccurrence dependent corpus mechanisms, whether we know the sense of the
            context words ("sense") or not ("word")
        whole_corpus (bool): For cooccurrence dependent corpus mechanisms, whether to include cooccurrent relations from
            the whole corpus (True) or not (False).
        index_info (bool): True indicates that the first entry in each dictionary value will indicate where in the
         corpus the word is from in an "index tuple": (sentence number out of corpus, word number out of sentence)
    Returns:
        list: A list of how accurate guesses were based on each sense-specific word in the corpus, and the overall accuracy
            of the guessing method.
    """
    # For saving the guesses to a file so data can be surveyed later
    curr_time = str(datetime.datetime.now())
    sentence_list = corpus_utilities.get_sentence_list()
    word_sense_dict = corpus_utilities.get_word_sense_dict()
    if iterations == 1:
        path = "./results/" + guess_method + "_"
    else:
        path = "./results/" + guess_method + "_iter_"
    if "semantic" in guess_method:
        path += str(len(sentence_list)) + "_sents_" + clear_network + "_clear_" + str(outside_corpus) + \
                "_outside_corpus_" + str(corpus_utilities.partition) + "_partition"
    else:
        path += str(len(sentence_list)) + "_sents_" + str(corpus_utilities.partition) + "_partition"
    if index_info:
        path += "_index"
    path += "_accuracy_list_" + curr_time + ".json"
    # Creating the agents for each guessing mechanism
    if (guess_method == "cooc") and (whole_corpus is False): # Only uses cooccurrent relations within each partition.
        cooc_agent = agent_cooccurrence.AgentCooccurrenceCorpus(corpus_utilities.num_sentences,
                                                                corpus_utilities.partition,
                                                                corpus_utilities,
                                                                context_type=context_type)
    elif (guess_method == "cooc") and (whole_corpus is True): # Uses cooccurrent relations across the whole corpus.
        whole_corpus_utilities = CorpusUtilities(-1, 1)
        cooc_agent = agent_cooccurrence.AgentCooccurrenceCorpus(whole_corpus_utilities.num_sentences,
                                                                whole_corpus_utilities.partition,
                                                                whole_corpus_utilities,
                                                                context_type=context_type)
    elif guess_method == "naive_semantic_spreading":
        sem_agent = AgentSpreadingCorpus(corpus_utilities, outside_corpus=outside_corpus, spreading=True,
                                         clear=clear_network, activation_base=activation_base,
                                         decay_parameter=decay_parameter, constant_offset=constant_offset)
        sem_network = sem_agent.create_sem_network()
        timer = 2  # All semantic connections stored at time 1, so start the timer at the next timestep.
    elif guess_method == "naive_semantic":
        sem_agent = AgentSpreadingCorpus(corpus_utilities, outside_corpus=outside_corpus, spreading=False,
                                         clear=clear_network, activation_base=activation_base,
                                         decay_parameter=decay_parameter, constant_offset=constant_offset)
        sem_network = sem_agent.create_sem_network()
        timer = 2  # All semantic connections stored at time 1, so start the timer at the next timestep.
    elif guess_method == "oracle":
        # The oracle needs different timers for its semantic mechanisms because they're reset at different stages of
        # the task.
        timer_never = 2
        timer_word = 2
        timer_sentence = 2
        oracle_agent = AgentOracleCorpus(corpus_utilities=corpus_utilities, outside_corpus=outside_corpus,
                                         activation_base=activation_base, decay_parameter=decay_parameter,
                                         constant_offset=constant_offset)
    elif guess_method == "cooc_thresh_sem":
        cooc_thresh_sem_agent = AgentCoocThreshSpreadingCorpus(context_type=context_type, whole_corpus=whole_corpus,
                                                               corpus_utilities=corpus_utilities,
                                                               outside_corpus=outside_corpus, spreading=True,
                                                               clear=clear_network, activation_base=activation_base,
                                                               decay_parameter=decay_parameter,
                                                               constant_offset=constant_offset)
        sem_network = cooc_thresh_sem_agent.create_sem_network()
        timer = 2
    elif guess_method == "sem_thresh_cooc":
        sem_thresh_cooc_agent = AgentSpreadingThreshCoocCorpus(num_sentences=corpus_utilities.num_sentences,
                                                               partition=corpus_utilities.partition,
                                                               corpus_utilities=corpus_utilities,
                                                               context_type=context_type,
                                                               outside_corpus=outside_corpus)
    elif guess_method == "frequency":
        sense_counts = corpus_utilities.get_sense_counts()
    # Checking for valid inputs to clear_network and context_type
    if clear_network != "never" and clear_network != "sentence" and clear_network != "word":
        raise ValueError(clear_network)
    if context_type != "word" and context_type != "sense":
        raise ValueError(context_type)
    accuracy_dicts = []  # List to store dictionaries of correct guesses for each iteration
    # Looping through each iteration...
    for iter in range(iterations):
        accuracy_dict = defaultdict(list)
        # Clear semantic network each iteration if > 1, so you don't have to remake the network (expensive)
        if "naive_semantic" in guess_method and iter > 1:
            sem_network = sem_agent.clear_sem_network(sem_network, 1)
            timer = 2  # reset timer
        elif guess_method == "cooc_thresh_sem" and iter > 1:
            sem_network = cooc_thresh_sem_agent.clear_sem_network(sem_network, 1)
            timer = 2
        elif guess_method == "oracle" and iter > 1:
            oracle_agent.sem_never_agent.clear_sem_network(oracle_agent.sem_never_network, 1)
            oracle_agent.sem_nospread_agent.clear_sem_network(oracle_agent.sem_nospread_network, 1)
            timer_never = 2
        # Looping through each sentence in the corpus.
        for sentence_index in range(len(sentence_list)):
            sentence = sentence_list[sentence_index]
            # Resetting semantic networks if clear_network == "sentence"
            if "naive_semantic" in guess_method and clear_network == "sentence":
                sem_network = sem_agent.clear_sem_network(sem_network, 1)
                timer = 2
            elif guess_method == "oracle" and clear_network == "sentence":
                sem_sentence_network = oracle_agent.sem_sentence_agent.clear_sem_network(
                    oracle_agent.sem_sentence_network, 1)
                timer_sentence = 2
            elif guess_method == "cooc_thresh_sem" and clear_network == "sentence":
                sem_network = cooc_thresh_sem_agent.clear_sem_network(sem_network, 1)
                timer = 2
            # Looping through each word in each sentence
            for target_index in range(len(sentence)):
                word = sentence[target_index]
                # Clearing semantic networks if clear_network == "word"
                if "naive_semantic" in guess_method and clear_network == "word":
                    sem_network = sem_agent.clear_sem_network(sem_network, 1)
                    timer = 2  # reset timer.
                elif guess_method == "oracle" and clear_network == "word":
                    sem_word_network = oracle_agent.sem_word_agent.clear_sem_network(oracle_agent.sem_word_network, 1)
                    timer_word = 2
                elif guess_method == "cooc_thresh_sem" and clear_network == "word":
                    sem_network = cooc_thresh_sem_agent.clear_sem_network(sem_network, 1)
                    timer = 2
                # Getting the guesses of the sense of each word for each guessing mechanism.
                if guess_method == "cooc":
                    guesses = cooc_agent.do_wsd(target_index, sentence)
                elif guess_method == "frequency":
                    guesses = guess_word_sense_frequency(target_index,
                                                         sentence,
                                                         word_sense_dict,
                                                         sense_counts)
                elif guess_method == "naive_semantic":
                    guesses = sem_agent.do_wsd(word=sentence[target_index],
                                               context=word_sense_dict[word[0]],
                                               time=timer,
                                               network=sem_network)
                    timer += 1
                elif guess_method == "naive_semantic_spreading":
                    guesses = sem_agent.do_wsd(word=sentence[target_index],
                                               context=word_sense_dict[word[0]],
                                               time=timer,
                                               network=sem_network)
                    timer += 1
                elif guess_method == "oracle":
                    guesses = oracle_agent.do_wsd(target_index, sentence, timer_word, timer_sentence, timer_never)
                    timer_word += 1
                    timer_sentence += 1
                    timer_never += 1
                elif guess_method == "sem_thresh_cooc":
                    guesses = sem_thresh_cooc_agent.do_wsd(target_index,
                                                           sentence)
                elif guess_method == "cooc_thresh_sem":
                    guesses = cooc_thresh_sem_agent.do_wsd(word=sentence[target_index],
                                                           context=word_sense_dict[word[0]],
                                                           time=timer,
                                                           network=sem_network)
                    timer += 1
                else:
                    raise ValueError(guess_method)
                target_sense = sentence[target_index]
                # Determining whether guesses were correct or incorrect
                if target_sense not in accuracy_dict:
                    accuracy_dict[target_sense] = []
                temp_sense_accuracies = []
                for guess in guesses:
                    if target_sense == guess:
                        temp_sense_accuracies.append(True)
                    else:
                        temp_sense_accuracies.append(False)
                if index_info:
                    accuracy_dict[target_sense].append(
                        [tuple([sentence_index, target_index, target_sense]), temp_sense_accuracies])
                else:
                    accuracy_dict[target_sense].append(temp_sense_accuracies)
        # Saving results to file
        accuracy_list = []
        for word in accuracy_dict.keys():
            accuracy_list.append([word, accuracy_dict[word]])
        if iterations > 1:
            iter_path = path
            iter_path.replace("iter", "iter" + str(iter))
            file = open(iter_path, 'w')
        else:
            file = open(path, 'w')
        json.dump(accuracy_list, file)
        file.close()
        accuracy_dicts.append(accuracy_dict)
    return accuracy_dicts


def dummy_predict_word_sense(sentence_list):
    """
    Dummy function to predict the word sense of all words in a sentence.
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


def guess_word_sense_frequency(target_index, sentence, word_sense_dict, sense_frequencies):
    """
    Guesses the sense of a target word based on the most frequent sense of that word in the corpus.
    Parameters:
        target_index (int): The index of the "target" word in the sentence given in the sentence parameter list.
        sentence (list): A list of lemma/synset tuples referring to all words in the sentence (including the target sense)
        word_sense_dict (dict): dictionary with the possible senses of each word in the corpus.
        sense_frequencies (dict): dictionary with keys a word sense and values the occurrences of that sense in the corpus
    Returns:
        tuple: The sense (synset/lemma tuple) guess of the target word in the sentence.
    """
    target_sense = sentence[target_index]
    target_word = target_sense[0]
    max_score = -float("inf")
    max_sense = None
    for target_sense_candidate in word_sense_dict[target_word]:
        if sense_frequencies[target_sense_candidate] > max_score:
            max_score = sense_frequencies[target_sense_candidate]
            max_sense = [target_sense_candidate]
        if sense_frequencies[target_sense_candidate] == max_score:
            max_sense.append(target_sense_candidate)
    return max_sense


def get_uniform_random_accuracy():
    """
    Returns a lower bound on the WSD by calculating the average uniform probability of selecting the correct sense for
    every word in the corpus.
    """
    corpus_utilities = CorpusUtilities()
    num_words = 0
    uniform_likelihood_sum = 0
    sent_list = corpus_utilities.get_sentence_list()
    ws_dict = corpus_utilities.get_word_sense_dict()
    sent_counter = 0
    accuracies = []
    for sent in sent_list:
        for word in sent:
            uniform_likelihood_sum += 1 / len(ws_dict[word[0]])
            num_words += 1
        sent_counter += 1
        if sent_counter % 5000 == 0:
            accuracies.append(uniform_likelihood_sum / num_words)
            uniform_likelihood_sum = 0
            num_words = 0
    return accuracies


def get_accuracy_from_file(input_file):
    """
    Calculates the accuracy range for a given output from get_corpus_accuracy stored in a json file.
    Parameters:
        input_file (string) file path
    Returns:
        (list) list with the 0th element the lower bound of the accuracy range and the 1st element the upper bound
        of the accuracy range
    """
    guess_list = json.load(open(input_file))
    guess_dict = defaultdict(list)
    for word_result in guess_list:
        guess_dict[tuple(word_result[0])] = word_result[1]
    accuracy_range = []
    total = 0
    upper_accuracies = 0
    lower_accuracies = 0
    for word in list(guess_dict.keys()):
        guesses = guess_dict[word]
        for guess_list in guesses:
            if len(set(guess_list)) == 2:
                upper_accuracies += 1
            elif len(set(guess_list)) == 1 and guess_list[0]:
                upper_accuracies += 1
                lower_accuracies += 1
            total += 1
    accuracy_range.append([lower_accuracies / total, upper_accuracies / total])
    return accuracy_range


def get_word_activations(word, time, word_sense_dict, network):
    """
    Debugging function, prints sense of a word and current activation for all senses of a given word
    Parameters:
        word (string): The word to guess the sense of (no sense known).
        time (int):"current" time for calculating activations
        word_sense_dict (dict): Dictionary with senseless words as keys and all senses of that word as values.
        network (sentenceLTM): Semantic network with all information there for getting the activations.
    """
    senses = word_sense_dict[word]
    print("Time =", time)
    for sense in senses:
        print("Sense=", sense, ", Activation =", network.activation.get_activation(sense, time))

# Testing --------------------------------------------------------------------------------------------------------------
# import time

for part in range(1, 7):
    for clear in ["never", "word", "sentence"]:
        for context in ["sense", "word"]:
            for cooc in [True, False]:
                print("new", part, clear, context, cooc,
                      run_wsd("cooc_thresh_sem", iterations=1, num_sentences=5000, clear_network=clear,
                                       partition=part, context_type=context, outside_corpus=False, whole_corpus=cooc), flush=True)

