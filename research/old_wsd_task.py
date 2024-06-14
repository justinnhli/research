import random
import datetime
from sentence_long_term_memory import sentenceLTM
from sentence_cooccurrence_activation import SentenceCooccurrenceActivation
from wsd_nltk_importer import *
import wsd_task


def run_wsd(guess_method, activation_base=2, decay_parameter=0.05, constant_offset=0, iterations=1, num_sentences=-1,
            partition=1, outside_corpus=True, clear_network="never", context="word"):
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
    Returns:
        (float): The raw percent accuracy of the guesses of context_sense_predict_word_sense.
    """
    sentence_list, word_sense_dict = extract_sentences(num_sentences=num_sentences, partition=partition)
    if guess_method == "cooc" and context == "word":
        # The guess_dict is a dictionary with keys the sense of each word in the corpus and values a list of boolean
        # values indicating whether the sense was guessed correctly each time it appears in the corpus.
        guess_dicts = get_corpus_accuracy(guess_method="cooc",
                                          context="word",
                                          sentence_list=sentence_list,
                                          word_sense_dict=word_sense_dict,
                                          iterations=iterations,
                                          partition=partition)
    elif guess_method == "cooc" and context == "sense":
        guess_dicts = get_corpus_accuracy(guess_method="cooc",
                                          context="sense",
                                          sentence_list=sentence_list,
                                          word_sense_dict=word_sense_dict,
                                          iterations=iterations,
                                          partition=partition)
    elif guess_method == "frequency":
        guess_dicts = get_corpus_accuracy(guess_method="frequency",
                                          sentence_list=sentence_list,
                                          word_sense_dict=word_sense_dict,
                                          iterations=iterations,
                                          partition=partition)
    elif guess_method == "naive_semantic":
        guess_dicts = get_corpus_accuracy("naive_semantic",
                                          sentence_list=sentence_list,
                                          word_sense_dict=word_sense_dict,
                                          clear_network=clear_network,
                                          activation_base=activation_base,
                                          decay_parameter=decay_parameter,
                                          constant_offset=constant_offset,
                                          iterations=iterations,
                                          partition=partition,
                                          outside_corpus=outside_corpus)
    elif guess_method == "naive_semantic_spreading":
        guess_dicts = get_corpus_accuracy("naive_semantic_spreading",
                                          sentence_list=sentence_list,
                                          word_sense_dict=word_sense_dict,
                                          clear_network=clear_network,
                                          activation_base=activation_base,
                                          decay_parameter=decay_parameter,
                                          constant_offset=constant_offset,
                                          iterations=iterations,
                                          partition=partition,
                                          outside_corpus=outside_corpus)
    elif guess_method == "oracle":
        guess_dicts = get_corpus_accuracy("oracle",
                                          context=context,
                                          sentence_list=sentence_list,
                                          word_sense_dict=word_sense_dict,
                                          clear_network=clear_network,
                                          activation_base=activation_base,
                                          decay_parameter=decay_parameter,
                                          constant_offset=constant_offset,
                                          iterations=iterations,
                                          partition=partition,
                                          outside_corpus=outside_corpus)
    elif guess_method == "sem_thresh_cooc":
        guess_dicts = get_corpus_accuracy("sem_thresh_cooc",
                                          sentence_list=sentence_list,
                                          word_sense_dict=word_sense_dict,
                                          iterations=iterations,
                                          partition=partition,
                                          context=context)
    elif guess_method == "cooc_thresh_sem":
        guess_dicts = get_corpus_accuracy("cooc_thresh_sem",
                                          sentence_list=sentence_list,
                                          word_sense_dict=word_sense_dict,
                                          clear_network=clear_network,
                                          activation_base=activation_base,
                                          decay_parameter=decay_parameter,
                                          constant_offset=constant_offset,
                                          iterations=iterations,
                                          partition=partition,
                                          outside_corpus=outside_corpus,
                                          context=context)
    else:
        raise ValueError(guess_method)
    accuracies = []
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


def adjust_sem_rel_dict(sem_rel_dict, sent_list, context_type, whole_corpus=True):
    """ This is for cooccurrence thresholded spreading """
    if whole_corpus:
        all_sents_list, word_sense_dict = extract_sentences()
        cooc_rel_dict = create_cooc_relations_dict(all_sents_list, context_type)
    else:
        cooc_rel_dict = create_cooc_relations_dict(sent_list, context_type)
    for word_key in sem_rel_dict.keys():
        word_rel_dict = sem_rel_dict[word_key]  # has all different relations to target word
        for cat in word_rel_dict.keys():  # looping through each relation category
            rels = word_rel_dict[cat]  # getting the relations in that category
            new_rels = []
            for rel in rels:  # going through words corresponding to each relation
                if context_type == "sense":
                    if rel in list(cooc_rel_dict[word_key]):
                        new_rels.append(rel)
                else:
                    if rel[0] in list(cooc_rel_dict[word_key]):
                        new_rels.append(rel)
            sem_rel_dict[word_key][cat] = new_rels
    return sem_rel_dict



def create_sem_network(sentence_list, spreading=True, outside_corpus=False, activation_base=2, decay_parameter=0.05,
                        constant_offset=0, partition=1, cooc_thresh=False, context="word"):
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
    if not spreading:
        spread_depth = 0
    semantic_relations_dict = get_semantic_relations_dict(sentence_list, partition=partition, outside_corpus=outside_corpus)
    if cooc_thresh:
        semantic_relations_dict = adjust_sem_rel_dict(semantic_relations_dict, sentence_list, context, whole_corpus=True)
    network = sentenceLTM(
        activation_cls=(lambda ltm:
                        SentenceCooccurrenceActivation(
                            ltm,
                            activation_base=activation_base,
                            constant_offset=constant_offset,
                            decay_parameter=decay_parameter
                        )))
    relations_keys = sorted(list(set(semantic_relations_dict.keys())))
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



def create_cooc_relations_dict(sentence_list, context="word"):
    """
    Creates a dictionary where each word in the corpus is a key and each of the other words it cooccurs with (are in
    the same sentence as our target word) are values (in a set).
    """
    cooc_rel_dict = defaultdict(set)
    for sent in sentence_list:
        for index in range(len(sent)):
            for context_index in range(len(sent)):
                if index != context_index:
                    target_sense = sent[index]
                    context_sense = sent[context_index]
                    if context == "sense":
                        cooc_rel_dict[target_sense].update([context_sense])
                    else:
                        context_word = context_sense[0]
                        cooc_rel_dict[target_sense].update([context_word])
    return cooc_rel_dict


def precompute_word_sense(sentence_list):
    """
    Calculates the number of times each word (across all senses of the word) shows up in the corpus and the number of times
        each sense of each word shows up in the corpus (or a subset of the corpus).
    Parameters:
        sentence_list (list): List of all sentences in the corpus or the first n sentences of the corpus (from extract_sentences function).
    Returns:
        list: word_counts (int default dict with each key the word and values the frequency of the word)
            sense_counts (int default dict with each key the sense and values the frequency of the sense)
    """
    word_counts = defaultdict(int)
    sense_counts = defaultdict(int)
    for sentence in sentence_list:
        for sense in sentence:
            word_counts[sense[0]] += 1
            sense_counts[sense] += 1
    return word_counts, sense_counts


def precompute_cooccurrences(sentence_list):
    """
    Precomputes word-word, sense-word, sense-sense cooccurrences and the number of times each sense shows up in the
    corpus (or a subset of the corpus).
    Parameters:
        sentence_list (list): List of all sentences in the corpus or the first n sentences of the corpus (from extract_sentences function).
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
            target_word = target_sense[0]
            sense_frequencies[target_sense] += 1
            for other_index in range(len(sentence)):
                if target_index != other_index:
                    other_sense = sentence[other_index]
                    other_word = other_sense[0]
                    word_word_cooccurrences[(target_word, other_word)] += 1
                    sense_word_cooccurrences[(target_sense, other_word)] += 1
                    sense_sense_cooccurrences[(target_sense, other_sense)] += 1
    return word_word_cooccurrences, sense_word_cooccurrences, sense_sense_cooccurrences, sense_frequencies


def clear_sem_network(sem_network, start_time):
    """
    Clears the semantic network by resetting activations to a certain "starting time".
    Parameters:
        sem_network (sentenceLTM): Network to clear.
        start_time (int): The network will be reset so that activations only at the starting time and before the
            starting time remain.
    Returns:
        sentenceLTM: Cleared semantic network.
    """
    activations = sem_network.activation.activations
    activated_words = activations.keys()
    for word in activated_words:
        old_acts = activations[word]
        new_acts = [act for act in activations[word] if act[0] <= start_time]
        activations[word] = new_acts
    return sem_network


def get_accuracy_from_file(input_file):
    """
    Calculates the accuracy range for a given output from get_corpus_accuracy stored in a json file.
    Parameters:
        input_file (string) file path
    Returns: (list) list with the 0th element the lower bound of the accuracy range and the 1st element the upper bound
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
    senses = word_sense_dict[word]
    print("Time =", time)
    for sense in senses:
        print("Sense=", sense, ", Activation =", network.activation.get_activation(sense, time))


def get_corpus_accuracy(guess_method, sentence_list, word_sense_dict, input_sem_network=None,
                        input_timer=None, clear_network="never", activation_base=2, decay_parameter=0.05,
                        constant_offset=0, iterations=1, partition=1, outside_corpus=False, context="word",
                        index_info=False):
    """
    Guesses the word sense for every word in the corpus based on a specified guess method.
    Parameters:
        guess_method (string): Which method to use when guessing the sense of each word. Possibilities are "cooc",
            "naive_semantic_spreading", "naive_semantic", "oracle", and "frequency". Integrated mechanisms include
            "sem_thresh_cooc", "cooc_thresh_sem".
        sentence_list (list): A nested list of all sentences in the corpus with each word referenced by a lemma/synset
            tuple.
        word_sense_dict (dict): dictionary with the possible senses of each word in the corpus
        input_sem_network (sentenceLTM): Optional network to input (so the network doesn't have to be reinitialized). If
            None, there is no input network.
        input_timer (int): Optional current time based on optional input_sem_network. If None, default starting time is 1.
        clear_network (string): How often to clear the network. Possible values are "never", "sentence", or "word",
            indicating that the network is never cleared, cleared after each sentence, or cleared after each word.
        activation_base (float): A parameter in the activation equation.
        decay_parameter (float): A parameter in the activation equation.
        constant_offset (float): A parameter in the activation equation.
        iterations (int): The number of times to run through the corpus (for semantic effects only).
        partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking at
            sentences 10000 - 14999.
        outside_corpus (bool): True if semantic relations can be considered outside the corpus and False if semantic
            relations are only considered from words inside the corpus.
        index_info (bool): True indicates that the first entry in each dictionary value will indicate where in the
         corpus the word is from in an "index tuple": (sentence number out of corpus, word number out of sentence)
    Returns:
        list: A list of how accurate guesses were based on each sense-specific word in the corpus, and the overall accuracy
            of the guessing method.
    """
    curr_time = str(datetime.datetime.now())
    if iterations == 1:
        path = "./results/" + guess_method + "_"
    else:
        path = "./results/" + guess_method + "_iter_"
    if "semantic" in guess_method:
        path += str(len(sentence_list)) + "_sents_" + clear_network + "_clear_" + str(outside_corpus) + \
                "_outside_corpus_" + str(partition) + "_partition"
    else:
        path += str(len(sentence_list)) + "_sents_" + str(partition) + "_partition"
    if index_info:
        path += "_index"
    path += "_accuracy_list_" + curr_time + ".json"
    word_word_cooccurrences, sense_word_cooccurrences, sense_sense_cooccurrences, sense_frequencies = precompute_cooccurrences(
        sentence_list)
    if guess_method == "naive_semantic" and input_sem_network is None and input_timer is None:
        # Same network for spreading and no spreading
        sem_network = create_sem_network(sentence_list, spreading=False, activation_base=activation_base,
                                         decay_parameter=decay_parameter, constant_offset=constant_offset,
                                         partition=partition, outside_corpus=outside_corpus, cooc_thresh=False)
        timer = 2
    elif guess_method == "naive_semantic_spreading" and input_sem_network is None and input_timer is None:
        sem_network = create_sem_network(sentence_list, spreading=True, activation_base=activation_base,
                                         decay_parameter=decay_parameter, constant_offset=constant_offset,
                                         partition=partition, outside_corpus=outside_corpus, cooc_thresh=False)
        timer = 2
    elif guess_method == "oracle":
        sem_network_never = create_sem_network(sentence_list, spreading=True, activation_base=activation_base,
                                               decay_parameter=decay_parameter, constant_offset=constant_offset,
                                               partition=partition, outside_corpus=outside_corpus)
        sem_network_word = create_sem_network(sentence_list, spreading=True, activation_base=activation_base,
                                              decay_parameter=decay_parameter, constant_offset=constant_offset,
                                              partition=partition, outside_corpus=outside_corpus)
        sem_network_sentence = create_sem_network(sentence_list, spreading=True, activation_base=activation_base,
                                                  decay_parameter=decay_parameter, constant_offset=constant_offset,
                                                  partition=partition, outside_corpus=outside_corpus)
        sem_network_no_spread = create_sem_network(sentence_list, spreading=False, activation_base=activation_base,
                                                   decay_parameter=decay_parameter, constant_offset=constant_offset,
                                                   partition=partition, outside_corpus=outside_corpus)
        timer_never = 2
        timer_word = 2
        timer_sentence = 2
    elif guess_method == "cooc_thresh_sem":
        sem_network = create_sem_network(sentence_list, spreading=True, activation_base=activation_base,
                                         decay_parameter=decay_parameter, constant_offset=constant_offset,
                                         partition=partition, outside_corpus=outside_corpus, cooc_thresh=True,
                                         context=context)
        timer = 2
    elif input_sem_network is not None and input_timer is not None and (
            guess_method == "naive_semantic" or guess_method == "naive_semantic_spreading"):
        sem_network = input_sem_network
    elif guess_method == "naive_semantic" or guess_method == "naive_semantic_spreading" or guess_method == "oracle":
        raise ValueError(input_sem_network, input_timer)
    if clear_network != "never" and clear_network != "sentence" and clear_network != "word":
        raise ValueError(clear_network)
    if context != "word" and context != "sense":
        raise ValueError(context)
    if guess_method == "sem_thresh_cooc":
        sem_rel_dict = get_semantic_relations_dict(sentence_list=sentence_list, partition=partition,
                                                   outside_corpus=False)
    accuracy_dicts = []  # List to store dictionaries of correct guesses for each iteration
    for iter in range(iterations):
        if "naive_semantic" in guess_method and iter > 1:
            # Clear semantic network each iteration, so you start fresh & don't have to remake the network (expensive)
            sem_network = clear_sem_network(sem_network, 1)
            timer = 2  # reset timer
        elif guess_method == "cooc_thresh_sem" and iter > 1:
            sem_network = clear_sem_network(sem_network, 1)
            timer = 2  # reset timer.
        elif guess_method == "oracle" and iter > 1:
            sem_network_never = clear_sem_network(sem_network_never, 1)
            sem_network_no_spread = clear_sem_network(sem_network_no_spread, 1)
            timer_never = 2
        accuracy_dict = defaultdict(list)
        for sentence_index in range(len(sentence_list)):
            sentence = sentence_list[sentence_index]
            if "naive_semantic" in guess_method and clear_network == "sentence":
                sem_network = clear_sem_network(sem_network, 1)
                timer = 2  # reset timer.
            elif guess_method == "cooc_thresh_sem" and clear_network == "sentence":
                sem_network = clear_sem_network(sem_network, 1)
                timer = 2  # reset timer.
            elif guess_method == "oracle" and clear_network == "sentence":
                sem_network_sentence = clear_sem_network(sem_network_sentence, 1)
                timer_sentence = 2
            for target_index in range(len(sentence)):
                if "naive_semantic" in guess_method and clear_network == "word":
                    sem_network = clear_sem_network(sem_network, 1)
                    timer = 2  # reset timer.
                elif guess_method == "cooc_thresh_sem" and clear_network == "word":
                    sem_network = clear_sem_network(sem_network, 1)
                    timer = 2  # reset timer.
                elif guess_method == "oracle" and clear_network == "word":
                    sem_network_word = clear_sem_network(sem_network_word, 1)
                    timer_word = 2
                if guess_method == "cooc" and context == "word":
                    guesses = guess_word_sense_context_word(target_index,
                                                            sentence,
                                                            word_sense_dict,
                                                            sense_word_cooccurrences,
                                                            word_word_cooccurrences)
                elif guess_method == "cooc" and context == "sense":
                    guesses = guess_word_sense_context_sense(target_index,
                                                             sentence,
                                                             word_sense_dict,
                                                             sense_word_cooccurrences,
                                                             sense_sense_cooccurrences)
                elif guess_method == "frequency":
                    guesses = guess_word_sense_frequency(target_index,
                                                         sentence,
                                                         word_sense_dict,
                                                         sense_frequencies)
                elif guess_method == "naive_semantic":
                    guesses = guess_word_sense_semantic(target_index,
                                                        sentence,
                                                        word_sense_dict,
                                                        sem_network,
                                                        timer,
                                                        spread_depth=0)
                    timer += 1
                elif guess_method == "naive_semantic_spreading":
                    guesses = guess_word_sense_semantic(target_index,
                                                        sentence,
                                                        word_sense_dict,
                                                        sem_network,
                                                        timer,
                                                        spread_depth=-1)
                    timer += 1
                elif guess_method == "oracle":
                    guesses = guess_word_sense_oracle(target_index,
                                                      sentence,
                                                      word_sense_dict,
                                                      sense_sense_cooccurrences,
                                                      sense_word_cooccurrences,
                                                      word_word_cooccurrences,
                                                      sem_network_no_spread,
                                                      sem_network_never,
                                                      sem_network_word,
                                                      sem_network_sentence,
                                                      timer_word,
                                                      timer_sentence,
                                                      timer_never)
                    timer_word += 1
                    timer_sentence += 1
                    timer_never += 1
                elif guess_method == "sem_thresh_cooc":
                    guesses = guess_word_sense_spreading_thresholded_cooccurrence(target_index,
                                                                                  sentence,
                                                                                  word_sense_dict,
                                                                                  sem_rel_dict,
                                                                                  sense_sense_cooccurrences,
                                                                                  sense_word_cooccurrences,
                                                                                  word_word_cooccurrences,
                                                                                  context=context)
                elif guess_method == "cooc_thresh_sem":
                    guesses = guess_word_sense_semantic(target_index,
                                                        sentence,
                                                        word_sense_dict,
                                                        sem_network,
                                                        timer,
                                                        spread_depth=-1)
                    timer += 1
                else:
                    raise ValueError(guess_method)
                target_sense = sentence[target_index]
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


def guess_word_sense_context_word(target_index, sentence, word_sense_dict, sense_word_cooccurrences,
                                  word_word_cooccurrences):
    """
    Guesses the sense of a "target" word based on how often it cooccurs with other words in the sentence
    Parameters:
        target_index (int): The index of the "target" word in the sentence given in the sentence parameter list.
        sentence (list): A list of lemma/synset tuples referring to all words in the sentence (including the target sense)
        word_sense_dict (dict): dictionary with the possible senses of each word in the corpus
        sense_word_cooccurrences (dict): a dictionary with keys as sense/word pairs and values the cooccurrences associated
        word_word_cooccurrences (dict): a dictionary with keys as word/word pairs and values the cooccurrences associated
    Returns:
        tuple: The sense (synset/lemma tuple) guess of the target word in the sentence.
    """
    max_score = -float("inf")
    max_sense = None
    target_sense = sentence[target_index]
    target_word = target_sense[0]
    for target_sense_candidate in word_sense_dict[target_sense[0]]:
        aggregate = 0
        for other_index in range(len(sentence)):
            if other_index != target_index:
                other_word = sentence[other_index][0]
                if (target_sense_candidate, other_word) not in sense_word_cooccurrences:
                    numerator = 0
                else:
                    numerator = sense_word_cooccurrences[(target_sense_candidate, other_word)]
                denominator = word_word_cooccurrences[(target_word, other_word)]
                aggregate += numerator / denominator
        if aggregate > max_score:
            max_score = aggregate
            max_sense = [target_sense_candidate]
        elif aggregate == max_score:
            max_sense.append(target_sense_candidate)
    return max_sense


def guess_word_sense_context_sense(target_index, sentence, word_sense_dict, sense_word_cooccurrences,
                                   sense_sense_cooccurrences):
    """
        Guesses the sense of a "target" word based on how often it cooccurs with other sense-specific words in the sentence
        Parameters:
            target_index (int): The index of the "target" word in the sentence given in the sentence parameter list.
            sentence (list): A list of lemma/synset tuples referring to all words in the sentence (including the target sense)
            word_sense_dict (dict): dictionary with the possible senses of each word in the corpus
          . sense_word_cooccurrences (dict): a dictionary with keys as sense/word pairs and values the cooccurrences associated
            sense_sense_cooccurrences (dict): a dictionary with keys as sense/sense pairs and values the cooccurrences associated
        Returns:
            tuple: The sense (synset/lemma tuple) guess of the target word in the sentence.
        """
    max_score = -float("inf")
    max_sense = None
    target_sense = sentence[target_index]
    target_word = target_sense[0]
    for target_sense_candidate in word_sense_dict[target_sense[0]]:
        aggregate = 0
        for other_index in range(len(sentence)):
            if other_index != target_index:
                other_sense = sentence[other_index]
                if (target_sense_candidate, other_sense) not in sense_sense_cooccurrences:
                    numerator = 0
                else:
                    numerator = sense_sense_cooccurrences[(target_sense_candidate, other_sense)]
                denominator = sense_word_cooccurrences[(other_sense, target_word)]
                aggregate += numerator / denominator
        if aggregate > max_score:
            max_score = aggregate
            max_sense = [target_sense_candidate]
        elif aggregate == max_score:
            max_sense.append(target_sense_candidate)
    return max_sense


def guess_word_sense_semantic(target_index, sentence, word_sense_dict, sem_network, time, spread_depth=-1):
    """
    Guesses the sense of the target word based on the sense with the highest activation (based on semantic connections)
    out of the possible senses of the target word. Note: this function does not increment the time, it must be done
    outside the function.
    Parameters:
        target_index (int): The index of the "target" word in the sentence given in the sentence parameter list.
        sentence (list): A list of lemma/synset tuples referring to all words in the sentence (including the target sense)
        word_sense_dict (dict): dictionary with the possible senses of each word in the corpus.
        sem_network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
        time (int): The current time. The activations of each possible sense are calculated at this time, and the guess
            and correct word sense are activated at this time.
        spread_depth (int): How many nodes deep in the semantic network to spread to after an element has been
            activated. If spread_depth = -1, full semantic spreading is implemented, if spread_depth = 0, no semantic
            spreading is implemented.
    Returns:
        tuple: The sense (synset/lemma tuple) guess of the target word in the sentence.
    """
    word = sentence[target_index]
    senses = word_sense_dict[word[0]]
    sense_guess = []
    max_act = float('-inf')
    for sense in senses:
        sense_act = sem_network.get_activation(mem_id=sense, time=time)
        if sense_act > max_act:
            max_act = sense_act
            sense_guess = [sense]
        elif sense_act == max_act:
            sense_guess.append(sense)
    sem_network.store(mem_id=word, time=time, spread_depth=spread_depth)
    for sense in sense_guess:
        sem_network.store(mem_id=sense, time=time, spread_depth=spread_depth)
    return sense_guess


def guess_word_sense_spreading_thresholded_cooccurrence(target_index, sentence, word_sense_dict, sem_rel_dict,
                                                        sense_sense_cooccurrences, sense_word_cooccurrences,
                                                        word_word_cooccurrences, context="word"):
    max_score = -float("inf")
    max_sense = None
    target_sense = sentence[target_index]
    target_word = target_sense[0]
    for target_sense_candidate in list(word_sense_dict[target_sense[0]]):
        aggregate = 0
        for other_index in range(len(sentence)):
            if other_index == target_index:
                continue
            other_sense = sentence[other_index]
            organized_target_rels = sem_rel_dict[target_sense_candidate]
            target_rels = sum(list(organized_target_rels.values()), [])
            if other_sense not in target_rels:
                continue
            if context == "sense":
                if (target_sense_candidate, other_sense) not in sense_sense_cooccurrences:
                    numerator = 0
                else:
                    numerator = sense_sense_cooccurrences[(target_sense_candidate, other_sense)]
                denominator = sense_word_cooccurrences[(other_sense, target_word)]
            else:  # context == "word"
                other_word = other_sense[0]
                if (target_sense_candidate, other_word) not in sense_word_cooccurrences:
                    numerator = 0
                else:
                    numerator = sense_word_cooccurrences[(target_sense_candidate, other_word)]
                denominator = word_word_cooccurrences[(other_word, target_word)]
            if denominator != 0:
                aggregate += numerator / denominator
        if aggregate > max_score:
            max_score = aggregate
            max_sense = [target_sense_candidate]
        elif aggregate == max_score:
            max_sense.append(target_sense_candidate)
    if max_score == 0 or max_sense == -float("inf"):
        return []
    return max_sense


def guess_word_sense_oracle(target_index, sentence, word_sense_dict, sense_sense_cooccurrences,
                            sense_word_cooccurrences, word_word_cooccurrences, sem_network_no_spread, sem_network_never,
                            sem_network_word, sem_network_sentence, timer_word, timer_sentence, timer_never):
    """
    Upper bound on WSD that assumes knowledge of the correct answer and tests if each cooccurrence and will answer
    correct if at least one cooccurrence or spreading mechanism gets it right.
    """
    correct_sense = sentence[target_index]
    cooc_word_guess = guess_word_sense_context_word(target_index, sentence, word_sense_dict, sense_word_cooccurrences,
                                                    word_word_cooccurrences)
    if correct_sense in cooc_word_guess:
        return [correct_sense]
    cooc_sense_guess = guess_word_sense_context_sense(target_index, sentence, word_sense_dict, sense_word_cooccurrences,
                                                      sense_sense_cooccurrences)
    if correct_sense in cooc_sense_guess:
        return [correct_sense]
    sem_guess_never = guess_word_sense_semantic(target_index, sentence, word_sense_dict, sem_network_never, timer_never,
                                                spread_depth=-1)
    if correct_sense in sem_guess_never:
        return [correct_sense]
    sem_guess_no_spread = guess_word_sense_semantic(target_index, sentence, word_sense_dict, sem_network_no_spread,
                                                    timer_never,
                                                    spread_depth=0)
    if correct_sense in sem_guess_no_spread:
        return [correct_sense]
    sem_guess_word = guess_word_sense_semantic(target_index, sentence, word_sense_dict, sem_network_word, timer_word,
                                               spread_depth=-1)
    if correct_sense in sem_guess_word:
        return [correct_sense]
    sem_guess_sentence = guess_word_sense_semantic(target_index, sentence, word_sense_dict, sem_network_sentence,
                                                   timer_sentence,
                                                   spread_depth=-1)
    if correct_sense in sem_guess_sentence:
        return [correct_sense]
    else:
        return [None]


def get_uniform_random_accuracy():
    """
    Returns a lower bound on the WSD by calculating the average uniform probability of selecting the correct sense for
    every word in the corpus.
    """
    num_words = 0
    uniform_likelihood_sum = 0
    sent_list, ws_dict = extract_sentences(-1)
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


# Testing --------------------------------------------------------------------------------------------------------------
#
# print("old, thresh, sense context")
# print(run_wsd(guess_method="cooc_thresh_sem", num_sentences=500, outside_corpus=False, context="sense",
#               clear_network="never"))
# print(run_wsd(guess_method="cooc_thresh_sem", num_sentences=500, outside_corpus=False, context="sense",
#               clear_network="sentence"))
# print(run_wsd(guess_method="cooc_thresh_sem", num_sentences=500, outside_corpus=False, context="sense",
#               clear_network="word"))
# print("new, thresh, sense_context ")
# print(wsd_task.run_wsd(guess_method="cooc_thresh_sem", num_sentences=500, outside_corpus=False, context_type="sense",
#                        clear_network="never", whole_corpus=True))
# print(wsd_task.run_wsd(guess_method="cooc_thresh_sem", num_sentences=500, outside_corpus=False, context_type="sense",
#                        clear_network="sentence", whole_corpus=True))
# print(wsd_task.run_wsd(guess_method="cooc_thresh_sem", num_sentences=500, outside_corpus=False, context_type="sense",
#                        clear_network="word", whole_corpus=True))
# print("old, thresh, word context")
# print(run_wsd(guess_method="cooc_thresh_sem", num_sentences=500, outside_corpus=False, context="word",
#               clear_network="never"))
# print(run_wsd(guess_method="cooc_thresh_sem", num_sentences=500, outside_corpus=False, context="word",
#               clear_network="sentence"))
# print(run_wsd(guess_method="cooc_thresh_sem", num_sentences=500, outside_corpus=False, context="word",
#               clear_network="word"))
# print("new, thresh, word context ")
# print(wsd_task.run_wsd(guess_method="cooc_thresh_sem", num_sentences=500, outside_corpus=False, context_type="word",
#                        clear_network="never", whole_corpus=True))
# print(wsd_task.run_wsd(guess_method="cooc_thresh_sem", num_sentences=500, outside_corpus=False, context_type="word",
#                        clear_network="sentence", whole_corpus=True))
# print(wsd_task.run_wsd(guess_method="cooc_thresh_sem", num_sentences=500, outside_corpus=False, context_type="word",
#                        clear_network="word", whole_corpus=True))
# print("old, sem")
# print(run_wsd(guess_method="naive_semantic_spreading", num_sentences=500, outside_corpus=False, context="sense",
#               clear_network="never"))
# print(run_wsd(guess_method="naive_semantic_spreading", num_sentences=500, outside_corpus=False, context="sense",
#               clear_network="sentence"))
# print(run_wsd(guess_method="naive_semantic_spreading", num_sentences=500, outside_corpus=False, context="sense",
#               clear_network="word"))
# print("new, sem")
# print(wsd_task.run_wsd(guess_method="naive_semantic_spreading", num_sentences=500, outside_corpus=False, context_type="sense",
#                        clear_network="never", whole_corpus=True))
# print(wsd_task.run_wsd(guess_method="naive_semantic_spreading", num_sentences=500, outside_corpus=False, context_type="sense",
#                        clear_network="sentence", whole_corpus=True))
# print(wsd_task.run_wsd(guess_method="naive_semantic_spreading", num_sentences=500, outside_corpus=False, context_type="sense",
#                        clear_network="word", whole_corpus=True))
#
# print("new, oracle")
# print(wsd_task.run_wsd(guess_method="oracle", num_sentences=500, outside_corpus=False))
# checking corpus accuracy
# print("old")
# import time
# start = time.time()
# sentence_list, word_sense_dict = extract_sentences(5000, 1)
# old = get_corpus_accuracy(guess_method="cooc_thresh_sem", sentence_list=sentence_list,
#                           word_sense_dict=word_sense_dict,  outside_corpus=False, context="sense",
#                           clear_network="never", index_info=True)[0]
# end = time.time()
# print("old run time", end - start)
# print(run_wsd(guess_method="cooc_thresh_sem", num_sentences=5000, outside_corpus=False, context="sense",
#                clear_network="sentence"))
# print("new")
# start = time.time()
# corpus_utils = wsd_task.CorpusUtilities(5000, 1)
# new = wsd_task.get_corpus_accuracy(guess_method="cooc_thresh_sem", corpus_utilities=corpus_utils, context_type="sense",
#                                     outside_corpus=False, clear_network="never", index_info=True, whole_corpus=True)[0]
# end = time.time()
# print("new run time", end - start)
# print(wsd_task.run_wsd(guess_method="naive_semantic_spreading", num_sentences=5000, outside_corpus=False, context_type="sense",
                       # clear_network="sentence", whole_corpus=True))
# for key in new.keys():
#     if new[key][0][1].count(True) != old[key][0][1].count(True):
#         print("new", new[key])
#         print("old", old[key])
#     elif new[key][0][1].count(False) != old[key][0][1].count(False):
#         print("new", new[key])
#         print("old", old[key])
#     elif len(new[key][0][1]) != len(old[key][0][1]):
#         print("new", new[key])
#         print("old", old[key])






