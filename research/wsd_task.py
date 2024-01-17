import random
from collections import defaultdict
import nltk
from sentence_long_term_memory import sentenceLTM
from sentence_long_term_memory import SentenceCooccurrenceActivation
from nltk.corpus import semcor
import pandas as pd


def run_wsd(guess_method, activation_base=2, decay_parameter=0.05, constant_offset=0, iterations=1, num_sentences=-1,
            clear_network=True):
    """
    Runs the word sense disambiguation task over the Semcor corpus (or a subset of it).
    Parameters:
        guess_method (string): The function used to guess the sense of each word in each sentence. Possibilities are:
            "context_word", "context_sense", "frequency", "naive_semantic", "naive_semantic_spreading".
        activation_base (float): A parameter in the activation equation.
        decay_parameter (float): A parameter in the activation equation.
        constant_offset (float): A parameter in the activation equation.
        iterations (int): The number of times to run through the corpus (for semantic effects only).
        num_sentences (int): The number of sentences from the corpus to use in the task. The first n sentences
            from the corpus are used and if n=-1, all sentences from the corpus are used.
        clear_network (string): How often to clear the network. Possible values are "never", "sentence", or "word",
            indicating that the network is never cleared, cleared after each sentence, or cleared after each word.
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
    elif guess_method == "naive_semantic":
        guess_dict = get_corpus_accuracy("naive_semantic",
                                         sentence_list=sentence_list,
                                         word_sense_dict=word_sense_dict,
                                         clear_network=clear_network,
                                         activation_base=activation_base,
                                         decay_parameter=decay_parameter,
                                         constant_offset=constant_offset,
                                         iterations=iterations)
    elif guess_method == "naive_semantic_spreading":
        guess_dict = get_corpus_accuracy("naive_semantic_spreading",
                                         sentence_list=sentence_list,
                                         word_sense_dict=word_sense_dict,
                                         clear_network=clear_network,
                                         activation_base=activation_base,
                                         decay_parameter=decay_parameter,
                                         constant_offset=constant_offset,
                                         iterations=iterations)
    else:
        raise ValueError(guess_method)
    raw_truths = sum(guess_dict.values(), [])
    accuracy = raw_truths.count(True) / len(raw_truths)
    return accuracy


def extract_sentences(num_sentences=-1):
    """
    Runs the word sense disambiguation task.
    Parameters:
        num_sentences (int): The number of sentences from the corpus to use in the task. The first n sentences
            from the corpus are used and if n=-1, all sentences from the corpus are used.
    Returns:
        list: sentence_list (list of all sentences or the first n sentences of the corpus), word_sense_dict (dictionary with the possible senses of
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


def create_sem_network(sentence_list, spreading=True, activation_base=2, decay_parameter=0.05, constant_offset=0):
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
    semcor_words = sum(sentence_list, [])
    spread_depth = -1
    if not spreading:
        spread_depth = 0
    for sentence in sentence_list:
        for word in sentence:
            syn = word[1]
            lemma = word[0]
            synonyms = [(synon, syn) for synon in syn.lemmas() if (synon, syn) in semcor_words and synon != lemma]
            # These are all synsets.
            synset_relations = [syn.hypernyms(), syn.hyponyms(),
                                syn.member_holonyms() + syn.substance_holonyms() + syn.part_holonyms(),
                                syn.member_meronyms() + syn.substance_meronyms() + syn.part_meronyms(),
                                syn.attributes(), syn.entailments(), syn.causes(), syn.also_sees(),
                                syn.verb_groups(),
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
            network.store(mem_id=word,
                          time=1,
                          spread_depth=spread_depth,
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
    return network


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
            word_counts[sense[0].name()] += 1
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
                aggregate += numerator / denominator
        if aggregate > max_score:
            max_score = aggregate
            max_sense = target_sense_candidate
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
                aggregate += numerator / denominator
        if aggregate > max_score:
            max_score = aggregate
            max_sense = target_sense_candidate
    return max_sense


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
    target_word = target_sense[0].name()
    max_score = -float("inf")
    max_sense = None
    for target_sense_candidate in word_sense_dict[target_word]:
        if sense_frequencies[target_sense_candidate] > max_score:
            max_score = sense_frequencies[target_sense_candidate]
            max_sense = target_sense_candidate
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
    senses = word_sense_dict[word[0].name()]
    max_act = float('-inf')
    for sense in senses:
        sense_act = sem_network.get_activation(mem_id=sense, time=time)
        if sense_act > max_act:
            max_act = sense_act
            sense_guess = sense
    sem_network.store(mem_id=word, time=time, spread_depth=spread_depth)
    sem_network.store(mem_id=sense_guess, time=time, spread_depth=spread_depth)
    return sense_guess


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
        activations[word] = [act for act in activations[word] if act[0] <= start_time]
    return sem_network


def get_corpus_accuracy(guess_method, sentence_list, word_sense_dict, input_sem_network=None, input_timer=None,
                        clear_network="never", activation_base=2, decay_parameter=0.05, constant_offset=0,
                        iterations=1):
    """
    Guesses the word sense for every word in the corpus based on a specified guess method.
    Parameters:
        guess_method (string): Which method to use when guessing the sense of each word. Possibilities are "context_word",
            "context_sense", and "frequency".
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
    Returns:
        list: A list of how accurate guesses were based on each sense-specific word in the corpus, and the overall accuracy
            of the guessing method.
    """
    word_word_cooccurrences, sense_word_cooccurrences, sense_sense_cooccurrences, sense_frequencies = precompute_cooccurrences(
        sentence_list)
    accuracy_dict = defaultdict(list)
    if guess_method == "naive_semantic" and input_sem_network is None and input_timer is None:
        # Same network for spreading and no spreading
        sem_network = create_sem_network(sentence_list, spreading=False, activation_base=activation_base,
                                         decay_parameter=decay_parameter, constant_offset=constant_offset)
    elif guess_method == "naive_semantic_spreading" and input_sem_network is None and input_timer is None:
        sem_network = create_sem_network(sentence_list, spreading=True, activation_base=activation_base,
                                         decay_parameter=decay_parameter, constant_offset=constant_offset)
    elif input_sem_network is not None and input_timer is not None:
        sem_network = input_sem_network
        timer = input_timer
    else:
        raise ValueError(input_sem_network, input_timer)
    if clear_network != "never" and clear_network != "sentence" and clear_network != "word":
        raise ValueError(clear_network)
    timer = 2  # All semantic connections stored at time 1, so start the timer at the next timestep.
    for iter in range(iterations):
        for sentence in sentence_list:
            if "naive_semantic" in guess_method and clear_network == "sentence":
                sem_network = clear_sem_network(sem_network, 1)
                timer = 2  # reset timer.
            for target_index in range(len(sentence)):
                if "naive_semantic" in guess_method and clear_network == "word":
                    sem_network = clear_sem_network(sem_network, 1)
                    timer = 2  # reset timer.
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
                elif guess_method == "naive_semantic":
                    guess = guess_word_sense_semantic(target_index,
                                                      sentence,
                                                      word_sense_dict,
                                                      sem_network,
                                                      timer,
                                                      spread_depth=0)
                    timer += 1
                elif guess_method == "naive_semantic_spreading":
                    guess = guess_word_sense_semantic(target_index,
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
                if target_sense == guess:
                    accuracy_dict[target_sense].append(True)
                else:
                    accuracy_dict[target_sense].append(False)
    return accuracy_dict


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


# Testing --------------------------------------------------------------------------------------------------------------
sentence_list, word_sense_dict = extract_sentences(200)
#clear_network = "word"
#guess_method = "naive_semantic"
print("Semantic: No Spreading, Clear After Word")
#no_spread_dict = get_corpus_accuracy(guess_method, sentence_list, word_sense_dict, clear_network=clear_network)
#no_spread_df = pd.DataFrame(list(no_spread_dict.items()), columns = ["Word", "Guess"])
#print(no_spread_df)

#print()
#print()

#guess_method = "naive_semantic_spreading"
print("Semantic: Spreading, Clear After Word")
#spread_dict = get_corpus_accuracy(guess_method, sentence_list, word_sense_dict, clear_network=clear_network)
#spread_df = pd.DataFrame(list(spread_dict.items()), columns = ["Word", "Guess"])
#print(spread_df)



