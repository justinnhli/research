from wsd_task import extract_sentences, create_sem_network, naive_predict_word_sense, senseless_predict_word_sense,\
    frequency_predict_word_sense
import matplotlib.pyplot as plt
import numpy as np
import math


def get_simple_plot(activation_base, decay_parameter, constant_offset, plot_type="word"):
    """
    Produces simple scatterplot of percentage correct vs. word appearances for each word in the corpus.
    """
    sentence_list, word_sense_dict = extract_sentences()
    sem_network = create_sem_network(sentence_list, activation_base, decay_parameter, constant_offset)
    guess_list = naive_predict_word_sense(sem_network, sentence_list, word_sense_dict)
    word_freq_guesses = {}
    for word_tuple in guess_list.keys():
        if plot_type == "word":
            key = word_tuple[0].name()
        elif plot_type == "sense":
            key = word_tuple
        if key not in word_freq_guesses.keys():
            word_freq_guesses[key] = []
        word_freq_guesses[key].extend(guess_list[word_tuple])
    word_appearances = []
    word_percent_correct = []
    for word in word_freq_guesses.keys():
        word_appearances.append(len(word_freq_guesses[word]))
        word_percent_correct.append(word_freq_guesses[word].count(True)/len(word_freq_guesses[word]))
    plt.scatter(word_appearances, word_percent_correct)
    if plot_type == "word":
        xlab = "Word Appearances"
    elif plot_type == "sense":
        xlab = "Sense Appearances"
    plt.xlabel(xlab)
    plt.ylabel("Percentage Correct")
    # Mess with this more
    plt.show()


def get_cooccurrence_plot(plot_type="word-word", prediction_type="naive", activation_base=2, decay_parameter=0.05,
                          constant_offset=0):
    sentence_list, word_sense_dict = extract_sentences()
    if prediction_type == "naive":
        guess_list = naive_predict_word_sense(sentence_list, word_sense_dict)
    elif prediction_type == "frequency":
        guess_list = frequency_predict_word_sense(sentence_list, word_sense_dict)
    elif prediction_type == "senseless":
        #sem_network = create_sem_network(sentence_list, activation_base=activation_base,
                                         #decay_parameter=decay_parameter, constant_offset=constant_offset)
        guess_list = senseless_predict_word_sense(sentence_list, word_sense_dict)
    else:
        return False
    #guess_list = naive_predict_word_sense(sentence_list, word_sense_dict)
    word_freq_guesses = {}
    for word_tuple in guess_list.keys():
        if plot_type == "word-word" or plot_type == "word-sense":
            word = word_tuple[0].name()
        else:
            word = word_tuple
        if word not in word_freq_guesses.keys():
            word_freq_guesses[word] = []
        word_freq_guesses[word].extend(guess_list[word_tuple])
    # Just do target = word, and context = word
    if plot_type == "word-sense":
        pair_counts = get_word_sense_pair_counts(sentence_list)
        abs_counts = get_absolute_sense_counts(sentence_list)
    elif plot_type == "sense-word":
        pair_counts = get_word_sense_pair_counts(sentence_list)
        abs_counts = get_absolute_word_counts(sentence_list)
    elif plot_type == "sense-sense":
        pair_counts = get_sense_pair_counts(sentence_list)
        abs_counts = get_absolute_sense_counts(sentence_list)
    else:
        pair_counts = get_word_pair_counts(sentence_list)
        abs_counts = get_absolute_word_counts(sentence_list)
    word_percent_correct = []
    cooccur_ratios = []
    for sentence in sentence_list:
        for word_index in range(len(sentence)):
            word = sentence[word_index]
            ratio = 0
            for context_word_index in range(word_index + 1, len(sentence)):
                context_word = sentence[context_word_index]
                # FIXME gotta make sure it pairs things up correctly depending on the case considered
                if plot_type == "word-sense":
                    word_pair = (word[0].name(), context_word)
                    ratio += math.log(pair_counts[word_pair] / abs_counts[context_word])
                elif plot_type == "sense-word":
                    word_pair = (context_word[0].name(), word)
                    ratio += math.log(pair_counts[word_pair] / abs_counts[context_word[0].name()])
                elif plot_type == "sense-sense":
                    if word[0] < context_word[0] or (word[0] == context_word[0] and word[1] <= context_word[1]):
                        word_pair = (word, context_word)
                    else:
                        word_pair = (context_word, word)
                    ratio += math.log(pair_counts[word_pair] / abs_counts[context_word])
                else:
                    if word[0] < context_word[0] or (word[0] == context_word[0] and word[1] <= context_word[1]):
                        word_pair = (word[0].name(), context_word[0].name())
                    else:
                        word_pair = (context_word[0].name(), word[0].name())
                    ratio += math.log(pair_counts[word_pair] / abs_counts[context_word[0].name()])
            cooccur_ratios.append(ratio)
            if plot_type == "word-word" or plot_type == "word-sense":
                word_percent_correct.append(
                    word_freq_guesses[word[0].name()].count(True) / len(word_freq_guesses[word[0].name()]))
            else:
                word_percent_correct.append(word_freq_guesses[word].count(True) / len(word_freq_guesses[word]))
    plt.scatter(cooccur_ratios, word_percent_correct)
    if plot_type == "word-sense":
        plt.xlabel("Cooccurrence Ratios (Word-Sense)")
    elif plot_type == "sense-word":
        plt.xlabel("Cooccurrence Ratios (Sense-Word)")
    elif plot_type == "sense-sense":
        plt.xlabel("Cooccurrence Ratios (Sense-Sense)")
    else:
        plt.xlabel("Cooccurrence Ratios (Word-Word)")
    plt.ylabel("Percentage Correct")
    plt.show()


def get_absolute_word_counts(sentence_list):
    absolute_word_counts = {}
    for sentence in sentence_list:
        for word in sentence:
            if word[0].name() not in absolute_word_counts.keys():
                absolute_word_counts[word[0].name()] = 0
            absolute_word_counts[word[0].name()] += 1
    return absolute_word_counts

def get_absolute_sense_counts(sentence_list):
    absolute_sense_counts = {}
    for sentence in sentence_list:
        for word in sentence:
            if word not in absolute_sense_counts.keys():
                absolute_sense_counts[word] = 0
            absolute_sense_counts[word] += 1
    return absolute_sense_counts



def get_word_pair_counts(sentence_list):
    word_pair_counts = {}
    for sentence in sentence_list:
        for word1_index in range(len(sentence)):
            for word2_index in range(word1_index + 1, len(sentence)):
                word1 = sentence[word1_index]
                word2 = sentence[word2_index]
                if word1[0] < word2[0] or (word1[0] == word2[0] and word1[1] <= word2[1]):
                    word_key = (word1[0].name(), word2[0].name())
                else:
                    word_key = (word2[0].name(), word1[0].name())
                if word_key not in word_pair_counts.keys():
                    word_pair_counts[word_key] = 0
                word_pair_counts[word_key] += 1
    return(word_pair_counts)



def get_sense_pair_counts(sentence_list):
    sense_pair_counts = {}
    for sentence in sentence_list:
        for word1_index in range(len(sentence)):
            for word2_index in range(word1_index + 1, len(sentence)):
                word1 = sentence[word1_index]
                word2 = sentence[word2_index]
                if word1[0] < word2[0] or (word1[0] == word2[0] and word1[1] <= word2[1]):
                    sense_key = (word1, word2)
                else:
                    sense_key = (word2, word1)
                if sense_key not in sense_pair_counts.keys():
                    sense_pair_counts[sense_key] = 0
                sense_pair_counts[sense_key] += 1
    return sense_pair_counts

def get_word_sense_pair_counts(sentence_list):
    word_sense_pair_counts = {}
    for sentence in sentence_list:
        for word1_index in range(len(sentence)):
            for word2_index in range(len(sentence)):
                if word1_index != word2_index:
                    word1 = sentence[word1_index]
                    word2 = sentence[word2_index]
                    word_sense_key = (word1[0].name(), word2)
                    if word_sense_key not in word_sense_pair_counts.keys():
                        word_sense_pair_counts[word_sense_key] = 0
                    word_sense_pair_counts[word_sense_key] += 1
    return word_sense_pair_counts


def get_corpus_stats():
    """
    Function to return the absolute counts of each word and each word sense, and the counts of cooccurrence between words
    and word senses
    Returns:
        absolute_word_counts (dict): A dictionary with keys the name of every lemma of the same word and values the
            number of times the word appears in the Semcor corpus.
        absolute_sense_counts (dict): A dictionary with keys a tuple with the lemma and synset of each word (tracking
            the sense of each word) and keys the number of times the tuple appears in the Semcor corpus.
        word_pair_counts (dict): A dictionary with keys a tuple of words appearing together and values the number of times
            the words appear together in the same sentence in the Semcor corpus.
        sense_pair_counts (dict): A dictionary with keys a tuple of two tuples each containing a lemma and its corresponding
            synset (tracking the sense of each word) representing two words of a certain sense occurrring together, and
            values the number of times the senses occur together in the same sentence in the Semcor corpus.
    """
    sentence_list = extract_sentences()[0]
    absolute_word_counts = {}
    absolute_sense_counts = {}
    word_pair_counts = {}
    sense_pair_counts = {}
    for sentence in sentence_list:
        for word1_index in range(len(sentence)):
            word1 = sentence[word1_index]
            if word1 in absolute_sense_counts.keys():
                absolute_sense_counts[word1] = absolute_sense_counts[word1] + 1
            else:
                absolute_sense_counts[word1] = 1
            if word1[0].name() in absolute_word_counts.keys():
                absolute_word_counts[word1[0].name()] = absolute_word_counts[word1[0].name()] + 1
            else:
                absolute_word_counts[word1[0].name()] = 1
            for word2_index in range(word1_index + 1, len(sentence)):
                word2 = sentence[word2_index]
                if word1[0] < word2[0] or (word1[0] == word2[0] and word1[1] <= word2[1]):
                    sense_key = (word1, word2)
                    word_key = (word1[0].name(), word2[0].name())
                else:
                    sense_key = (word2, word1)
                    word_key = (word2[0].name(), word1[0].name())
                if sense_key in sense_pair_counts.keys():
                    sense_pair_counts[sense_key] = sense_pair_counts[sense_key] + 1
                else:
                    sense_pair_counts[sense_key] = 1
                if word_key in word_pair_counts.keys():
                    word_pair_counts[word_key] = word_pair_counts[word_key] + 1
                else:
                    word_pair_counts[word_key] = 1
    return absolute_word_counts, absolute_sense_counts, word_pair_counts, sense_pair_counts


# Testing...
#get_simple_plot(2, 0.05, 0, plot_type="sense")
get_cooccurrence_plot(plot_type="word-word", prediction_type="naive")
get_cooccurrence_plot(plot_type="word-sense", prediction_type="naive")
get_cooccurrence_plot(plot_type="sense-word", prediction_type="naive")
get_cooccurrence_plot(plot_type="sense-sense", prediction_type="naive")