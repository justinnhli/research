from wsd_task import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
import statistics
import math


def get_simple_plot(activation_base, decay_parameter, constant_offset, plot_type="word"):
    """
    Produces simple scatterplot of percentage correct vs. word appearances for each word in the corpus.
    """
    sentence_list, word_sense_dict = extract_sentences()
    sem_network = create_sem_network(sentence_list, activation_base, decay_parameter, constant_offset)
    guess_list = get_corpus_accuracy("context_sense", sentence_list, word_sense_dict)[0]
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


def get_cooccurrence_plot(guess_type, plot_type):
    # plot types are # correct over all senses of target word based on (1) other word in sentence sense and (2) other
    #   word in sentence word
    # plot type can be other_word or other_sense
    sentence_list, word_sense_dict = extract_sentences()
    word_word_cooccurrences, sense_word_cooccurrences, sense_sense_cooccurrences, sense_frequencies = precompute_cooccurrences(sentence_list)
    word_counts, sense_counts = precompute_word_sense(sentence_list)
    if guess_type == "context_word":
        accuracy_dict = get_corpus_accuracy("context_word", sentence_list, word_sense_dict)
    elif guess_type == "context_sense":
        accuracy_dict = get_corpus_accuracy("context_sense", sentence_list, word_sense_dict)
    elif guess_type == "frequency":
        accuracy_dict = get_corpus_accuracy("frequency", sentence_list, word_sense_dict)
    else:
        return False
    y_accuracies = []
    x_cooccurrences = []
    for sentence in sentence_list:
        for target_index in range(len(sentence)):
            target_sense = sentence[target_index]
            target_word = target_sense[0].name()
            cumulative_cooccurrrence_ratio = 0
            for other_index in range(len(sentence)):
                if other_index != target_index:
                    other_sense = sentence[other_index]
                    other_word = other_sense[0].name()
                    if plot_type == "other_word":
                        cumulative_cooccurrrence_ratio += math.log(word_word_cooccurrences[(target_word, other_word)]/word_counts[target_word])
                    elif plot_type == "other_sense":
                        cumulative_cooccurrrence_ratio += math.log(sense_word_cooccurrences[(other_sense, target_word)]/word_counts[target_word])
                    else:
                        raise ValueError(plot_type)
                #Make x the number of times other words in sentence cooccur with word we are interested in
                target_word_accuracy_list = []
                for sense in word_sense_dict[target_word]:
                    target_word_accuracy_list.append(accuracy_dict[sense])
                flat_target_accuracy_list = sum(target_word_accuracy_list, [])
                y_accuracies.append(flat_target_accuracy_list.count(True) / len(flat_target_accuracy_list))
                x_cooccurrences.append(cumulative_cooccurrrence_ratio)
    plt.scatter(x_cooccurrences, y_accuracies)
    if guess_type == "context_word":
        plt.title("Accuracy vs. Cooccurrence (Context Word)")
    elif guess_type == "context_sense":
        plt.title("Accuracy vs. Cooccurrence (Context Sense)")
    elif guess_type == "frequency":
        plt.title("Accuracy vs. Cooccurrence (Frequency)")
    else:
        raise ValueError(guess_type)
    if plot_type == "other_word":
        plt.xlabel("Word Cooccurrence")
    elif plot_type == "other_sense":
        plt.xlabel("Sense Cooccurrence")
    else:
        raise ValueError(plot_type)
    plt.ylabel("Target Word Accuracy")
    plt.show()


def get_cooccurrence_sentence_bin_plot(guess_type, plot_type, bin_width, bin_colors=True, error=False):
    sentence_list, word_sense_dict = extract_sentences()
    word_word_cooccurrences, sense_word_cooccurrences, sense_sense_cooccurrences, sense_frequencies = precompute_cooccurrences(
        sentence_list)
    word_counts, sense_counts = precompute_word_sense(sentence_list)
    target_accuracy_list = []
    target_cooc_list = []
    for sentence in sentence_list:
        for target_index in range(len(sentence)):
            target_sense = sentence[target_index]
            target_word = target_sense[0].name()
            if guess_type == "context_word":
                guess = guess_word_sense_context_word(target_index, sentence, word_sense_dict, sense_word_cooccurrences, word_word_cooccurrences)
            elif guess_type == "context_sense":
                guess = guess_word_sense_context_sense(target_index, sentence, word_sense_dict, sense_word_cooccurrences, sense_sense_cooccurrences)
            elif guess_type == "frequency":
                guess = guess_word_sense_frequency(target_index, sentence, word_sense_dict, sense_frequencies)
            else:
                raise ValueError(guess_type)
            if guess == target_sense:
                target_accuracy_list.append(1)
            else:
                target_accuracy_list.append(0)
            temp_cooccurrence = 0
            for other_index in range(len(sentence)):
                if other_index == target_index:
                    continue
                other_sense = sentence[other_index]
                other_word = other_sense[0].name()
                if plot_type == "other_word":
                    temp_cooccurrence += math.log(
                            word_word_cooccurrences[(target_word, other_word)] / word_counts[target_word])
                elif plot_type == "other_sense":
                    temp_cooccurrence += math.log(
                        sense_word_cooccurrences[(other_sense, target_word)] / word_counts[target_word])
                else:
                    raise ValueError(plot_type)
            target_cooc_list.append(temp_cooccurrence)
    max_bin_width = math.ceil(max(target_cooc_list)/bin_width) * bin_width
    min_bin_width = math.floor(min(target_cooc_list)/bin_width) * bin_width
    x_cooccurrences = []
    y_accuracies = []
    y_err = []
    z_binsizes = []
    for bin_index in range(min_bin_width, max_bin_width, bin_width):
        bin_values = [target_accuracy_list[ii] for ii in range(len(target_cooc_list)) if target_cooc_list[ii] >= bin_index and target_cooc_list[ii] < bin_index + bin_width]
        if len(bin_values) > 0:
            x_cooccurrences.append(bin_index + (bin_width / 2))
            y_accuracies.append(statistics.mean(bin_values))
            z_binsizes.append(math.log(len(bin_values)))
            if len(bin_values) > 1:
                y_err.append(statistics.stdev(bin_values))
            else:
                y_err.append(0)
    #plt.scatter(x_cooccurrences, y_accuracies)
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(x_cooccurrences, y_accuracies, c=z_binsizes, s=80)
    if bin_colors:
        legend = ax.legend(*scatter.legend_elements(), loc="lower right",
                       fontsize='x-small', title=" Log Bin Size")
        ax.add_artist(legend)
    if error:
        ax.errorbar(x_cooccurrences, y_accuracies, yerr=y_err, capsize=1, fmt='none')
    if guess_type == "context_word":
        ax.set_title("Accuracy vs. Bin Cooccurrence (Context Word)")
    elif guess_type == "context_sense":
        ax.set_title("Accuracy vs. Bin Cooccurrence (Context Sense)")
    elif guess_type == "frequency":
        ax.set_title("Accuracy vs. Bin Cooccurrence (Frequency)")
    else:
        raise ValueError(guess_type)
    if plot_type == "other_word":
        ax.set_xlabel("Word Cooccurrence (bin = " + str(bin_width) + ")")
    elif plot_type == "other_sense":
        ax.set_xlabel("Sense Cooccurrence (bin = " + str(bin_width) + ")")
    else:
        raise ValueError(plot_type)
    ax.set_ylabel("Avg. Binned Target Word Accuracy")
    plt.show()



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
#get_cooccurrence_sentence_bin_plot(plot_type="other_sense", guess_type="frequency", bin_width=1)
#get_cooccurrence_sentence_bin_plot(plot_type="other_sense", guess_type="frequency", bin_width=20)
get_cooccurrence_sentence_bin_plot(plot_type="other_sense", guess_type="frequency", bin_width=50)
#get_cooccurrence_plot(plot_type="other_sense", guess_type="context_sense")

#get_cooccurrence_plot(plot_type="other_word", guess_type="frequency")
#get_cooccurrence_plot(plot_type="other_sense", guess_type="frequency")