# Implements the RAT to test each model
import json
import csv
import os
from collections import defaultdict
from research import sentence_long_term_memory
from agent_cooccurrence import AgentCooccurrenceNGrams
from agent_spreading import AgentSpreadingNGrams
from agent_spreading_thresh_cooccurrence import AgentSpreadingThreshCoocNGrams
from agent_cooccurrence_thresh_spreading import AgentCoocThreshSpreadingNGrams


def run_rat(rat_file_link, sem_rel_link, stopwords_link, spreading=True, guess_type="dummy"):
    """
    Runs the RAT test.
    The stopwords_link is a text file that contains all stopwords
    Guessing mechanisms available at the moment: "dummy", "semantic", "cooccurrence"
    The rat_file_link is where the RAT file is stored containing all 3 context words and the correct answers.
    The sem_rel_link is where the semantic relations dictionary is stored, at this point either coming from SWOWEN, or
    SFFAN, or a combined dict containing both.
    """
    rat_file = csv.reader(open(rat_file_link))
    next(rat_file)
    with open(stopwords_link, "r") as stopwords_file:
        lines = stopwords_file.readlines()
        stopwords = []
        for l in lines:
            stopwords.append(l[:-1])
    if guess_type == "semantic":
        sem_rel_dict = json.load(open(sem_rel_link))
        sem_agent = AgentSpreadingNGrams(sem_rel_dict=sem_rel_dict)
        if spreading:
            spread_depth = -1
        else:
            spread_depth = 0
    elif guess_type == "cooccurrence":
        cooc_agent = AgentCooccurrenceNGrams(stopwords=stopwords)
    elif guess_type == "sem_thresh_cooc":
        sem_rel_dict = json.load(open(sem_rel_link))
        sem_thresh_cooc_agent = AgentSpreadingThreshCoocNGrams(stopwords=stopwords, sem_rel_dict=sem_rel_dict)
    guesses = []
    count = 0
    for trial in rat_file:
        count += 1
        print("count", count)
        if count > 10:
            break
        row = trial
        context = row[:3]
        true_guess = row[3]
        if guess_type == "dummy":
            guess = guess_rat_dummy(context)
        elif guess_type == "semantic":
            network = sem_agent.clear_sem_network(network, 0)
            guess = sem_agent.do_rat(context[0], context[1], context[2], spread_depth)
        elif guess_type == "cooccurrence":
            guess = cooc_agent.do_rat(context[0], context[1], context[2])
        else:
            raise ValueError(guess_type)
        guesses.append([true_guess, guess])
    accuracy = 0
    for trial in guesses:
        if trial[0] == trial[1]:
            accuracy += 1
    accuracy = accuracy / len(guesses)
    return guesses, accuracy


def guess_rat_dummy(context):
    return context[0]

def make_rat_dict(rat_file):
    """
    Produces a dictionary with the correct "answer" to the RAT problem as the key and the three context words as the value
    """
    rat_file = csv.reader(open(rat_file))
    next(rat_file)
    rat_dict = defaultdict(str)
    for trial in rat_file:
        row = trial
        context = row[:3]
        true_guess = row[3]
        rat_dict[true_guess] = context
    return rat_dict

def make_combined_dict(swowen_link, sffan_link):
    """
    Combines preprocessed swowen and sffan dictionaries (with word as key and all of its connections/associations as values)
        into a single dictionary, saved in file combined_spreading.json
    Returns nothing
    """
    filename = '/Users/lilygebhart/Downloads/combined_spreading.json'
    if not os.path.isfile(filename):
        swowen_dict = json.load(open(swowen_link))
        sffan_dict = json.load(open(sffan_link))
        combined_dict = defaultdict(set)
        for key in swowen_dict.keys():
            combined_dict[key].update(swowen_dict[key])
        for key in sffan_dict.keys():
            combined_dict[key].update(sffan_dict[key])
        for key in combined_dict.keys():
            combined_dict[key] = list(combined_dict[key])
        combined_file = open(filename, 'w')
        json.dump(combined_dict, combined_file)
        combined_file.close()
    return


def create_RAT_sem_network(sem_rel_dict, spreading=True, activation_base=2, decay_parameter=0.05,
                                constant_offset=0):
    """
    Builds a semantic network with each key word in the SWOWEN and South Florida Free Association Norms (SFFAN).
        Note that all words are stored at time 1.
    Parameters:
        SWOWEN_link (string): link to the SWOWEN preprocessed dictionary
        SFFAN_link (string): link to the SFFAN preprocessed dictionary
        spreading (bool): Whether to include the effects of spreading in creating the semantic network.
        activation_base (float): A parameter in the activation equation.
        decay_parameter (float): A parameter in the activation equation.
        constant_offset (float): A parameter in the activation equation.
        partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking at
            sentences 10000 - 14999.
    Returns:
        network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
    """
    if spreading:
        spread_depth = -1
    else:
        spread_depth = 0
    network = sentence_long_term_memory.sentenceLTM(
        activation_cls=(lambda ltm:
                        sentence_long_term_memory.SentenceCooccurrenceActivation(
                            ltm,
                            activation_base=activation_base,
                            constant_offset=constant_offset,
                            decay_parameter=decay_parameter
                        )))
    keys = list(sem_rel_dict.keys())
    for word in keys:
        assocs = sem_rel_dict[word]
        network.store(mem_id=word,
                      time=1,
                      spread_depth=spread_depth,
                      assocs=assocs)
    return network



# Testing.... ______________________________________________________________________________________________________
# sffan_link = '/Users/lilygebhart/Downloads/south_florida_free_assoc_norms/sf_spreading_sample.json'
# swowen_link = '/Users/lilygebhart/Downloads/SWOWEN_data/SWOWEN_spreading_sample.json'
# rat_link = '/Users/lilygebhart/Documents/GitHub/research/research/RAT/RAT_items.txt'
#
# results, accuracy = run_rat(rat_link, swowen_link, guess_type="semantic")
# print("semantic", results)
# file = open("rat_test_results_semantic.txt", mode="w")
# json.dump(results, file)
# file.close()
# print(accuracy)
#
# results, accuracy = run_rat(rat_link, swowen_link, guess_type="cooccurrence")
# print("cooccurrence", results)
# file = open("rat_test_results_cooc.txt", mode="w")
# json.dump(results, file)
# file.close()
# print(accuracy)


