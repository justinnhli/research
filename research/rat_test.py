# Implements the RAT to test each model
import json
import csv
import os
from collections import defaultdict
from agent_cooccurrence import AgentCooccurrenceNGrams
from agent_spreading import AgentSpreadingNGrams
from agent_spreading_thresh_cooccurrence import AgentSpreadingThreshCoocNGrams
from agent_cooccurrence_thresh_spreading import AgentCoocThreshSpreadingNGrams
from agent_joint_probability import AgentJointProbabilityNGrams, AgentJointVarianceNGrams, AgentAdditiveProbabilityNGrams
from agent_cooccurrence_thresh_spreading_soft import AgentSoftCooccurrenceThreshSpreadingNGrams
from agent_oracle import AgentOracleNGrams
from n_gram_cooccurrence.google_ngrams import GoogleNGram


def run_rat(guess_type, sem_rel_source, ngrams=GoogleNGram('~/ngram'), spreading=True, clear="never", activation_base=2,
            decay_parameter=0.05, constant_offset=0, threshold=0.0):
    """
    Runs the RAT test.
    The stopwords_link is a text file that contains all stopwords
    Guessing mechanisms available at the moment: "dummy", "semantic", "cooccurrence"
    The rat_file_link is where the RAT file is stored containing all 3 context words and the correct answers.
    The sem_rel_source is  what semantic relations dict to use, with options "SWOWEN", "SFFAN", or "combined"
    """
    with open('./nltk_english_stopwords', "r") as stopwords_file:
        lines = stopwords_file.readlines()
        stopwords = []
        for l in lines:
            stopwords.append(l[:-1])
    rat_file = csv.reader(open('./RAT/RAT_items.txt'))
    next(rat_file)
    if guess_type == "oracle":
        sem_rel_swowen_link = "./semantic_relations_lists/SWOWEN_sem_rel_dict.json"
        sem_rel_sffan_link = "./semantic_relations_lists/SFFAN_sem_rel_dict.json"
        sem_rel_combined_link = "./semantic_relations_lists/combined_sem_rel_dict.json"
        sem_rel_dict_swowen = json.load(open(sem_rel_swowen_link, "r"))
        sem_rel_dict_sffan = json.load(open(sem_rel_sffan_link, "r"))
        sem_rel_dict_combined = json.load(open(sem_rel_combined_link, "r"))
    elif guess_type != "cooccurrence" and guess_type != "frequency":
        sem_rel_link = "./semantic_relations_lists/" + sem_rel_source
        if guess_type == "cooc_thresh_sem":
            sem_rel_link += "_thresh"
        sem_rel_link += "_sem_rel_dict.json"
        if os.path.isfile(sem_rel_link):
            sem_rel_file = open(sem_rel_link, "r")
            sem_rel_dict = json.load(sem_rel_file)
        else:
            raise ValueError(sem_rel_link)
    if guess_type == "semantic":
        sem_agent = AgentSpreadingNGrams(stopwords, source=sem_rel_source, clear="never",
                                         activation_base=activation_base,
                                         decay_parameter=decay_parameter, constant_offset=constant_offset)
    elif guess_type == "cooccurrence" or guess_type == "frequency":
        cooc_agent = AgentCooccurrenceNGrams(stopwords=stopwords)
    elif guess_type == "oracle":
        oracle_agent = AgentOracleNGrams(sem_rel_dict_combined=sem_rel_dict_combined,
                                         sem_rel_dict_sffan=sem_rel_dict_sffan,
                                         sem_rel_dict_swowen=sem_rel_dict_swowen, stopwords=stopwords)
    elif guess_type == "sem_thresh_cooc":
        sem_thresh_cooc_agent = AgentSpreadingThreshCoocNGrams(stopwords=stopwords, sem_rel_dict=sem_rel_dict,
                                                               ngrams=ngrams)
    elif guess_type == "cooc_thresh_sem":
        cooc_thresh_sem_agent = AgentCoocThreshSpreadingNGrams(stopwords=stopwords,
                                                               source=sem_rel_source,
                                                               ngrams=ngrams,
                                                               spreading=spreading,
                                                               clear=clear,
                                                               activation_base=activation_base,
                                                               decay_parameter=decay_parameter,
                                                               constant_offset=constant_offset,
                                                               threshold=threshold)
    elif guess_type == "soft_cooc_thresh_sem":
        soft_cooc_thresh_sem_agent = AgentSoftCooccurrenceThreshSpreadingNGrams(source=sem_rel_source,
                                                                                stopwords=stopwords,
                                                                                ngrams=ngrams,
                                                                                spreading=spreading,
                                                                                clear=clear,
                                                                                activation_base=activation_base,
                                                                                decay_parameter=decay_parameter,
                                                                                constant_offset=constant_offset)
    elif guess_type == "joint_probability":
        joint_agent = AgentJointProbabilityNGrams(source=sem_rel_source,
                                                  stopwords=stopwords,
                                                  ngrams=ngrams,
                                                  spreading=True,
                                                  clear=clear,
                                                  activation_base=activation_base,
                                                  decay_parameter=decay_parameter,
                                                  constant_offset=constant_offset)
    elif guess_type == "additive_probability":
        add_agent = AgentAdditiveProbabilityNGrams(source=sem_rel_source,
                                                  stopwords=stopwords,
                                                  ngrams=ngrams,
                                                  spreading=True,
                                                  clear=clear,
                                                  activation_base=activation_base,
                                                  decay_parameter=decay_parameter,
                                                  constant_offset=constant_offset)
    elif "joint_variance" in guess_type:
        if guess_type == "joint_variance_stdev":
            var_type = "stdev"
        else:
            var_type = "maxdiff"
        jointvar_agent = AgentJointVarianceNGrams(source=sem_rel_source,
                                                  stopwords=stopwords,
                                                  ngrams=ngrams,
                                                  spreading=True,
                                                  clear=clear,
                                                  activation_base=activation_base,
                                                  decay_parameter=decay_parameter,
                                                  constant_offset=constant_offset,
                                                  var_type=var_type)
    guesses = []
    for trial in rat_file:
        row = trial
        context = row[:3]
        answer = row[3]
        if guess_type == "dummy":
            guess = guess_rat_dummy(context)
        elif guess_type == "semantic":
            sem_agent.clear_sem_network(0)
            guess = sem_agent.do_rat(context[0], context[1], context[2])
        elif guess_type == "cooccurrence":
            guess = cooc_agent.do_rat(context[0], context[1], context[2])
        elif guess_type == "frequency":
            guess = guess_rat_frequency(context[0], context[1], context[2], cooc_agent)
        elif guess_type == "oracle":
            oracle_agent.sffan_network = oracle_agent.spreading_sffan_agent.clear_sem_network(
                oracle_agent.sffan_network, 0)
            oracle_agent.swowen_network = oracle_agent.spreading_swowen_agent.clear_sem_network(
                oracle_agent.swowen_network, 0)
            oracle_agent.combined_network = oracle_agent.spreading_combined_agent.clear_sem_network(
                oracle_agent.combined_network, 0)
            guess = oracle_agent.do_rat(context[0], context[1], context[2], answer.upper())
        elif guess_type == "cooc_thresh_sem":
            cooc_thresh_sem_agent.clear_sem_network(0)
            guess = cooc_thresh_sem_agent.do_rat(context[0], context[1], context[2])
        elif guess_type == "soft_cooc_thresh_sem":
            soft_cooc_thresh_sem_agent.clear_sem_network(0)
            guess = soft_cooc_thresh_sem_agent.do_rat(context[0], context[1], context[2])
        elif guess_type == "sem_thresh_cooc":
            guess = sem_thresh_cooc_agent.do_rat(context[0], context[1], context[2])
        elif guess_type == "joint_probability":
            guess = joint_agent.do_rat(context[0], context[1], context[2])
        elif guess_type == "additive_probability":
            guess = add_agent.do_rat(context[0], context[1], context[2])
        elif "joint_variance" in guess_type:
            guess = jointvar_agent.do_rat(context[0], context[1], context[2])
        else:
            raise ValueError(guess_type)
        guesses.append([answer, guess])
        print([answer, guess])
    num_correct_lb = 0
    num_correct_ub = 0
    num_guessed = 0
    for trial in guesses:
        answer = trial[0]
        trial_guesses = trial[1]
        if not trial_guesses:
            continue
        elif len(trial_guesses) == 1:
            num_guessed += 1
            guess = trial_guesses[0]
            if answer.upper() == guess.upper():
                num_correct_lb += 1
                num_correct_ub += 1
        else:
            num_guessed += 1
            for guess in trial_guesses:
                if answer.upper() == guess.upper():
                    num_correct_ub += 1
    guesses_filename = "./results/rat_" + guess_type
    if guess_type != "cooccurrence":
        guesses_filename += "_" + sem_rel_source
    guesses_filename += ".json"
    file = open(guesses_filename, 'w')
    json.dump(guesses, file)
    file.close()
    print("most correct:", num_correct_ub, "least correct:", num_correct_lb, "num guessed:", num_guessed)
    return guesses


def guess_rat_dummy(context):
    return context[0]

def guess_rat_frequency(context1, context2, context3, cooc_agent):
    word_counts_list = cooc_agent.cooc_cache[context1.upper(), context2.upper(), context3.upper()]
    print(word_counts_list)
    max_freq = 0
    max_guess = []
    for candidate in word_counts_list:
        guess = candidate[0]
        freq = cooc_agent.ngrams.get_ngram_counts(guess)[guess]
        print(guess, freq)
        if freq > max_freq:
            max_freq = freq
            max_guess = [guess]
        elif freq == max_freq:
            max_guess.append(guess)
    return max_guess

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

def get_avg_uniform_probability(guess_type, source=None):
    with open('./nltk_english_stopwords', "r") as stopwords_file:
        lines = stopwords_file.readlines()
        stopwords = []
        for l in lines:
            stopwords.append(l[:-1])
    rat_file = csv.reader(open('./RAT/RAT_items.txt'))
    next(rat_file)
    unif_sum = 0
    for trial in rat_file:
        row = trial
        context = row[:3]
        answer = row[3].upper()
        if guess_type == "cooccurrence":
            agent = AgentCooccurrenceNGrams(stopwords=stopwords)
            cooc_prob_candidates = agent.cooc_cache[tuple([context[0].upper(), context[1].upper(), context[2].upper()])]
            cooc_candidates = [elem[0] for elem in cooc_prob_candidates]
            print(cooc_candidates)
            if answer in cooc_candidates:
                unif_sum += 1/len(cooc_candidates)
        if guess_type == "spreading" and source is not None:
            agent = AgentSpreadingNGrams(stopwords=stopwords, source=source)
            context_list = [context[0].upper(), context[1].upper(), context[2].upper()]
            for context in context_list:
                agent.network.store(mem_id=context.upper(), time=2, spread_depth=-1)
            sem_candidates = sorted(set(agent.network.activation.activations.keys()))
            print(sem_candidates)
            if answer in sem_candidates:
                unif_sum += 1/len(sem_candidates)
            agent.clear_sem_network(0)
    return unif_sum / 142

# Testing.... ______________________________________________________________________________________________________

