# Implements the RAT to test each model
import json
import csv
from collections import defaultdict
from research import agent_sem_network
from research.wsd_task import clear_sem_network


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


def run_rat(rat_file_link, swowen_link, sffan_link, spreading=True, guess_type="dummy"):
    rat_file = csv.reader(open(rat_file_link))
    next(rat_file)
    if guess_type == "semantic":
        combined_dict = agent_sem_network.make_combined_dict(swowen_link, sffan_link)
        network = agent_sem_network.create_combined_sem_network(combined_dict)
        if spreading:
            spread_depth = -1
        else:
            spread_depth = 0
    guesses = []
    counter = 0
    for trial in rat_file:
        counter += 1
        if counter > 10:
            break
        row = trial
        context = row[:3]
        true_guess = row[3]
        if guess_type == "dummy":
            guess = guess_rat_dummy(context)
        if guess_type == "semantic":
            network = clear_sem_network(network, 0)
            guess = guess_rat_semantic(context, network, spread_depth)
        guesses.append([true_guess, guess])
    accuracy = 0
    for trial in guesses:
        if trial[0] == trial[1]:
            accuracy += 1
    accuracy = accuracy / len(guesses)
    return guesses, accuracy


def guess_rat_dummy(context):
    return context[0]


def guess_rat_semantic(context, network, spread_depth):
    for word_index in range(3):
        network.store(mem_id=context[word_index], time=2, spread_depth=spread_depth)
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







def solve_rat_spreading(swowen_link, sffan_link, rat_link):
    results = defaultdict(set)
    combined_dict = agent_sem_network.make_combined_dict(swowen_link, sffan_link)
    print(len(list(combined_dict.keys())))
    rat_dict = make_rat_dict(rat_link)
    counter = 0
    for answer in rat_dict.keys():
        print("word", counter)
        context_words = rat_dict[answer]
        results[answer] = get_semantic_guess(combined_dict, context_words[0], context_words[1], context_words[2], max_counter=5)
        counter += 1
    return results

def get_semantic_guess(network_dict, context1, context2, context3, max_counter=5):
    context1_links = set(network_dict[context1])
    context2_links = set(network_dict[context2])
    context3_links = set(network_dict[context3])
    all_links = set(network_dict.keys())
    guess_pool = context1_links & context2_links & context3_links
    counter = 0
    while not guess_pool:
        context1_adder = all_links & context1_links
        context2_adder = all_links & context2_links
        context3_adder = all_links & context3_links
        for elem in context1_adder:
            context1_links.update(network_dict[elem])
        for elem in context2_adder:
            context2_links.update(network_dict[elem])
        for elem in context3_adder:
            context3_links.update(network_dict[elem])
        guess_pool = context1_links & context2_links & context3_links
        counter += 1
        if counter == max_counter:
            break
    return guess_pool



def get_all_network_links(network_dict, word):
    """
    Returns all connections to a word (regardless of graph distance) in the semantic network.
    For solving the semantic network problem.
    """
    past_links = set()
    dict_words = set(network_dict.keys())
    curr_links = network_dict[word]
    while curr_links:
        next_links = set()
        for link in curr_links:
            if link not in past_links:
                past_links.update(link)
                if link in dict_words:
                    next_links.update(network_dict[link])
        curr_links = next_links.difference(past_links)
    return past_links





# Testing.... ______________________________________________________________________________________________________
sffan_link = '/Users/lilygebhart/Downloads/south_florida_free_assoc_norms/sf_spreading_sample.json'
swowen_link = '/Users/lilygebhart/Downloads/SWOWEN_data/SWOWEN_spreading_sample.json'
rat_link = '/Users/lilygebhart/Documents/GitHub/research/research/RAT/RAT_items.txt'

results, accuracy = run_rat(rat_link, swowen_link, sffan_link, guess_type="semantic")
print(results)
file = open("rat_test_results_semantic.txt", mode="w")
json.dump(results, file)
file.close()
print(accuracy)