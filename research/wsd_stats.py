from wsd_task import *
from collections import defaultdict
import os


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


def get_results_file(partition, guess_method):
    base_directory = "/Users/lilygebhart/Documents/GitHub/research/research/results_partitions_5000/partition"
    partition_files = os.listdir(base_directory + str(partition))
    if guess_method == "naive_semantic":
        file = [file for file in partition_files if guess_method in file and "spreading" not in file]
    elif guess_method == "word" or guess_method == "never" or guess_method == "sentence":
        file = [file for file in partition_files if guess_method in file and "spreading" in file]
    else:
        file = [file for file in partition_files if guess_method in file]
    return base_directory + str(partition) + "/" + file[0]


def get_method_accuracy_breakdown(handle,partition):
    accuracy_breakdown_dict = defaultdict(list)
    indexed_guess_dict = get_indexed_guess_dict(handle,
                                                num_sentences=5000,
                                                partition=partition,
                                                indexed=False)
    for word in list(indexed_guess_dict.keys()):
        word_instance_dict = indexed_guess_dict[word]
        for index in list(word_instance_dict.keys()):
            guess = word_instance_dict[index]
            if True in guess:
                if len(guess) == 1:
                    guess_acc_key = "true_one"
                else:
                    guess_acc_key = "true_mult"
            else:
                if len(guess) == 1:
                    guess_acc_key = "false_one"
                else:
                    guess_acc_key = "false_mult"
            accuracy_breakdown_dict[guess_acc_key].append(tuple([word, index]))
    return accuracy_breakdown_dict


def get_method_accuracy_breakdown_counts(handle):
    acc_breakdown_dict = get_method_accuracy_breakdown(handle)
    breakdown_counts = defaultdict(int)
    breakdown_counts["true_one"] = len(acc_breakdown_dict["true_one"])
    breakdown_counts["true_mult"] = len(acc_breakdown_dict["true_mult"])
    breakdown_counts["false_one"] = len(acc_breakdown_dict["false_one"])
    breakdown_counts["false_mult"] = len(acc_breakdown_dict["false_mult"])
    return breakdown_counts


def get_sentence_positions(num_sentences=-1, partition=1):
    sentence_position_dict = defaultdict(list)
    sentence_list, wsd = extract_sentences(num_sentences, partition)
    for sent_index in range(len(sentence_list)):
        sentence = sentence_list[sent_index]
        for word_index in range(len(sentence)):
            word = sentence[word_index]
            sentence_position_dict[word].append(tuple([sent_index, word_index]))
    return sentence_position_dict


def get_indexed_guess_dict(handle, num_sentences=-1, partition=1, indexed=False):
    guesses_json = json.load(open(handle))
    indexed_guess_dict = defaultdict(dict)
    if not indexed:
        sentence_position_dict = get_sentence_positions(num_sentences, partition)
    for index in range(len(guesses_json)):
        word_appearances = guesses_json[index]
        word = tuple(word_appearances[0])
        guess_list = word_appearances[1]
        instance_dict = defaultdict(dict)
        if not indexed:
            word_positions = sentence_position_dict[word]
        for instance_index in range(len(guess_list)):
            instance = guess_list[instance_index]
            if indexed:
                instance_dict[instance[0]] = instance[1]
            else:
                instance_dict[word_positions[instance_index]] = instance
        indexed_guess_dict[tuple(word)] = instance_dict
    return indexed_guess_dict


def get_pairwise_guess_comp(handle1, handle2, num_sentences=-1, partition=1):
    indexed_guess_dict1 = get_indexed_guess_dict(handle1,
                                                 num_sentences=num_sentences,
                                                 partition=partition,
                                                 indexed=False)
    indexed_guess_dict2 = get_indexed_guess_dict(handle2,
                                                 num_sentences=num_sentences,
                                                 partition=partition,
                                                 indexed=False)
    comp_guesses_dict = defaultdict(list)
    for word in list(indexed_guess_dict1.keys()):
        word_instance_dict1 = indexed_guess_dict1[word]
        word_instance_dict2 = indexed_guess_dict2[word]
        for index in list(word_instance_dict1.keys()):
            guess1 = word_instance_dict1[index]
            guess2 = word_instance_dict2[index]
            if True in guess1 and True in guess2:
                if len(guess1) == 1 and len(guess2) == 1:
                    comp_guesses_key = tuple(["true_one", "true_one"])
                elif len(guess1) == 1 and len(guess2) > 1:
                    comp_guesses_key = tuple(["true_one", "true_mult"])
                elif len(guess1) > 1 and len(guess2) == 1:
                    comp_guesses_key = tuple(["true_mult", "true_one"])
                else:
                    comp_guesses_key = tuple(["true_mult", "true_mult"])
            elif True in guess1 and True not in guess2:
                if len(guess1) == 1 and len(guess2) == 1:
                    comp_guesses_key = tuple(["true_one", "false_one"])
                elif len(guess1) > 1 and len(guess2) == 1:
                    comp_guesses_key = tuple(["true_mult", "false_one"])
                elif len(guess1) == 1 and len(guess2) > 1:
                    comp_guesses_key = tuple(["true_one", "false_mult"])
                else:
                    comp_guesses_key = tuple(["truemult", "false_mult"])
            elif True not in guess1 and True in guess2:
                if len(guess1) == 1 and len(guess2) == 1:
                    comp_guesses_key = tuple(["false_one", "true_one"])
                elif len(guess1) > 1 and len(guess2) == 1:
                    comp_guesses_key = tuple(["false_mult", "true_one"])
                elif len(guess1) == 1 and len(guess2) > 1:
                    comp_guesses_key = tuple(["false_one", "true_mult"])
                else:
                    comp_guesses_key = tuple(["false_mult", "true_mult"])
            else:
                # if everything is false in here
                if len(guess1) > 1 and len(guess2) > 1:
                    comp_guesses_key = tuple(["false_mult", "false_mult"])
                elif len(guess1) == 1 and len(guess2) == 1:
                    comp_guesses_key = tuple(["false_one", "false_one"])
                elif len(guess1) > 1 and len(guess2) == 1:
                    comp_guesses_key = tuple(["false_mult", "false_one"])
                else:
                    comp_guesses_key = tuple(["false_one", "false_mult"])
            comp_guesses_dict[comp_guesses_key].append(tuple([word, index]))
    return comp_guesses_dict


def get_mult_simple_guess_comps(handle_list, num_sentences=-1, partition=1):
    guess_dicts = []
    comp_guesses_dict = defaultdict(list)
    for handle in handle_list:
        guess_dicts.append(get_indexed_guess_dict(handle,
                                                  num_sentences=num_sentences,
                                                  partition=partition,
                                                  indexed=False))
    sample_dict = guess_dicts[0]
    guess_dict_keys = list(sample_dict.keys())
    for word_index in range(len(guess_dict_keys)):
        word = guess_dict_keys[word_index]
        index_keys = list(sample_dict[word].keys())
        for index in index_keys:
            true_list = []
            for d in range(len(guess_dicts)):
                guess_accuracies = guess_dicts[d][word][index]
                if True in guess_accuracies:
                    true_list.append(True)
                else:
                    true_list.append(False)
            true_list = list(set(true_list))
            if len(true_list) == 1:
                if true_list[0]:
                    comp_guesses_key = "all_true"
                else:
                    comp_guesses_key = "all_false"
            else:
                comp_guesses_key = "true_false"
            comp_guesses_dict[comp_guesses_key].append(tuple([word, index]))
    return comp_guesses_dict


def get_pairwise_counts(handle1, handle2, num_sentences, partition, detailed=False):
    pairwise_guesses = get_pairwise_guess_comp(handle1, handle2, num_sentences=num_sentences, partition=partition)
    count_difs = defaultdict(int)
    categories = [tuple(["true_one", "true_one"]),
                  tuple(["true_one", "true_mult"]),
                  tuple(["true_mult", "true_one"]),
                  tuple(["true_mult", "true_mult"]),
                  tuple(["true_one", "false_one"]),
                  tuple(["true_mult", "false_one"]),
                  tuple(["true_one", "false_mult"]),
                  tuple(["true_mult", "false_mult"]),
                  tuple(["false_one", "true_one"]),
                  tuple(["false_mult", "true_one"]),
                  tuple(["false_one", "true_mult"]),
                  tuple(["false_mult", "true_mult"]),
                  tuple(["false_mult", "false_mult"]),
                  tuple(["false_one", "false_one"]),
                  tuple(["false_mult", "false_one"]),
                  tuple(["false_one", "false_mult"])]
    if not detailed:
        for cat in categories[0:4]:
            count_difs["true_true"] += len(pairwise_guesses[cat])
        for cat in categories[4:8]:
            count_difs["true_false"] += len(pairwise_guesses[cat])
        for cat in categories[8:12]:
            count_difs["false_true"] += len(pairwise_guesses[cat])
        for cat in categories[12:]:
            count_difs["false_false"] += len(pairwise_guesses[cat])
    else:
        for cat in categories:
            count_difs[cat] += len(pairwise_guesses[cat])
    return count_difs


def get_contradictory_results(handle_list1, handle_list2, num_sentences=-1, partition=1):
    """
    Pits everything in handle_list1 against everything in handle_list2
    """
    comps_1 = get_mult_simple_guess_comps(handle_list1, num_sentences=num_sentences, partition=partition)
    comps_2 = get_mult_simple_guess_comps(handle_list2, num_sentences=num_sentences, partition=partition)
    trues1 = set(comps_1["all_true"])
    falses1 = set(comps_1["all_false"])
    trues2 = set(comps_2["all_true"])
    falses2 = set(comps_2["all_false"])
    true_false = trues1 & falses2
    false_true = falses1 & trues2
    return true_false, false_true


def get_contradictory_partition_results(false_guess_methods, true_guess_methods):
    for partition in range(1, 7):
        false_directories = []
        for method in false_guess_methods:
            false_directories.append(get_results_file(partition=partition, guess_method=method))
        true_directories = []
        for method in true_guess_methods:
            false_directories.append(get_results_file(partition=partition, guess_method=method))
        true_false, results = get_contradictory_results(false_directories, true_directories, num_sentences=5000,
                                                        partition=partition)
        print("partition: ", partition)
        print("Number of Discrepancies", len(results))
        presorted_results = []
        frequency_dict = defaultdict(int)
        sent_dict = defaultdict(list)
        for elem in list(results):
            presorted_results.append(tuple([elem[1], elem[0]]))
            frequency_dict[elem[0]] += 1
            sent_dict[elem[1][0]].append(elem[0])
        sorted_results = sorted(presorted_results)

        frequency_list = []
        for i in list(frequency_dict.keys()):
            frequency_list.append([frequency_dict[i], i])
        frequency_list = (sorted(frequency_list))[::-1]
        print("freq list ")
        for i in frequency_list:
            if i[0] > 1:
                print(i)

        sent_list = []
        for i in list(sent_dict.keys()):
            if len(sent_dict[i]) > 1:
                sent_list.append([i, sent_dict[i]])
        print("sentlist ")
        for i in sorted(sent_list):
            print(i)
        print()


# Testing --------------------------------------------------------------------------------------------------------------
# loc_words = [('be', 'exist.v.01'),
# ('exist', 'exist.v.01')]
# # Getting frequencies of certain words in the corpus.
# for partition in [1,4]:
#     print("Partition:", partition)
#     sent_list = extract_sentences(num_sentences=5000, partition=partition)[0]
#     frequencies = precompute_word_sense(sent_list)[1]
#     for sent_index in range(len(sent_list)):
#         sent = sent_list[sent_index]
#         for word_index in range(len(sent)):
#             word = sent[word_index]
#             if word in loc_words:
#                 print(word, ", sent=", sent_index, ", word=", word_index)


# for partition in [1,4]:
#     sent_list = extract_sentences(num_sentences=5000, partition=partition)[0]
#     sem_rels = get_semantic_relations_dict(sent_list, partition=partition, outside_corpus=False)
#     for word in [('constitute', 'constitute.v.01'), ('comprise', 'constitute.v.01'), ('represent', 'constitute.v.01'), ('make_up', 'constitute.v.01'), ('exist', 'exist.v.01') ]:
#         if word in list(sem_rels.keys()):
#             print("Partition:", partition, ", Word:", word)
#             print(list(sem_rels[word].keys()))
#             primary_connects = set()
#             print("Primary Connections:")
#             for connect in list(sem_rels[word].keys()):
#                 print(connect, ":", sem_rels[word][connect])
#                 primary_connects.update(sem_rels[word][connect])
#             print("Number of Primary Connections:", len(list(primary_connects)))
#             print()
#


# for partition in range(1,7):
#     print("partition :", partition)
#     accs = get_method_accuracy_breakdown(get_results_file(partition=partition, guess_method="context_sense"), partition)
#     #sent_list = extract_sentences(num_sentences=5000, partition=partition)[0]
#     #sents = sum(sent_list, [])
#     #sem_rels = get_semantic_relations_dict(sent_list[0], 1, False)
#     occurrence_dict = defaultdict(dict)
#     print(list(accs.keys()))
#     for key in list(accs.keys()):
#         guess_accs = accs[key]
#         for i in guess_accs:
#             if "be" in i[0]:
#                 if i[0] not in list(occurrence_dict.keys()):
#                     occurrence_dict[i[0]]["true_mult"] = 0
#                     occurrence_dict[i[0]]["true_one"] = 0
#                     occurrence_dict[i[0]]["false_mult"] = 0
#                     occurrence_dict[i[0]]["false_one"] = 0
#                 occurrence_dict[i[0]][key] += 1
#     keys = list(occurrence_dict.keys())
#     for i in keys:
#         print(i, ":")
#         for j in list(accs.keys()):
#             print(j, ":", occurrence_dict[i][j])
#     print()





#for partition in range(1, 7):
    #for guess in ["never", "sentence", "word"]:
        #handle = get_results_file(partition, guess)
        #context_word_handle = get_results_file(partition, "context_word")
        #context_sense_handle = get_results_file(partition, "context_sense")
        #no_spread_handle = get_results_file(partition, "naive_semantic")

        # print("partition:", partition, ",", guess, "clear Spreading vs Context Sense")
        # sense_counts = get_pairwise_counts(handle, context_sense_handle, 5000, partition, detailed=True)
        # for i in list(sense_counts.keys()):
        # print(i, " : ", sense_counts[i])
        # print()
        # print("partition:", partition, ",", guess, "clear Spreading vs Context Word")
        # word_counts = get_pairwise_counts(handle, context_word_handle, 5000, partition, detailed=True)
        # for i in list(sense_counts.keys()):
        # print(i, " : ", word_counts[i])
        # print()
        # print("partition:", partition, ",", guess, "clear Spreading vs No Spreading")
        # no_spread_counts = get_pairwise_counts(handle, no_spread_handle, 5000, partition, detailed=True)
        # for i in list(no_spread_counts.keys()):
        # print(i, " : ", no_spread_counts[i])
        # print()
