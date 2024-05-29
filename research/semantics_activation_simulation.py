from wsd_stats import *
import math

# Getting frequencies of certain words in the corpus.
for partition in [1, 4]:
    print("Partition:", partition)
    sent_list = extract_sentences(num_sentences=5000, partition=partition)[0]
    sem_rels = get_semantic_relations_dict(sent_list, partition=partition, outside_corpus=False)
    exp_keys = []
    for i in list(sem_rels.keys()):
        if 'be' == i[0]:
            exp_keys.append(i)
    #Gets all words that have "to be"
    secondary_connections = defaultdict(list)
    for word in exp_keys:
        relations = sem_rels[word]
        word_rels = set()
        for rel_type in list(relations.keys()):
            word_rels.update(relations[rel_type])
        secondary_connections[word] = list(word_rels)
        for rel in secondary_connections[word]:
            if rel not in list(secondary_connections.keys()):
                word_rels = set()
                for rel_type in list(relations.keys()):
                    word_rels.update(relations[rel_type])
                secondary_connections[rel] = list(word_rels)
    activations_dict = defaultdict(list)
    counter = 1
    for sent in sent_list:
        for word in sent:
            if word in list(secondary_connections.keys()):
                activations_dict[word].append([counter, 0])
                for con in secondary_connections[word]:
                    activations_dict[con].append([counter, 1])
            if counter % 100 == 0:
                act_checks = defaultdict(int)
                for elem in sorted(list(activations_dict.keys())):
                    curr_time = counter
                    acts = activations_dict[elem]
                    temp_act = 0
                    for act in acts:
                        graph_dist = act[1]
                        act_time = act[0]
                        if curr_time != act_time:
                            temp_act += (0.5**graph_dist)*((curr_time - act_time)**(-0.05))
                    act_checks[elem] = math.log(temp_act)
                for i in list(act_checks.keys()):
                    print("Time:", curr_time, "Elem:", i, ":", act_checks[i])
                print()
            counter += 1
    # Note: time ticks up at each word
    #Gets all of the primary connections to words related to "to be" words



