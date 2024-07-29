""" Gets the distribution for spreading and cooccurrence on RAT and WSD specific corpora/networks"""
from agent_joint_probability import AgentJointProbabilityCorpus, AgentJointProbabilityNGrams
from corpus_utilities import CorpusUtilities
import csv, json, os
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

def get_distribution(task, type, subtype):
    """ Gets the non-averaged distribution for each trial of the specified task (WSD, RAT), specified mechanism
     (cooccurrence, spreading), and subtype (word/sense, never/word/sentence, NA, SFFAN/SWOWEN/combined)"""
    filename = "/Users/lilygebhart/Documents/GitHub/research/research/distributions/" + str(task) + "_" + str(type) + "_" + str(subtype) + "_non_averaged_distributions.json"
    if not os.path.isfile(filename):
        distributions = []
        if task == "WSD":  # Task is WSD
            corpus_utils = CorpusUtilities(-1, 1)
            agent = AgentJointProbabilityCorpus(context_type=subtype, corpus_utilities=corpus_utils, num_sentences=-1,
                                            partition=1, outside_corpus=False)
            sentence_list = corpus_utils.get_sentence_list()
            word_sense_dict = corpus_utils.get_word_sense_dict()
            if type == "cooccurrence":  # Type is cooccurrence
                len_sents = len(sentence_list)
                counter = 0
                for sent in sentence_list:
                    counter += 1
                    if counter % 50 == 0:
                        print(counter, "out of", len_sents)
                    for word_index in range(len(sent)):
                        word = sent[word_index]
                        word_senses = word_sense_dict[word[0]]
                        dist = agent.get_cooccurrence_distribution(word_index, word_senses, sent, subtype)
                        distributions.append(sorted(list(dist.values())))
            else:  # Type is semantic spreading
                timer = 2
                counter = 0
                len_sents = len(sentence_list)
                for sent in sentence_list:
                    counter += 1
                    if counter % 50 == 0:
                        print(counter, "out of", len_sents)
                    if subtype == "sentence":
                        agent.clear_sem_network(1)
                        timer = 2
                    for word_index in range(len(sent)):
                        word = sent[word_index]
                        word_senses = word_sense_dict[word[0]]
                        dist = agent.get_spreading_distribution(word_senses=word_senses, time=timer)
                        distributions.append(sorted(list(dist.values())))
                        if subtype != "word":
                            max_spread = -float("inf")
                            guesses = []
                            for key in list(dist.keys()):
                                prob = dist[key]
                                if prob > max_spread:
                                    guesses = [key]
                                    max_spread = prob
                                if prob == max_spread:
                                    guesses.append(key)
                            for guess in guesses:
                                agent.spreading_agent.network.store(guess, timer)
                            agent.spreading_agent.network.store(word, timer)
                            timer += 1
        else:  # Task is RAT
            with open('./nltk_english_stopwords', "r") as stopwords_file:
                lines = stopwords_file.readlines()
                stopwords = []
                for l in lines:
                    stopwords.append(l[:-1])
            if subtype is None:
                subtype = "SFFAN"
            agent = AgentJointProbabilityNGrams(stopwords=stopwords, source=subtype, spreading=True)
            rat_file = csv.reader(open('./RAT/RAT_items.txt'))
            next(rat_file)
            for trial in rat_file:
                print("trial:", trial)
                context = [trial[0].upper(), trial[1].upper(), trial[2].upper()]
                answer = trial[3]
                if type == "cooccurrence":  # Type is cooccurrence
                    dist = agent.get_cooccurrence_distribution(context[0], context[1], context[2])
                else:  # Type is semantic spreading
                    dist = agent.get_spreading_distribution(context[0], context[1], context[2])
                distributions.append(sorted(list(dist.values())))
        file = open(filename, 'w')
        json.dump(distributions, file)
        file.close()
    else:
        distributions = json.load(open(filename))
    return distributions

def get_average_distribution(task, type, subtype, num_points):
    filename = "/Users/lilygebhart/Documents/GitHub/research/research/distributions/" + str(task) + "_" + str(
        type) + "_" + str(subtype) + "_averaged_distributions.json"
    if not os.path.isfile(filename):
        dists = get_distribution(task, type, subtype)
        avg_dists = [0] * num_points
        print(avg_dists)
        num_dists = len(dists)
        x_vals = np.linspace(0, 1, num_points)
        counter = 0
        for dist in dists:  # Getting each distribution
            counter += 1
            print(counter)
            if len(dist) == 0:
                num_dists -= 1
                continue
            elif len(dist) == 1:
                dist = [dist[0], dist[0]]
                print(dist)
            dist_x_vals = np.linspace(0, 1, len(dist))
            y_interp = interp1d(dist_x_vals, dist)
            for i in range(num_points):
                # if i == 0:
                #     continue
                # elif i == (num_points - 1):
                #     avg_dists[num_points - 1] += dist[-1]
                #else:
                avg_dists[i] += y_interp(x_vals[i])
        avg_dist = [list(x_vals), [elem/num_dists for elem in avg_dists]]
        print("interpolation", avg_dist[1])
        file = open(filename, 'w')
        json.dump(avg_dist, file)
        file.close()
    else:
        avg_dist = json.load(open(filename))
    return avg_dist


def plot_distributions(task, num_points):
    plt.xlim([0, 1])
    plt.ylim([0, 0.5])
    plt.xlabel("Normalized, Ranked Words")
    plt.ylabel("Average Conditional Probability")
    if task == "WSD":
        plt.title("Average Distributions for Spreading & Co-Occurrence on the WSD")
        cooc_word_dist = get_average_distribution(task, "cooccurrence", "word", num_points)
        cooc_sense_dist = get_average_distribution(task, "cooccurrence", "sense", num_points)
        sem_never_dist = get_average_distribution(task, "spreading", "never", num_points)
        sem_word_dist = get_average_distribution(task, "spreading", "word", num_points)
        sem_sent_dist = get_average_distribution(task, "spreading", "sentence", num_points)
        plt.plot(cooc_word_dist[0][1:-1], cooc_word_dist[1][1:-1], label="Co-occurrence: Context Word",
                color="tab:blue")
        plt.plot(cooc_sense_dist[0][1:-1], cooc_sense_dist[1][1:-1], label="Co-occurrence: Context Sense",
                color="tab:green")
        plt.plot(sem_never_dist[0][1:-1], sem_never_dist[1][1:-1], label="Spreading: Clear Never",
                color="tab:orange")
        plt.plot(sem_sent_dist[0][1:-1], sem_sent_dist[1][1:-1], label="Spreading: Clear Sentence",
                color="tab:red")
        plt.plot(sem_word_dist[0][1:-1], sem_word_dist[1][1:-1], label="Spreading: Clear Word",
                color="tab:pink")
    if task == "RAT":
        plt.title("Average Distributions for Spreading & Co-Occurrence on the RAT")
        cooc_dist = get_average_distribution(task, "cooccurrence", "SFFAN", num_points)
        sem_SFFAN_dist = get_average_distribution(task, "spreading", "SFFAN", num_points)
        #sem_SWOWEN_dist = get_average_distribution(task, "spreading", "SWOWEN", num_points)
        #sem_comb_dist = get_average_distribution(task, "spreading", "combined", num_points)
        plt.plot(cooc_dist[0], cooc_dist[1], label="Co-occurrence", color="tab:blue")
        plt.plot(sem_SFFAN_dist[0], sem_SFFAN_dist[1], label="Spreading: SFFAN", color="tab:orange")
        #plt.plot(sem_SWOWEN_dist[0], sem_SWOWEN_dist[1], label="Spreading: SWOWEN", marker='o', color="tab:pink")
        #plt.plot(sem_comb_dist[0], sem_comb_dist[1], label="Spreading: combined", marker='o', color="tab:red")
    plt.legend(loc="upper left")
    plt.show()

plot_distributions("RAT", 100)

