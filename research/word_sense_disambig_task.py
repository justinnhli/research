import os
import random
from sentence_long_term_memory import sentenceLTM
from sentence_long_term_memory import SentenceCooccurrenceActivation
from nltk.corpus import wordnet

def run_word_sense_disambig(directory_list, activation_base, decay_parameter, constant_offset):
    sent_list_sense_dict = extract_sentences(directory_list)
    sentence_list = sent_list_sense_dict[0]
    word_sense_dict = sent_list_sense_dict[1]
    #print(sentence_list)
    sem_network = create_sem_network(sentence_list, activation_base, decay_parameter, constant_offset)
    #print(sem_network.knowledge)
    guess_list = naive_predict_word_sense(sem_network, sentence_list, word_sense_dict)
    return guess_list.count(True) / len(guess_list)


def extract_sentences(directory_list):
    """
    Runs the word sense disambiguation task.
    Parameters:
        directory_list (list): A list of directories where the tag files from the semcor corpus can be found.
    Returns:
        list: List of words in each sentence
    """
    sentence_list = []
    word_sense_dict = {}
    for directory in directory_list:
        for corpus_file in os.listdir(directory):
            if (corpus_file != ".DS_Store"):
                f = open(directory + corpus_file, "rt")
                corpus_words = f.readlines()
                sentence_word_list = []
                for line in range(len(corpus_words)):
                    if (corpus_words[line].find("snum=") != -1):
                        if len(sentence_word_list) > 1:
                            if len(set(sentence_word_list)) > 1:
                                sentence_list.append(sentence_word_list)
                        sentence_word_list = []
                    elif corpus_words[line].find("cmd=done") > 0 \
                            and corpus_words[line].find("pos=") > 0 \
                            and corpus_words[line].find("lemma=") > 0 \
                            and corpus_words[line].find("lexsn=") > 0:
                        word_index1 = corpus_words[line].find("lemma=")
                        word_index2 = corpus_words[line].find("wnsn=")
                        word = corpus_words[line][word_index1 + 6: word_index2 - 1]
                        sense_index2 = corpus_words[line].find(" lexsn")
                        word_sense = corpus_words[line][word_index2 + 5: sense_index2]
                        if ~word.isnumeric() and len(wordnet.synsets(word)) != 0:
                            word_senses = word_sense.split(";")
                            for sense in word_senses:
                                sentence_word_list.append((word, int(sense)))
                                if word in word_sense_dict.keys():
                                    if int(sense) not in word_sense_dict[word]:
                                        word_sense_dict[word].append(int(sense))
                                else:
                                    word_sense_dict[word] = [int(sense)]
                f.close()
    return sentence_list, word_sense_dict


def create_sem_network(sentence_list, activation_base, decay_parameter, constant_offset):
    network = sentenceLTM(
        activation_cls=(lambda ltm:
                        SentenceCooccurrenceActivation(
                            ltm,
                            activation_base=activation_base,
                            constant_offset=constant_offset,
                            decay_parameter=decay_parameter
                        )))
    for sentence in sentence_list:
        for word1_index in range(len(sentence)):
            for word2_index in range(word1_index + 1, len(sentence)):

                network.activate_cooccur_pair(sentence[word1_index],
                                              sentence[word2_index])
    return network


def dummy_predict_word_sense(sentence_list):
    """
    Dummy function to predict the word sense of all of the words in a sentence.
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


def naive_predict_word_sense(sem_network, sentence_list, word_sense_dict):
    guess_list = []
    time = 1
    for sentence in sentence_list:
        for word in sentence:
            sense_cooccurrence_dict = {}
            for sense in word_sense_dict[word[0]]:
                for cooccur_word in sentence:
                    if cooccur_word != word:
                        word_guess = (word[0], sense)
                        if sense in sense_cooccurrence_dict:
                            sense_cooccurrence_dict[sense] = sense_cooccurrence_dict[sense] + \
                                                         sem_network.get_cooccurrence(word_guess, cooccur_word)
                        else:
                            sense_cooccurrence_dict[sense] = sem_network.get_cooccurrence(word_guess, cooccur_word)
            word_sense_guess = max(zip(sense_cooccurrence_dict.values(), sense_cooccurrence_dict.keys()))[1]
            for cooccur_word in sentence:
                if cooccur_word != word:
                    sem_network.activate_cooccur_pair((word[0], word_sense_guess), cooccur_word)
            time += 1
            if word_sense_guess == word[1]:
                guess_list.append(True)
            else:
                guess_list.append(False)
    return guess_list

def get_corpus_stats(directory_list):
    sent_list_sense_dict = extract_sentences(directory_list)
    sentence_list = sent_list_sense_dict[0]
    word_sense_dict = sent_list_sense_dict[1]
    absolute_word_counts = {}
    absolute_sense_counts = {}
    word_pair_counts = {}
    sense_pair_counts = {}
    for sentence in sentence_list:
        for word1_index in range(len(sentence)):
            word1 = sentence[word1_index]
            if word1 in absolute_sense_counts:
                absolute_sense_counts[word1] = absolute_sense_counts[word1] + 1
            else:
                absolute_sense_counts[word1] = 1
            if word1[0] in absolute_word_counts:
                absolute_word_counts[word1[0]] = absolute_word_counts[word1[0]] + 1
            else:
                absolute_word_counts[word1[0]] = 1
            for word2_index in range(word1_index + 1, len(sentence)):
                word2 = sentence[word2_index]
                if word1[0] < word2[0] or (word1[0] == word2[0] and word1[1] < word2[1]):
                    sense_key = (word1, word2)
                    word_key = (word1[0], word2[0])
                else:
                    sense_key = (word2, word1)
                    word_key = (word1[0], word2[0])
                if sense_key in sense_pair_counts:
                    sense_pair_counts[sense_key] = sense_pair_counts[sense_key] + 1
                else:
                    sense_pair_counts[sense_key] = 1
                if word_key in word_pair_counts:
                    word_pair_counts[word_key] = word_pair_counts[word_key] + 1
                else:
                    word_pair_counts[word_key] = 1
    return absolute_word_counts, absolute_sense_counts, word_pair_counts, sense_pair_counts







# Testing...
# print(extract_sentences(["/Users/lilygebhart/Downloads/Li_Research_Test_Corpus/"]))

# print(extract_sentences(["/Users/lilygebhart/Downloads/semcor3.0/brown1/tagfiles/",
# "/Users/lilygebhart/Downloads/semcor3.0/brown2/tagfiles/",
# "/Users/lilygebhart/Downloads/semcor3.0/brownv/tagfiles/"])[1])

#print(run_word_sense_disambig(["/Users/lilygebhart/Downloads/Li_Research_Test_Corpus/"], 1, 2, 0, 0.05))

#print(run_word_sense_disambig(["/Users/lilygebhart/Downloads/semcor3.0/brown1/tagfiles/",
#"/Users/lilygebhart/Downloads/semcor3.0/brown2/tagfiles/",
#"/Users/lilygebhart/Downloads/semcor3.0/brownv/tagfiles/"], 3, 2, 0, 0.05))

#print(get_corpus_stats(["/Users/lilygebhart/Downloads/semcor3.0/brown1/tagfiles/",
# "/Users/lilygebhart/Downloads/semcor3.0/brown2/tagfiles/",
# "/Users/lilygebhart/Downloads/semcor3.0/brownv/tagfiles/"]))



