import os
import random
from sentence_long_term_memory import sentenceLTM
from sentence_long_term_memory import SentenceCooccurrenceActivation


# FIXME Fix this function!! Probably want to separate it into naive_guess_word() functions and this function...
def run_word_sense_disambig(directory_list, num_trials, activation_base, decay_parameter, constant_offset):
    sent_list_sense_dict = extract_sentences(directory_list)
    sentence_list = sent_list_sense_dict[0]
    word_sense_dict = sent_list_sense_dict[1]
    print(sentence_list)
    sem_network = create_sem_network(sentence_list, activation_base, decay_parameter, constant_offset)
    print(sem_network.knowledge)
    overall_accuracy_list = []
    for trial in range(num_trials):
        for sentence in sentence_list:
            for word in sentence:
                adjusted_sentence = sentence
                adjusted_sentence.remove(word)
                word = word[0: word.rfind(".")]
                print("word = " + word)
                guessed_sense = sem_network.guess_word_sense(word, adjusted_sentence)
                print("guessed sense = "+ guessed_sense)
                print(word[word.rfind(".") + 1: len(word)])
                if guessed_sense == word[word.rfind(".") + 1: len(word)]:
                    overall_accuracy_list.append(True)
                else:
                    overall_accuracy_list.append(False)
    return overall_accuracy_list.count(True)/len(overall_accuracy_list)


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
                f = open(directory+corpus_file, "rt")
                corpus_words = f.readlines()
                sentence_word_list = []
                for line in range(len(corpus_words)):
                    if (corpus_words[line].find("snum=") != -1):
                        if sentence_word_list != []:
                            sentence_list.append(sentence_word_list)
                        sentence_word_list = []
                    elif corpus_words[line].find("cmd=done") > 0 \
                            and corpus_words[line].find("pos=") > 0\
                            and corpus_words[line].find("lemma=") > 0 \
                            and corpus_words[line].find("lexsn=") > 0:
                        word_index1 = corpus_words[line].find("lemma=")
                        word_index2 = corpus_words[line].find("wnsn=")
                        word = corpus_words[line][word_index1 + 6: word_index2-1]
                        sense_index2 = corpus_words[line].find(" lexsn")
                        word_sense = corpus_words[line][word_index2 + 5: sense_index2]
                        if ~word.isnumeric():
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
                decay_parameter=decay_parameter,
    )))
    for sentence in sentence_list:
        for word_index1 in range(len(sentence) - 1):
            network.store(sentence[word_index1], time=0)
            for word_index2 in range(word_index1 + 1, len(sentence)):
                network.store(sentence[word_index2], time=0)
                if sentence[word_index1] > sentence[word_index2]:
                    pair_id = (sentence[word_index1], sentence[word_index2])
                else:
                    pair_id = (sentence[word_index2], sentence[word_index1])
                network.store(mem_id=pair_id,
                              time=0,
                              word_1=sentence[word_index1],
                              word_2=sentence[word_index1])
                # FIXME Change this.
                network.activate_sentence(sentence)
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
        guess_sense = random.randint(1,3)
        if guess_sense == int(word[2]):
            accuracy_list.append(True)
        else:
            accuracy_list.append(False)
    return accuracy_list



# Testing...
#print(extract_sentences(["/Users/lilygebhart/Downloads/Li_Research_Test_Corpus/"]))

print(extract_sentences(["/Users/lilygebhart/Downloads/semcor3.0/brown1/tagfiles/",
                    "/Users/lilygebhart/Downloads/semcor3.0/brown2/tagfiles/",
                    "/Users/lilygebhart/Downloads/semcor3.0/brownv/tagfiles/"])[1])

#run_word_sense_disambig(["/Users/lilygebhart/Downloads/Li_Research_Test_Corpus/"], 1, 2, 0, 0.05)

#run_word_sense_disambig(["/Users/lilygebhart/Downloads/semcor3.0/brown1/tagfiles/",
                    #"/Users/lilygebhart/Downloads/semcor3.0/brown2/tagfiles/",
                    #"/Users/lilygebhart/Downloads/semcor3.0/brownv/tagfiles/"], 1, 2, 0, 0.05)