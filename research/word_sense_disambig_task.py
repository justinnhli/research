import os
import random

def run_word_sense_disambig(directory_list):
    """
    Runs the word sense disambiguation task.
    Parameters:
        directory_list (list): A list of directories where the tag files from the semcor corpus can be found.
    Returns:
        float: The accuracy of the word sense disambiguation task for correctly determining the sense of a word.
    """
    accuracy_list = []
    for directory in directory_list:
        for corpus_file in os.listdir(directory):
            f = open(directory+corpus_file, "rt")
            corpus_words = f.readlines()
            sentence_list = []
            for line in range(len(corpus_words)):
                if (corpus_words[line].find("snum=")>0):
                    accuracy_list = accuracy_list + predict_word_sense(sentence_list)
                    sentence_list.clear()
                elif corpus_words[line].find("cmd=done") > 0 \
                        and corpus_words[line].find("pos=") > 0\
                        and corpus_words[line].find("lemma=") > 0 \
                        and corpus_words[line].find("lexsn=") > 0:
                    pos_index1 = corpus_words[line].find("pos=")
                    pos_index2 = corpus_words[line].find("lemma=")
                    pos = corpus_words[line][pos_index1 + 4: pos_index2-1]
                    word_index2 = corpus_words[line].find("wnsn=")
                    word = corpus_words[line][pos_index2 + 6: word_index2-1]
                    sense_index2 = corpus_words[line].find(" lexsn")
                    word_sense = corpus_words[line][word_index2 + 5: sense_index2]
                    if ~word.isnumeric() & len(word_sense)==1:
                        sentence_list.append([word, pos, word_sense])
                    #print("pos = " + pos + " word = " + word + " word_sense = " + word_sense)
            f.close()
    return accuracy_list.count(True)/ len(accuracy_list)



def predict_word_sense(sentence_list):
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
#print(run_word_sense_disambig(["/Users/lilygebhart/Downloads/semcor3.0/brown1/tagfiles/",
                    #"/Users/lilygebhart/Downloads/semcor3.0/brown2/tagfiles/",
                    #"/Users/lilygebhart/Downloads/semcor3.0/brownv/tagfiles/"]))