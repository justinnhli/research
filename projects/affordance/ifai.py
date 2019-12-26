"""ifai is an interface to query possible actions for given sentence from word2vec and probability model"""

import sys
from time import time
from os.path import dirname, realpath, join as join_path

ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)

from research.word_embedding import load_model
from students.lijia.utils import get_manipulable_noun, get_sentence_from_file
from students.lijia.extraction import get_verbs_for_noun as prob_get_verbs_for_noun
from students.lijia.word2vec import get_verbs_for_noun as w2v_get_verbs_for_noun


OUTPUT_DIR = join_path(ROOT_DIRECTORY, 'data/output') # todo: change this
GOOGLE_NEWS_MODEL_PATH = join_path(ROOT_DIRECTORY, 'data/models/GoogleNews-vectors-negative300.bin')
W2V_MODEL = load_model(GOOGLE_NEWS_MODEL_PATH)


def get_action(sentence):
    verb_dict = {}
    actions = []
    nouns = get_manipulable_noun(sentence)
    print(nouns)
    for noun in nouns:
        prob_verbs = prob_get_verbs_for_noun(noun)
        word2vec_verbs = w2v_get_verbs_for_noun(W2V_MODEL, noun)
        verb_dict[noun] = prob_verbs.extend(word2vec_verbs)
        actions.extend(["%s %s" % (verb, noun) for verb in verb_dict[noun]])
    return actions


def temp_output(noun, verb_list):
    output_file = join_path(dirname(OUTPUT_DIR), "prob_model_output.txt")
    with open(output_file, "a+", encoding="utf-8") as f:
        f.write(noun + "\n")
        f.write("\t" + verb_list + "\n")


def output(i, sentence, noun, prob_model_verbs, w2c_verbs, actions):
    output_file = join_path(dirname(OUTPUT_DIR), "action_test.txt")
    with open(output_file, "a+", encoding="utf-8") as f:
        if i == 0:
            f.write(sentence + "\n")
        f.write("\t" + noun)
        f.write("\t\tprobability model output: %s\n" % prob_model_verbs)
        f.write("\t\tword embedding output: %s\n" % w2c_verbs)
        f.write("actions: %s\n" % str(actions))


def main():
    start = time()
    for sentence in get_sentence_from_file(dirname(OUTPUT_DIR), "test_sentences.txt"): # todo: change this
        get_action(sentence)
    end = time()
    print("total use time %s s" % (end - start))

if __name__ == '__main__':
    main()
