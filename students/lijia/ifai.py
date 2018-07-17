"""ifai is an interface to query possible actions for given sentence from word2vec and probability model"""

import sys
from os.path import dirname, realpath

ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)

from students.lijia.utils import get_manipulable_noun
from students.lijia.extraction import get_verbs_for_noun as prob_get_verbs_for_noun
from students.lijia.word2vec import get_verbs_for_noun as w2v_get_verbs_for_noun
# todo: rename extraction.py


def get_action(sentence):
    verb_dict = {}
    actions = []
    nouns = get_manipulable_noun(sentence)
    print(nouns)
    for noun in nouns:
        prob_verbs = prob_get_verbs_for_noun(noun)
        word2vec_verbs = w2v_get_verbs_for_noun(noun)
        verb_dict[noun] = prob_verbs + word2vec_verbs
        actions.extend(["%s %s" % (verb, noun) for verb in verb_dict[noun]])
    return actions


def main():
    sentence = "She loves the kitten that was thrown away by her Dad. She carve down the cat with paper and knife."
    print(get_action(sentence))

if __name__ == '__main__':
    main()

