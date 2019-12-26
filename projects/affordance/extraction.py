"""extract phrase from folder and calculate p(verb_adj)"""

import sys
from time import time
from os import mkdir
from os.path import exists as file_exists, dirname, realpath
from collections import namedtuple
from students.lijia.dumpstats import DumpStats
from students.lijia.utils import *

# make sure research library code is available
ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)

VPO = namedtuple('VPO', ('verb', 'prep', 'object'))
NP = namedtuple('NP', ['noun', 'adjectives'])


def extract_vpo(doc):
    """extract verbs with prep and tangible objects

    :param doc: tokenized sentence
    :return: a list of namedtuple VPO
    """
    results = []
    for token in doc:
        if not is_good_verb(token):
            continue
        # look for direct and indirect objects
        for child in token.children:
            # direct objects
            if is_good_obj(child) and is_manipulable(child):
                results.append(VPO(verb=token.lemma_, prep=None, object=child.lemma_))
            # indirect (proposition) objects
            elif child.dep_ == "prep":
                objects = []
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj" and is_manipulable(grandchild):
                        objects.append(grandchild)
                        objects.extend([ggrandchild for ggrandchild in grandchild.children if ggrandchild.dep_ == "conj"])
                for obj in objects:
                    results.append(VPO(verb=token.lemma_, prep=child.lemma_, object=obj.lemma_))
        # if the verb has neither direct nor indirect objects
        if not results:
            continue
    return results


def extract_np(doc):
    """extract adjectives and tangible nouns

    :param doc: tokenized sentence
    :return: a list of namedtuple NP
    """
    results = []
    for token in doc:
        if not is_good_adj(token):
            continue

        # identify all subjects being described
        if token.dep_ == "amod" and is_manipulable(token.head):
            # eg. "The tall and skinny girl..."
            subjects = [token.head.lemma_]
        elif token.dep_ == "acomp":
            # eg. "The girl is tall and skinny."
            subjects = [
                child.lemma_
                for child in token.head.children
                if child.dep_ == "nsubj" and is_manipulable(child)
            ]
        else:
            subjects = []

        adjectives = [token.lemma_]
        adjectives.extend(child.lemma_ for child in token.children if child.dep_ == "conj")

        for subject in subjects:
            results.append(NP(subject, adjectives))
    return results


def extract_from_folder(dump_dir, stats_dir):
    extract_adj_noun_dir = join_path(stats_dir, "adj_noun")
    extract_verb_noun_dir = join_path(stats_dir, "verb_noun")
    try:
        mkdir(extract_adj_noun_dir)
        mkdir(extract_verb_noun_dir)
    except FileExistsError:
        pass

    for file in get_filename_from_folder(dump_dir):
        print(file)
        file_name_vo = join_path(extract_verb_noun_dir, file[:-4] + "_vo.txt")
        file_name_np = join_path(extract_adj_noun_dir, file[:-4] + "_np.txt")

        # start extraction if the file does not exist yet
        if file_exists(file_name_vo) and file_exists(file_name_np):
            continue

        vo_lines = []
        np_lines = []
        for doc in get_nlp_sentence_from_file(dump_dir, file):
            for vo in extract_vpo(doc):
                try:
                    assert vo.verb != 'takes', "takes output occurs in sentence %s" % doc
                except AssertionError:
                    pass
                if vo.prep is not None:
                    line = "%s_%s %s" % (vo.verb, vo.prep, vo.object)
                else:
                    line = "%s %s" % (vo.verb, vo.object)
                vo_lines.append(line)
            for np in extract_np(doc):
                for adjective in np.adjectives:
                    np_lines.append("%s %s" % (adjective, np.noun))
        with open(file_name_vo, 'w', encoding='utf-8') as vo_file:
            vo_file.write('\n'.join(vo_lines))
        with open(file_name_np, 'w', encoding='utf-8') as np_file:
            np_file.write('\n'.join(np_lines))
    print("finish extracting")


def get_verbs_for_noun(noun):
    def get_verbs_from_dump(dump_stats, noun):
        """calculate overall probability of verb given noun"""

        dict_adj_noun = dump_stats.prob_adj_noun_db.get_given_dict(noun)
        results = {}
        # p1 is P(adj|noun)
        for adj, p1 in dict_adj_noun.items():
            dict_verb_adj = dump_stats.prob_verb_adj_db.get_given_dict(adj)
            # p2 is P(verb|adj)
            for verb, p2 in dict_verb_adj.items():
                if verb in results:
                    results[verb] += p1 * p2
                else:
                    results[verb] = p1 * p2
        return sorted(results.items(), key=(lambda kv: kv[1]), reverse=True)[:100]

    dump_dir = join_path(ROOT_DIRECTORY, "data/temp_test/dump")  # todo: change this
    stats_dir = join_path(ROOT_DIRECTORY, "data/temp_test/stats")  # todo: change this
    extract_from_folder(dump_dir, stats_dir)
    dump_stats = DumpStats(dump_dir, stats_dir)
    return get_verbs_from_dump(dump_stats, noun)
