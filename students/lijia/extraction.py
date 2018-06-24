"""extract phrase from folder and calculate p(verb|adj)"""

import sys
import time
import datetime
from collections import Counter, defaultdict, namedtuple
from os.path import dirname, realpath, join as join_path, exists as file_exists
from os import listdir
import spacy
import numpy as np
from nltk.wsd import lesk
from research.knowledge_base import KnowledgeFile, URI
from ifai import wn_is_manipulable_noun, umbel_is_manipulable_noun

# make sure research library code is available
ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)
# STORY_DIRECTORY = join_path(ROOT_DIRECTORY, 'data/fanfic_stories/sub1')
# OUTPUT_DIR = join_path(ROOT_DIRECTORY, 'data/output')
STORY_DIRECTORY = join_path(ROOT_DIRECTORY, 'students/lijia/fantacy')
OUTPUT_DIR = join_path(ROOT_DIRECTORY, 'students/lijia/output_files')
NP_DIR = join_path(OUTPUT_DIR, "np")
VO_DIR = join_path(OUTPUT_DIR, "vo")

# load nlp model
model = 'en_core_web_sm'
nlp = spacy.load(model)

# set up namedtuple for extraction
SVO = namedtuple('SVO', ('subject', 'verb', 'preposition', 'object'))
NP = namedtuple('NP', ['noun', 'adjectives'])


def get_filename_from_folder(dir):
    """read a folder  that yield filename to read from individual file"""
    for filename in listdir(dir):
        if not filename.startswith("."):
            yield filename


def get_nlp_sentence_from_file(dir, filename):
    """separate document to yield tokenized individual sentences"""
    for line in open(join_path(dir, filename), encoding='utf-8'):
        s_ls = line.replace("\"","").split(". ")
        for s in s_ls:
            yield nlp(s)


def is_stop_verb(token):
    return token.pos_ == "VERB" and token.is_stop


def is_subject_noun(token):
    return (token.pos_ == "NOUN" or token.pos_ == "PRON") and token.dep_ == "nsubj"


def is_good_obj(token):
    return token.dep_ == "dobj" and token.lemma_ != "…" and token.tag_ != "WP"


def is_good_verb(token):
    return token.head.pos_ == "VERB" and not is_stop_verb(token.head) and not token.text.startswith('\'')


def is_tool(token):
    return (
        token.pos_ == "NOUN" and
        token.lemma_ != "-PRON-" and
        not token.ent_type_ == "PERSON" and
        (
            wn_is_manipulable_noun(token.lemma_) or
            umbel_is_manipulable_noun(token.lemma_)
        ))


def replace_wsd(doc, token):
    # TODO (Lijia): use the function in extraction
    """return the wsd token of a doc"""
    return lesk(doc.text.split(), str(token))


def check_tool_pattern(doc):
    """extract tool from a few prescript patterns"""
    for token in doc:
        # use NP to V
        if token.lemma_ == "use":
            # extract object
            obj = [child.lemma_ for child in token.head.children if child.dep_ == "dobj"]
            if obj:
                if umbel_is_manipulable_noun(obj[0]) or wn_is_manipulable_noun(obj[0]):
                    for child in token.children:
                        # find "to" that compliment "use"
                        if child.dep_ == "xcomp" and [grandchild for grandchild in child.children if grandchild.text == "to"]:
                            return [token.lemma_, obj, "to", child.lemma_]

        # V NP with NP
        elif token.text == "with" and token.head.pos_ == "VERB":
            obj = [child.lemma_ for child in token.head.children if child.dep_ == "dobj"]
            for child in token.children:
                # find the preposition object and check manipulability
                if child.dep_ == "pobj" and (umbel_is_manipulable_noun(child.lemma_) or wn_is_manipulable_noun(child.lemma_)):
                    # it does not really matter weather object exist or not, the tool is the 'pobj'
                    if obj:
                        return [token.head.lemma_, obj[0], "with", child.lemma_]
                    else:
                        return [token.head.lemma_, "with", child.lemma_]


def extract_sentence_phrase(doc):
    """extract phrases that are SVO or SV with prep and return a list of namedtuple SV(O)

        Keyword arguments:
        doc: processed sentences
    """
    results = []
    for token in doc:
        if not (is_subject_noun(token) and is_good_verb(token.head)):
            continue
        # if the token is likely a person's name, replace it
        if token.ent_type_ == "PERSON" or token.lemma_ == "-PRON-":
            s = "PRON"
        else:
            s = token.lemma_
        v = token.head.lemma_

        # look for direct and indirect objects
        for child in token.head.children:
            # direct objects
            if child.dep_ == "dobj" and is_good_obj(child):
                results.append(SVO(s, v, None, child.lemma_))
            # indirect (proposition) objects
            elif child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        results.append(SVO(s, v, child.lemma_, pobj.lemma_))
        # if the verb has neither direct nor indirect objects
        if not results:
            results.append(SVO(s, v, None, None))
    return results


def extract_vo(doc):
    results = extract_sentence_phrase(doc) # ignore tools by indexing
    return [result for result in results if result.object is not None]


def extract_sentence_np(doc):
    """extract the [adj + noun] from the given doc sentence"""
    results = []
    for token in doc:
        # a = replace_wsd(doc, token)
        if token.pos_ != "ADJ":
            continue
        # identify all subjects being described
        if token.dep_ == "amod" and token.head.pos_ == "NOUN":
            # eg. "The tall and skinny girl..."
            subjects = [token.head.lemma_]
        elif token.dep_ == "acomp" and len([child for child in token.children]) == 0:
            # eg. "The girl is tall and skinny."
            subjects = [
                child.lemma_
                for child in token.head.children
                if child.dep_ == "nsubj"
            ]
        else:
            subjects = []
        adjectives = [token.lemma_]
        adjectives.extend(child.lemma_ for child in token.children if child.dep_ == "conj")

        for subject in subjects:
            results.append(NP(subject, adjectives))
    return results


def test_svo(nlp):
    """adding test cases for extracting svo/sv"""
    TestCase = namedtuple('TestCase', ['sentence', 'phrase'])
    test_cases = [
        TestCase(
            "Ashley snorted earning another chorus of 'yeah's'.",
            ["Ashley", "snort"],
        ),
        TestCase(
            "We're not poor! The slightly taller blonde haired girl known as Olivia replied, her lips pursing in anger at the insult the other girl had thrown at not only her, but her Mom to.",
            ['lip', 'purse in', 'anger'],
        ),
        TestCase(
            "Ashley snorted earning another chorus of 'yeah's'.",
            ['ashley', 'snort'],
        ),
        TestCase(
            "They had been arguing since the start of recess, and what had initially started as a small altercation over a burst ball, had quickly degenerated into a full blow argument.",
            [],
        ),
        TestCase(
            "Rachel Berry thought that she would be upset for a long time after Jesse had broken up with her, breaking his heart.",
            [['-PRON-', 'think'], ['-PRON-', 'break with', '-PRON-']]
        ),
    ]
    for test_case in test_cases:
        message = [
            "Parsing sentence: " + test_case.sentence,
            "    but failed to see expected result: " + str(test_case.phrase),
        ]
        assert test_case.phrase in extract_sentence_phrase(nlp(test_case.sentence)), "\n".join(message)


def test_np(nlp):
    """test cases for extracting [adj + NOUN]"""
    TestCase = namedtuple('TestCase', ['sentence', 'phrase'])
    test_cases = [
        # FIXME fix the testcases to match the new return value of extract_sentence_np
        TestCase(
            "Olivia guessed that even Ashley's parents weren't that rich, they didn't live near the park or have a house that backed onto the forest and that house, well It was the biggest house in Lima, Olivia was sure of it.",
            ['house', 'big'],
        ),
        TestCase(
            "The tall girl is cute and very funny.",
            ["girl", "tall", "cute", "funny"],
        ),
    ]
    for test_case in test_cases:
        message = [
            "Parsing sentence: " + test_case.sentence,
            "    but failed to see expected result: " + str(test_case.phrase),
        ]
        assert test_case.phrase in extract_sentence_np(nlp(test_case.sentence)), "\n".join(message)


def extract_from_directory(directory):
    """extract and write (adj + noun) and (verb + noun) from given dump"""
    for file in get_filename_from_folder(directory):
        print(file)
        file_name_vo = join_path(VO_DIR, file[:-4] + "_vo.txt")
        file_name_np = join_path(NP_DIR, file[:-4] + "_np.txt")

        # start extraction if the file does not exist yet
        if file_exists(file_name_vo) and file_exists(file_name_np):
            continue

        vo_lines = []
        np_lines = []
        for s in get_nlp_sentence_from_file(directory, file):
            for vo in extract_vo(s):
                if vo.preposition is not None:
                    line = "%s_%s\t%s" % (vo.verb, vo.preposition, vo.object)
                else:
                    line = "%s\t%s" % (vo.verb, vo.object)
                vo_lines.append(line)
            # write adj + obj result
            for np in extract_sentence_np(s):
                for adjective in np.adjectives:
                    np_lines.append("%s\t%s" % (adjective, np.noun))
        with open(file_name_vo, 'w', encoding='utf-8') as vo_file:
            vo_file.write('\n'.join(vo_lines))
        with open(file_name_np, 'w', encoding='utf-8') as np_file:
            np_file.write('\n'.join(np_lines))


def calculate_individual_p(dir, output_filename, outer_index, inner_index):
    """
    a generic function that calculate probability and count
    :param dir: the folder that contains extracted words
    :param output_filename
    :param outer_index: default dictionary's key (the condition)
    :param inner_index: counter's key
    :return:
    """
    def write_out_count():
        """ write out counts"""
        with open(join_path(OUTPUT_DIR, "count_" + output_filename), 'w', encoding='utf-8') as f:
            for k in d:
                c = d[k]
                f.write("%s (%s)\n" % (k, sum(c.values())))
                for x in d[k]:
                    f.write("---%s (%s)\n" % (x, c[x]))

    def write_out_prob():
        """write out probability for each pair"""
        with open(join_path(OUTPUT_DIR, "prob_" + output_filename), 'w', encoding='utf-8') as f:
            for k in d:
                c = d[k]
                total = sum(c.values())
                for x in d[k]:
                    f.write("%s %s %s\n" % (k, x, float(c[x] / total)))

    # read from a directory to a nested dictionary
    file_gen = get_filename_from_folder(dir)
    d = defaultdict(Counter)
    i = 0
    for file in file_gen:
        i += 1
        print(file)
        lines = [line.rstrip('\n').replace('\t', ' ') for line in open(join_path(dir, file), 'r', encoding='utf-8')]
        # add to defaultdict with x as key and a counter as value
        # the counter counts y's appearance
        for line in lines:
            print(line)
            x = line.split()[outer_index].replace("-", "")
            y = line.split()[inner_index].replace("-", "")
            if d[x]:
                d[x].update([y])
            else:
                d[x] = Counter([y])

    write_out_count()
    print("Finished writing count. Total file processed: ", i)
    write_out_prob()
    print("Finished writing probability")
    return


def calculate_prob(file, output_filename):
    """"reading from count file to calculate probability"""
    with open(file, 'r', encoding='utf-8') as f, open(join_path(OUTPUT_DIR, output_filename), 'w', encoding='utf-8') as w:
        total = 0
        for line in f.readlines():
            line = line.split(" ")

            if not line[0].startswith("-"):
                a = line[0]
                total = int(line[1].replace('(', '').replace(')',""))
            else:
                b = line[0].replace('-', "")
                count = int(line[1].replace('(', '').replace(')', ""))
                try:
                    w.write("%s %s %s\n" % (a, b, float(count / total)))
                except:
                    pass


def read_to_nested_dict(filename, outer_index, inner_index, value_index):
    """ read from file and store in nested dict
    Arguments:
        outer_index: outer dictionary key index
        inner_index: inner dictionary key index
        value_index: value index
    OUTPUT = {outer_index: { inner_index: value_index }}"""
    d = defaultdict(lambda: defaultdict(float))
    with open(filename, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            ls = l.rstrip("\n").split(" ")
            d[ls[outer_index]][ls[inner_index]] = float(ls[value_index])
    return d


def calculate_stats():
    """output a file with frequency of extraction from previously stored files"""
    calculate_individual_p(VO_DIR, "verb|noun.txt", 1, 0)
    calculate_individual_p(NP_DIR, "noun|adj.txt", 0, 1)
    calculate_individual_p(VO_DIR, "noun|verb.txt", 0, 1)

    return


def calculate_verb_given_adj():
    # read file in to dictioanry

    """ {object : {verb: prob(v|n)}}"""
    d_ov = read_to_nested_dict(join_path(OUTPUT_DIR, 'prob_verb|noun.txt'), 0, 1, 2)

    # {object: {adj: prob(n|a}}
    d_oa = read_to_nested_dict(join_path(OUTPUT_DIR, 'prob_noun|adj.txt'), 1, 0, 2)

    adj_verb = set()
    for o in d_ov:
        for adj, _ in d_oa[o].items():
            adj = adj.strip()
            if not adj:
                continue
            for v, _ in d_ov[o].items():
                v = v.strip()
                if not v: continue
                adj_verb.add((adj, v))
    with open(join_path(OUTPUT_DIR, 'verb_adj_pair.txt'), 'w', encoding='utf-8') as w:
        for adj, v in adj_verb:
            w.write("%s %s\n" % (adj, v))

    # {verb : {obj: prob(n|v)}}
    d_vo = read_to_nested_dict(join_path(OUTPUT_DIR, 'prob_noun|verb.txt'), 0, 1, 2)
    with open(join_path(OUTPUT_DIR, 'prob_verb|adj.txt'), 'w', encoding='utf-8') as f:
        for i, (adj, verb) in enumerate(adj_verb):
            prob = sum(
                # P(V|O) * P(O|A)
                # d_vo[verb][obj] * d_oa[obj][adj]
                d_ov[obj][verb] * d_oa[obj][adj]
                for obj in d_vo[verb]
            )
            if not (0 <= prob <= 1):
                continue
            f.write("%s %s %s\n" % (adj, verb, prob))
            # print(i, adj, verb, prob)


def pipe():
    # extract_from_directory(STORY_DIRECTORY)
    # calculate_stats()
    calculate_verb_given_adj()


def main():
    start = time.time()
    pipe()
    end = time.time()
    print("total time cost %s" % datetime.timedelta(seconds= (end - start)))


if __name__ == '__main__':
    # test_svo(nlp)
    # test_np(nlp)
    main()