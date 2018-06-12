import sys
from os.path import dirname, realpath, join as join_path
from os import listdir
from collections import namedtuple
import spacy
import numpy as np

from nltk.corpus import wordnet as wn
from PyDictionary import PyDictionary
from research.knowledge_base import KnowledgeFile, URI

# make sure research library code is available
ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)
STORY_DIRECTORY = './fanfic_stories'
UMBEL_KB_PATH = join_path(ROOT_DIRECTORY, 'data/kbs/umbel-concepts-typology.rdfsqlite')

UMBEL = KnowledgeFile(UMBEL_KB_PATH)
DICTIONARY = PyDictionary()


def separate_sentence(story_file):
    """separate document to return a list of individual sentences"""
    story_path = join_path(STORY_DIRECTORY, story_file)
    ls = []
    for line in open(story_path):
        ls.extend([s for s in line.replace("\"","").split(". ")])
    return ls


def is_stop_verb(token):
    return True if token.pos_ == "VERB" and token.is_stop else False


def is_subject_noun(token):
    return True if (token.pos_ == "NOUN" or token.pos_ == "PRON") and token.dep_ == "nsubj" else False


def is_good_verb(token):
    return True if token.head.pos_ == "VERB" and not is_stop_verb(token.head) \
                   and not token.text.startswith('\'') else False


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
    """extract phrases that are SVO or SV

        Argument:
        doc: processed sentences
    """

    # iterate through each word of the sentence
    results = []
    tools = []
    for token in doc:
        # token.tag_ != "WP" and
        if is_subject_noun(token) and is_good_verb(token.head):
            # extract tool
            if token.lemma_ != "-PRON-":
                tools.append(token.lemma_ if wn_is_manipulable_noun(token.lemma_) or umbel_is_manipulable_noun(token.lemma_) else None)

            # extract np
            sentence_results = []

            # if the token is likely a person's name, replace it
            if token.ent_type_ == "PERSON":
                s = "-PRON-"
            else:
                s = token.lemma_
            v = token.head.lemma_

            for child in token.head.children:
                # direct objects
                if child.dep_ == "dobj":
                    sentence_results.append([s, v, child.lemma_])
                # indirect (proposition) objects
                elif child.dep_ == "prep":
                    for pobj in child.children:
                        sentence_results.append([s, v + " " + child.lemma_, str(pobj.lemma_)])

            # if the verb has neither direct nor indirect objects
            if not sentence_results:
                sentence_results.append([s, v])
            results.extend(sentence_results)

    return results, tools

def wn_is_manipulable_noun(noun):

    def get_all_hypernyms(root_synset):
        hypernyms = set()
        queue = [root_synset]
        while queue:
            synset = queue.pop(0)
            new_hypernyms = synset.hypernyms()
            for hypernym in new_hypernyms:
                if hypernym.name() not in hypernyms:
                    hypernyms.add(hypernym.name())
                    queue.append(hypernym)
        return hypernyms

    for synset in wn.synsets(noun, pos=wn.NOUN):
        if 'physical_entity.n.01' in get_all_hypernyms(synset):
            return True
    return False


def w2v_rank_manipulability(model, nouns):
    """Rank nouns from most manipulable to least manipulable.

    Arguments:
        model (VectorModel): Gensim word vector model.
        nouns (list[str]): The nouns to rank.

    Returns:
        list[tuple[str, float]]: The list of nouns and cosine distances.
    """
    # anchor x_axis by using forest & tree vector difference
    manipulable_basis = model.word_vec('forest') - model.word_vec('tree')
    # map the noun's vectors to the x_axis and spit out a list from small to big
    word_cosine_list = [tuple([noun, np.dot(model.word_vec(noun), manipulable_basis)]) for noun in set(nouns)]

    return sorted(word_cosine_list, key=(lambda kv: kv[1]))



def extract_sentence_np(doc):
    """extract the [adj + noun] from the given doc sentence"""
    results = []
    for token in doc:
        sentence_results = []

        # testing
        # if token.pos_ == "ADJ":
            # print(token)

        # attributive adjective
        if token.dep_ == "amod" and token.pos_ == "ADJ":

            # Example: "Two cute girls no more than eight years old stood in the centre of their friends"
            # should result in "funny" and "eight years old"
            # if the children is amod and adj and does not have any children
            # not a good idea b/c "the eight year old girl is cute and very funny."
            # if not [child for child in token.children if child.dep_ != "advmod"]:

            if token.head.pos_ == "NOUN":
                sentence_results.append([token.lemma_, token.head.lemma_])

            for child in token.children:
                if child.dep_ == "conj":
                    sentence_results.append([child.lemma_, token.head.lemma_])

        # predicative adjective
        elif token.dep_ == "acomp" \
                and not [child for child in token.children]:
            # to fight against counter example: "Olivia was sure of it." --> sure olivia

            for child in token.head.children:
                if child.dep_ == "nsubj":
                    sentence_results.append([token.lemma_, child.lemma_])
        results.extend(sentence_results)
    return results


def extract_pobj(doc):
    """extract prep + object from the given doc sentence"""
    results = []
    for token in doc:
        if token.dep_ == "prep":
            for child in token.children:
                if child.dep_ == "pobj":
                    results.append([token.lemma_, child.lemma_])
    return results


def get_synonyms(word, pos=None):
    """Get synonyms for a word using PyDictionary and WordNet.

    Arguments:
        word (str): The word to find synonyms for.
        pos (int): WordNet part-of-speech constant. Defaults to None.

    Returns:
        set[str]: The set of synonyms.
    """
    syn_list = []

    # add WordNet synonyms to the list
    for synset in wn.synsets(word, pos):
        for lemma in synset.lemmas():
            syn = lemma.name()
            if syn != word:
                syn_list.append(syn)
    # add thesaurus synonyms
    dict_syns = DICTIONARY.synonym(word)
    # combine them and return
    if dict_syns:
        return set(syn_list) | set(dict_syns)
    else:
        return set(syn_list)


def umbel_is_manipulable_noun(noun):

    def get_all_superclasses(kb, concept):
        superclasses = set()
        queue = [str(URI(concept, 'umbel-rc'))]
        query_template = 'SELECT ?parent WHERE {{ {child} {relation} ?parent . }}'
        while queue:
            child = queue.pop(0)
            query = query_template.format(child=child, relation=URI('subClassOf', 'rdfs'))
            for bindings in kb.query_sparql(query):
                parent = str(bindings['parent'])
                if parent not in superclasses:
                    superclasses.add(parent)
                    queue.append(str(URI(parent)))
        return superclasses

    # create superclass to check against
    solid_tangible_thing = URI('SolidTangibleThing', 'umbel-rc').uri
    for synonym in get_synonyms(noun, wn.NOUN):
        # find the corresponding concept
        variations = [synonym, synonym.lower(), synonym.title()]
        variations = [variation.replace(' ', '') for variation in variations]
        # find all ancestors of all variations
        for variation in variations:
            if solid_tangible_thing in get_all_superclasses(UMBEL, variation):
                return True
    return False



def test_svo(model):
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
        TestCase(
            "She wasn't sure how long she could keep the visage up with them around; she needed practice with her peers first.",
            [] # todo: fill this up
        )
    ]
    for test_case in test_cases:
        message = [
            "Parsing sentence: " + test_case.sentence,
            "    but failed to see expected result: " + str(test_case.phrase),
        ]
        assert test_case.phrase in extract_sentence_phrase(nlp(test_case.sentence)), "\n".join(message)


def test_np(model):
    """test cases for extracting [adj + NOUN]"""
    TestCase = namedtuple('TestCase', ['sentence', 'phrase'])
    test_cases = [
        TestCase(
            "Olivia guessed that even Ashley's parents weren't that rich, they didn't live near the park or have a house that backed onto the forest and that house, well It was the biggest house in Lima, Olivia was sure of it.",
            ['big', 'house'],
        ),
    ]
    for test_case in test_cases:
        message = [
            "Parsing sentence: " + test_case.sentence,
            "    but failed to see expected result: " + str(test_case.phrase),
        ]
        assert test_case.phrase in extract_sentence_phrase(nlp(test_case.sentence)), "\n".join(message)


def main():
    # load nlp model
    model = 'en_core_web_sm'
    nlp = spacy.load(model)

    story_file = [filename for filename in listdir(STORY_DIRECTORY) if not filename.startswith('.')]

    for file in story_file:
        ls = separate_sentence(file)

        for sentence in ls:
            print(sentence)
            print("extracted noun phrase", extract_sentence_np(nlp(sentence)))
            print("extracted prep + noun", extract_pobj(nlp(sentence)))
            print("extracted tool", extract_sentence_phrase(nlp(sentence))[1])
            print("extracted phrase", extract_sentence_phrase(nlp(sentence))[0])

            # check patterns
            i = check_tool_pattern(doc)
            print(i)
            if i is not None: tools.append(" ".join(i))

            print()
    print("extracted tool phrase", tools)

if __name__ == '__main__':
    main()