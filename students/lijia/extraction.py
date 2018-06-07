import spacy
from collections import namedtuple
from os.path import dirname, realpath, join as join_path
from os import listdir

STORY_DIRECTORY = './fanfic_stories'


def separate_sentence(story_file):
    """separate document to return a list of individual sentences"""
    story_path = join_path(STORY_DIRECTORY, story_file)
    ls = []
    for line in open(story_path):
        ls.extend([s + "." for s in line.replace("\"","").split(". ")])
    return ls


def is_stop_verb(token):
    if token.pos_ == "VERB" and token.is_stop:
        return True
    else:
        return False


def is_subject_noun(token):
    if (token.pos_ == "NOUN" or token.pos_ == "PROPN") and token.dep_ == "nsubj":
        return True
    else:
        return False


def extract_sentence_phrase(doc):
    """extract phrases that are SVO or SV

        Argument:
        doc: processed sentences
    """

    # iterate through each word of the sentence
    results = []
    for token in doc:
        # token.tag_ != "WP" and
        if is_subject_noun(token) and \
                        token.head.pos_ == "VERB" and not is_stop_verb(token.head):
            if is_stop_verb(token.head):
                print(token.head.lemma_)

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
                        sentence_results.append([s, v + " " + child.lemma_, str(pobj)])

            # if the verb has neither direct nor indirect objects
            if not sentence_results:
                sentence_results.append([s, v])
            results.extend(sentence_results)

    return results


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

def extract_phrases(path):
    """take in the path and return a list of phrases"""


def test_svo():
    """adding test cases for extracting svo/sv"""
    model = 'en_core_web_sm'
    nlp = spacy.load(model)
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
        )
    ]
    for test_case in test_cases:
        message = [
            "Parsing sentence: " + test_case.sentence,
            "    but failed to see expected result: " + str(test_case.phrase),
        ]
        assert test_case.phrase in extract_sentence_phrase(nlp(test_case.sentence)), "\n".join(message)


def test_np():
    """test cases for extracting [adj + NOUN]"""
    model = 'en_core_web_sm'
    nlp = spacy.load(model)
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
            print("extracted phrase", extract_sentence_phrase(nlp(sentence)))
            print("extracted noun phrase", extract_sentence_np(nlp(sentence)))
            print("extracted prp + noun", extract_pobj(nlp(sentence)))
            print()

if __name__ == '__main__':
    main()