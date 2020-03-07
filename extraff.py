from collections import namedtuple

import spacy
import nltk
from nltk.corpus import wordnet as wn

SPACY_MODEL = 'en_core_web_sm'
try:
    NLP = spacy.load(SPACY_MODEL)
except OSError:
    spacy.cli.download(SPACY_MODEL)
    NLP = spacy.load(SPACY_MODEL)

nltk.download('wordnet', quiet=True)


def is_good_noun(token):
    """Check if the token is an acceptable noun.

    Parameters:
        token (spacy.Token): The token to check.

    Returns:
        bool: Whether the token contains an acceptable noun.
    """
    return (
        token.pos_ == 'NOUN'
        and wn.morphy(token.text, wn.NOUN) 
        and not has_number(token.text)
    )


def is_good_verb(token):
    """Check if the token is an acceptable verb.

    Parameters:
        token (spacy.Token): The token to check.

    Returns:
        bool: Whether the token contains an acceptable verb.
    """
    exclude_verbs = ['have']
    return (
        token.pos_ == 'VERB' 
        and wn.morphy(token.text, wn.VERB) 
        and not token.is_stop 
        and token.text not in exclude_verbs 
        and not token.text.startswith("'")
    )


RULES = {
    '': (
        '((conj [VERB] (dobj [OBJ])))',
    ),
    'VERB': (
        '([VERB] (prep [PREP] (pobj [OBJ])) (nsubj [SUB]))',
        '([VERB] (nsubj [SUB]))',
        '("use" (dobj [TOOL]) (xcomp [VERB] (dobj [OBJ])))',
    ),
    'NOUN': (
        '([NOUN] (amod [ADJ]))',
    ),
}


class FrozenDict:

    def __init__(self, src_dict):
        self._src_dict = src_dict
        self._hash = hash(tuple(
            (key, value) for key, value in self._src_dict.items()
        ))

    def __len__(self):
        return len(self._src_dict)

    def __contains__(self, key):
        return key in self._src_dict

    def __getitem__(self, key):
        return self._src_dict[key]

    def __eq__(self, other):
        if isinstance(other, dict):
            return self._src_dict == other
        elif isinstance(other, FrozenDict):
            return self._src_dict == other._src_dict
        else:
            return False

    def __hash__(self):
        return self._hash

    def keys(self):
        return self._src_dict.keys()

    def values(self):
        return self._src_dict.values()

    def items(self):
        return self._src_dict.items()


def match_tree(token, rule):

    def _match_tree(token, rule, result):
        paren_count = 0
        index = 0
        for index, char in enumerate(rule):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0:
                    break
        rule = rule[:index + 1]
        index = 1
        while index < len(rule):
            word = rule[index:rule.find(' ', index)]
            if word.startswith('-'):
                # FIXME negations
                pass
            elif word.startswith('"') and word.endswith('"'):
                stem = word[1:-1]
                if wn.morphy(token.text) != stem:
                    return len(rule), None
                index = index + len(word) + 1
            elif word.startswith('[') and word.endswith(']'):
                result[word[1:-1]] = str(token)
                index = index + len(word) + 1
            elif word.startswith('('):
                prev_index = index
                for child in token.children:
                    if child.dep_ == word[1:]:
                        span = _match_tree(child, rule[index:-1], result)
                        index += span
                if prev_index == index:
                    result.clear()
                    return len(rule)
            else:
                index += len(word) + 1
        return len(rule)

    result = {}
    _match_tree(token, rule, result)
    return result


def print_parse_tree(doc, prefix=''):

    def print_token_tree(token, depth=0):
        print(f'{prefix}{depth * "  "} {token.dep_}: {token} ({token.pos_})')
        for child in token.children:
            print_token_tree(child, depth + 1)

    for token in doc:
        if token.dep_ == 'ROOT':
            print_token_tree(token)
            break


def extract_semantics(doc):
    semantics = []
    for token in doc:
        for rules in (RULES.get(''), RULES.get(token.pos_, [])):
            max_length = 0
            rule_semantics = []
            for rule in rules:
                match = match_tree(token, rule)
                if match:
                    if len(match) > max_length:
                        rule_semantics = [match]
                        max_length = len(match)
                    elif len(match) == max_length:
                        rule_semantics.append(match)
            semantics.extend(rule_semantics)
    return semantics


def main():
    sentences = [
        'The knife cut through the rancid meat like hot butter.',
        'She used the knife to free herself, and cut herself in the process.',
        'She used the knife to free herself, and she cut herself in the process.',
    ]
    for sentence in sentences:
        doc = NLP(sentence)
        semantics = extract_semantics(doc)
        print(sentence)
        print_parse_tree(doc)
        for semantic in semantics:
            print(semantic)
        print()


if __name__ == '__main__':
    main()
