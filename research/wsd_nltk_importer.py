import random
from collections import defaultdict
from sentence_long_term_memory import sentenceLTM
from sentence_long_term_memory import SentenceCooccurrenceActivation
import nltk
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn_corpus
from nltk.stem import wordnet as wn
import pandas as pd
import json
import os.path


def extract_sentences(num_sentences=-1):
    """
    Runs the word sense disambiguation task.
    Parameters:
        num_sentences (int): The number of sentences from the corpus to use in the task. The first n sentences
            from the corpus are used and if n=-1, all sentences from the corpus are used.
    Returns:
        list: sentence_list (list of all sentences or the first n sentences of the corpus), word_sense_dict (dictionary with the possible senses of
            each word in the corpus)
    """
    if not (os.path.isfile("./sentence_list.json") or os.path.isfile("./word_sense_dict.json")):
        # Checking that file exists
        sentence_list = []
        word_sense_dict = defaultdict(set)
        if num_sentences == -1:
            semcor_sents = semcor.tagged_sents(tag="sem")
        else:
            semcor_sents = semcor.tagged_sents(tag="sem")[0:num_sentences]
        for sentence in semcor_sents:
            temp_word_sense_dict = defaultdict(set)
            sentence_word_list = []
            for item in sentence:
                if not isinstance(item, nltk.Tree):
                    continue
                if not isinstance(item.label(), nltk.corpus.reader.wordnet.Lemma):
                    continue
                corpus_word = lemma_to_tuple(item.label())
                sentence_word_list.append(corpus_word)
                temp_word_sense_dict[corpus_word[0]].add(corpus_word)
            if len(temp_word_sense_dict) > 1:
                for word, senses in temp_word_sense_dict.items():
                    word_sense_dict[word] = set(word_sense_dict[word])
                    word_sense_dict[word] |= senses
                    word_sense_dict[word] = list(word_sense_dict[word])
                sentence_list.append(sentence_word_list)
        sent_list_file = open("./sentence_list.json", 'w')
        json.dump(sentence_list, sent_list_file)
        sent_list_file.close()
        word_sense_dict_file = open("./word_sense_dict.json", 'w')
        # Making word sense list from word sense dict
        word_sense_list = []
        for word in word_sense_dict.keys():
            temp_sense_list = []
            temp_sense_list.append(word)
            temp_val_list = []
            for val in word_sense_dict[word]:
                temp_val_list.append(list(val))
            temp_sense_list.append(temp_val_list)
            word_sense_list.append(temp_sense_list)
        json.dump(word_sense_list, word_sense_dict_file, indent=4)
        word_sense_dict_file.close()
    else:
        # Getting json file containing the sentence list and converting the words stored as strings into tuples
        sentence_list = json.load(open("./sentence_list.json"))
        for sentence_index in range(len(sentence_list)):
            for word_index in range(len(sentence_list[sentence_index])):
                word = sentence_list[sentence_index][word_index]
                sentence_list[sentence_index][word_index] = tuple(word)
        # Getting json file containing word sense dict and converting words stored (values) as strings into tuples
        word_sense_list = json.load(open("./word_sense_dict.json"))
        word_sense_dict = defaultdict(set)
        for pair_index in range(len(word_sense_list)):
            key = word_sense_list[pair_index][0]
            vals = word_sense_list[pair_index][1]
            for val in vals:
                word_sense_dict[key].add(tuple(val))
            #sense_values = word_sense_dict[key]
            #tuple_vals = []
            #for val in sense_values:
                #tuple_vals.append(lemmastring_to_tuple(val))
            #word_sense_dict[key] = tuple_vals
    return sentence_list, word_sense_dict



def get_semantic_relations_dict(sentence_list, inside_corpus=True):
    """
    Note: will have to make more edits to the function before inside_corpus = False is accurate, since entries will need
        to be made for all words that have links to them.
    """
    if not os.path.isfile("./semantic_relations_dict.json"):
        semantic_relations_dict = defaultdict(set)
        semcor_words = set(sum(sentence_list, []))
        counter = 0
        for sentence in sentence_list:
            counter += 1
            print(str(counter)+" out of "+str(len(sentence_list)))
            for word in sentence:
                word_string = tuple_to_lemmastring(word)
                if word_string not in semantic_relations_dict.keys():
                    syn = wn_corpus.synset(word[1])
                    lemma = word[0]
                    #synonyms = [(synon, syn) for synon in syn.lemmas() if (synon, syn) in semcor_words and synon != lemma]
                    synonyms = [lemma_to_tuple(synon) for synon in syn.lemmas() if lemma_to_tuple(synon) in
                                semcor_words and lemma_to_tuple(synon) != lemma]
                    # These are all synsets.
                    synset_relations = [syn.hypernyms(), syn.hyponyms(),
                                        syn.member_holonyms() + syn.substance_holonyms() + syn.part_holonyms(),
                                        syn.member_meronyms() + syn.substance_meronyms() + syn.part_meronyms(),
                                        syn.attributes(), syn.entailments(), syn.causes(), syn.also_sees(),
                                        syn.verb_groups(), syn.similar_tos()]
                    lemma_relations = []
                    for ii in range(len(synset_relations)):
                        lemma_relations.append([])
                        # Getting each of the synsets above in synset_relations.
                        for jj in range(len(synset_relations[ii])):
                            # Getting the lemmas in each of the synset_relations synsets.
                            syn_lemmas = synset_relations[ii][jj].lemmas()
                            # Adding each lemma to the list
                            for lemma in syn_lemmas:
                                lemma_relations[ii].append((lemma, synset_relations[ii][jj]))
                    if inside_corpus:
                        for ii in range(len(lemma_relations)):
                            #lemma_relations[ii] = [word_tuple for word_tuple in set(lemma_relations[ii]) if word_tuple
                                                #in semcor_words and word_tuple != word]
                            lemma_relations[ii] = [lemma_to_tuple(word_tuple[0]) for word_tuple in set(lemma_relations[ii])
                                                   if lemma_to_tuple(word_tuple[0]) in semcor_words and
                                                   lemma_to_tuple(word_tuple[0]) != word]
                    word_sem_rel_subdict = create_word_sem_rel_dict(synonyms=synonyms,
                                                                    hypernyms=lemma_relations[0],
                                                                    hyponyms=lemma_relations[1],
                                                                    holonyms=lemma_relations[2],
                                                                    meronyms=lemma_relations[3],
                                                                    attributes=lemma_relations[4],
                                                                    entailments=lemma_relations[5],
                                                                    causes=lemma_relations[6],
                                                                    also_sees=lemma_relations[7],
                                                                    verb_groups=lemma_relations[8],
                                                                    similar_tos=lemma_relations[9])
                    semantic_relations_dict[word] = word_sem_rel_subdict
        sem_rel_file = open("./semantic_relations_dict.json", 'w')
        json.dump(semantic_relations_dict, sem_rel_file)
        sem_rel_file.close()
    semantic_relations_dict = json.load(open("./semantic_relations_dict.json"))
    return semantic_relations_dict


# Helper Functions ----------------------------------------------------------------------------------------------------
def lemma_to_lemmastring(lemma):
    """
    Takes in lemma object
    Converts Lemma object to string tuple compatible with all python versions
    """
    if isinstance(lemma, nltk.corpus.reader.wordnet.Lemma):
        lemma_word = lemma.name()
        synset_string = lemma.synset().name()
        lemma_string = lemma_word + " " + synset_string
        return lemma_string
    else:
        raise ValueError("Lemma object could not be converted into string. Check Lemma type.")

def lemma_to_tuple(lemma):
    lemma_word = lemma.name()
    synset_string = lemma.synset().name()
    lemma_tuple = (lemma_word, synset_string)
    return lemma_tuple

def lemmastring_to_wordstring(lemmastring):
    return lemmastring[:lemmastring.find(' ')]


def lemmastring_to_tuple(word_string):
    word_tuple = (word_string[:word_string.find(' ')], word_string[word_string.find(' ') + 1:])
    return word_tuple


def tuple_to_lemmastring(word_tuple):
    lemma_string = word_tuple[0] + " " + word_tuple[1]
    return lemma_string


def create_word_sem_rel_dict(synonyms, hypernyms, hyponyms, holonyms, meronyms, attributes,
                                   entailments, causes, also_sees, verb_groups, similar_tos):
    """
    Creates an empty semantic relations dictionary with given semantic relations for a word.
    Also converts tuples into lemmastrings for storage in json file.
    """
    sem_rel_dict = {"synonyms": set(synonyms), "hypernyms": set(hypernyms), "hyponyms": set(hyponyms),
                    "holonyms": set(holonyms), "meronyms": set(meronyms), "attributes": set(attributes),
                    "entailments": set(entailments), "causes": set(causes), "also_sees": set(also_sees),
                    "verb_groups": set(verb_groups), "similar_tos": set(similar_tos)}
    for rel in sem_rel_dict.keys():
        vals = sem_rel_dict[rel]
        string_vals = []
        for val in vals:
            string_vals.append(tuple_to_lemmastring(val))
        sem_rel_dict[rel] = string_vals
    return sem_rel_dict

# Testing---------------------------------------------------------------------------------------------------------------

sentence_list, word_sense_dict = extract_sentences()
sem_relations_dict = get_semantic_relations_dict(sentence_list)
