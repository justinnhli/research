import random
import nltk
from sentence_long_term_memory import sentenceLTM
from sentence_long_term_memory import SentenceCooccurrenceActivation
from nltk.corpus import wordnet
from nltk.corpus import semcor


def run_wsd(activation_base, decay_parameter, constant_offset):
    """
    Runs the word sense disambiguation task over the Semcor corpus.
    Parameters:
        activation_base (float): A parameter in the activation equation.
        decay_parameter (float): A parameter in the activation equation.
        constant_offset (float): A parameter in the activation equation.
    Returns:
        (float): The raw percent accuracy of the guesses of naive_predict_word_sense.
    """
    sentence_list, word_sense_dict = extract_sentences()
    sem_network = create_sem_network(sentence_list, activation_base, decay_parameter, constant_offset)
    guess_list = naive_predict_word_sense(sem_network, sentence_list, word_sense_dict)
    return guess_list.count(True) / len(guess_list)


def extract_sentences():
    """
    Runs the word sense disambiguation task.
    Parameters:
        directory_list (list): A list of directories where the tag files from the semcor corpus can be found.
    Returns:
        list: List of words in each sentence
    """
    sentence_list = []
    word_sense_dict = {}
    semcor_sents = semcor.tagged_sents(tag="sem")
    sentence_word_list = []
    for sentence in semcor_sents:
        if len(set(sentence_word_list)) > 1:
            sentence_list.append(sentence_word_list)
        sentence_word_list = []
        for word in sentence:
            if isinstance(word, nltk.Tree):
                if isinstance(word.label(), nltk.corpus.reader.wordnet.Lemma):
                    sentence_word_list.append((word.label(), word.label().synset()))
    for sentence in sentence_list:
        for word in sentence:
            if word[0].name() in word_sense_dict.keys():
                curr_synsets = [item[1] for item in word_sense_dict[word[0].name()]]
                if word[0].synset() not in curr_synsets:
                    word_sense_dict[word[0].name()] = word_sense_dict[word[0].name()] + [word]
            else:
                word_sense_dict[word[0].name()] = [word]
    return sentence_list, word_sense_dict


def create_sem_network(sentence_list, activation_base, decay_parameter, constant_offset):
    """
    Builds a semantic network with each word in the Semcor corpus and its corresponding synonyms, hypernyms, hyponyms,
        holonyms, meronyms, attributes, entailments, causes, also_sees, verb_groups, and similar_tos.
    Parameters:
        sentence_list (Nested String List): A list of the sentences in the Semcor corpus with each word represented by
            a tuple: (lemma, lemma synset).
        activation_base (float): A parameter in the activation equation.
        decay_parameter (float): A parameter in the activation equation.
        constant_offset (float): A parameter in the activation equation.
    Returns:
        network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
    """
    network = sentenceLTM(
        activation_cls=(lambda ltm:
                        SentenceCooccurrenceActivation(
                            ltm,
                            activation_base=activation_base,
                            constant_offset=constant_offset,
                            decay_parameter=decay_parameter
                        )))
    semcor_words = sum(sentence_list, [])
    for word in semcor_words:
        if not network.retrievable(word):
            syn = word[1]
            lemma = word[0]
            synonyms = [synon for synon in syn.lemmas() if synon in semcor_words and synon != lemma]
            hypernyms = [hyper for hyper in syn.hypernyms() if hyper in semcor_words and hyper != lemma]
            hyponyms = [hypo for hypo in syn.hyponyms() if hypo in semcor_words and hypo != lemma]
            holonyms = [holo for holo in syn.member_holonyms() + syn.substance_holonyms() + syn.part_holonyms() if
                    holo in semcor_words and holo != lemma]
            meronyms = [mero for mero in syn.member_meronyms() + syn.substance_meronyms() + syn.part_meronyms() if
                    mero in semcor_words and mero != lemma]
            attributes = [attr for attr in syn.attributes() if attr in semcor_words and attr != lemma]
            entailments = [entail for entail in syn.entailments() if entail in semcor_words and entail != lemma]
            causes = [cause for cause in syn.causes() if cause in semcor_words and cause != lemma]
            also_sees = [also_see for also_see in syn.also_sees() if also_see in semcor_words and also_see != lemma]
            verb_groups = [verb for verb in syn.verb_groups() if verb in semcor_words and verb != lemma]
            similar_tos = [sim for sim in syn.similar_tos() if sim in semcor_words and sim != lemma]
            # word_relations = synonyms + hypernyms + hyponyms + holonyms + meronyms + attributes + entailments + causes + also_sees + verb_groups + similar_tos
            network.store(mem_id=word,
                      time=0,
                      synonyms=synonyms,
                      hypernyms=hypernyms,
                      hyponyms=hyponyms,
                      holynyms=holonyms,
                      meronyms=meronyms,
                      attributes=attributes,
                      entailments=entailments,
                      causes=causes,
                      also_sees=also_sees,
                      verb_groups=verb_groups,
                      similar_tos=similar_tos)
    for sentence in sentence_list:
        for index1 in range(len(sentence)):
            for index2 in range(index1 + 1, len(sentence)):
                network.activate_cooccur_pair(sentence[index1], sentence[index2])
    return network


def naive_predict_word_sense(sem_network, sentence_list, word_sense_dict):
    """
    Predicts the correct word sense for a word in a sentence based on how frequently it occurs with the other words in
        the sentence compared to other possible senses of the word.
    Parameters:
        sem_network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
        sentence_list (Nested String List): A list of the sentences in the Semcor corpus with each word represented by
            a tuple: (lemma, lemma synset).
        word_sense_dict (dict): A dictionary with keys for each word in the sentence and values tuples (lemma, lemma synset)
            indicating the senses of the words in the dictionary.
    Returns:
        guess_list (Boolean list): A list for each word in the Semcor corpus indicating whether the word sense was guesssed
            correctly (True) or incorrectly (False).
    """
    guess_list = {}
    time = 1
    for sentence in sentence_list:
        for word in sentence:
            sense_cooccurrence_dict = {}
            for lemma in word_sense_dict[word[0].name()]:
                for cooccur_word in sentence:
                    if cooccur_word != word:
                        if lemma not in sense_cooccurrence_dict.keys():
                            sense_cooccurrence_dict[lemma] = 0
                        sense_cooccurrence_dict[lemma] += sem_network.get_cooccurrence(lemma, cooccur_word)
            word_sense_guess = max(zip(sense_cooccurrence_dict.values(), sense_cooccurrence_dict.keys()))[1]
            # for cooccur_word in sentence:
            # if cooccur_word != word:
            # sem_network.activate_cooccur_pair(word_sense_dict[word_sense_guess][0], cooccur_word)
            time += 1
            if word not in guess_list.keys():
                guess_list[word] = []
            if word_sense_guess == word:
                guess_list[word] += True
            else:
                guess_list[word] += False
    return guess_list


def get_corpus_stats():
    """
    Function to return the absolute counts of each word and each word sense, and the counts of cooccurrence between words
    and word senses
    Returns:
        absolute_word_counts (dict): A dictionary with keys the name of every lemma of the same word and values the
            number of times the word appears in the Semcor corpus.
        absolute_sense_counts (dict): A dictionary with keys a tuple with the lemma and synset of each word (tracking
            the sense of each word) and keys the number of times the tuple appears in the Semcor corpus.
        word_pair_counts (dict): A dictionary with keys a tuple of words appearing together and values the number of times
            the words appear together in the same sentence in the Semcor corpus.
        sense_pair_counts (dict): A dictionary with keys a tuple of two tuples each containing a lemma and its corresponding
            synset (tracking the sense of each word) representing two words of a certain sense occurrring together, and
            values the number of times the senses occur together in the same sentence in the Semcor corpus.
    """
    sentence_list = extract_sentences()[0]
    absolute_word_counts = {}
    absolute_sense_counts = {}
    word_pair_counts = {}
    sense_pair_counts = {}
    for sentence in sentence_list:
        for word1_index in range(len(sentence)):
            word1 = sentence[word1_index]
            if word1 in absolute_sense_counts.keys():
                absolute_sense_counts[word1] = absolute_sense_counts[word1] + 1
            else:
                absolute_sense_counts[word1] = 1
            if word1[0].name() in absolute_word_counts.keys():
                absolute_word_counts[word1[0].name()] = absolute_word_counts[word1[0].name()] + 1
            else:
                absolute_word_counts[word1[0].name()] = 1
            for word2_index in range(word1_index + 1, len(sentence)):
                word2 = sentence[word2_index]
                if word1[0] < word2[0] or (word1[0] == word2[0] and word1[1] <= word2[1]):
                    sense_key = (word1, word2)
                    word_key = (word1[0].name(), word2[0].name())
                else:
                    sense_key = (word2, word1)
                    word_key = (word2[0].name(), word1[0].name())
                if sense_key in sense_pair_counts.keys():
                    sense_pair_counts[sense_key] = sense_pair_counts[sense_key] + 1
                else:
                    sense_pair_counts[sense_key] = 1
                if word_key in word_pair_counts.keys():
                    word_pair_counts[word_key] = word_pair_counts[word_key] + 1
                else:
                    word_pair_counts[word_key] = 1
    return absolute_word_counts, absolute_sense_counts, word_pair_counts, sense_pair_counts


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
        guess_sense = random.randint(1, 3)
        if guess_sense == int(word[2]):
            accuracy_list.append(True)
        else:
            accuracy_list.append(False)
    return accuracy_list


# Testing...
#print(extract_sentences()[1])
#print(create_sem_network(extract_sentences()[0], 2, 0.05, 0))
#print(run_wsd(2, 0.05, 0))
#print(get_corpus_stats())
