import random
import nltk
from sentence_long_term_memory import sentenceLTM
from sentence_long_term_memory import SentenceCooccurrenceActivation
from nltk.corpus import wordnet
from nltk.corpus import semcor


def run_wsd(activation_base=2, decay_parameter=0.05, constant_offset=0, predict_word_sense="naive", iterations=1):
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
    if predict_word_sense == "naive":
        guess_list = naive_predict_word_sense(sentence_list, word_sense_dict)
    elif predict_word_sense == "frequency":
        guess_list = frequency_predict_word_sense(sentence_list, word_sense_dict)
    elif predict_word_sense == "senseless":
        #sem_network = create_sem_network(sentence_list, spreading=False, activation_base=activation_base,
                                         #decay_parameter=decay_parameter, constant_offset=constant_offset)
        guess_list = senseless_predict_word_sense(sentence_list, word_sense_dict)
    elif predict_word_sense == "naive_sem":
        sem_network = create_sem_network(sentence_list, spreading=False, time=True, activation_base=activation_base,
                                         decay_parameter=decay_parameter, constant_offset=constant_offset)
        guess_list = naive_semantic_predict_word_sense(sem_network[0], sentence_list, word_sense_dict, iterations,
                                                       sem_network[1])
    elif predict_word_sense == "naive_sem_spreading":
        sem_network = create_sem_network(sentence_list, spreading=True, time=True, activation_base=activation_base,
                                         decay_parameter=decay_parameter, constant_offset=constant_offset)
        guess_list = naive_semantic_predict_word_sense(sem_network[0], sentence_list, word_sense_dict, iterations,
                                                       sem_network[1])
    else:
        return False
    raw_truths = sum(guess_list.values(), [])
    return raw_truths.count(True) / len(raw_truths)


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


def create_sem_network(sentence_list, spreading=True, time=False, activation_base=2, decay_parameter=0.05,
                       constant_offset=0):
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
    timer = 1
    semcor_words = sum(sentence_list, [])
    if spreading:
        for sentence in sentence_list:
            for word in sentence:
                # if not network.retrievable(word):
                syn = word[1]
                lemma = word[0]
                synonyms = [(synon, syn) for synon in syn.lemmas() if (synon, syn) in semcor_words and synon != lemma]
                # These are all synsets.
                synset_relations = [syn.hypernyms(), syn.hyponyms(),
                                  syn.member_holonyms() + syn.substance_holonyms() + syn.part_holonyms(),
                                  syn.member_meronyms() + syn.substance_meronyms() + syn.part_meronyms(),
                                  syn.attributes(), syn.entailments(), syn.causes(), syn.also_sees(), syn.verb_groups(),
                                  syn.similar_tos()]
                lemma_relations = []
                for ii in range(len(synset_relations)):
                    lemma_relations.append([])
                    for jj in range(len(synset_relations[ii])):
                        syn_lemmas = synset_relations[ii][jj].lemmas()
                        for lemma in syn_lemmas:
                            lemma_relations[ii].append((lemma, synset_relations[ii][jj]))
                for ii in range(len(lemma_relations)):
                    lemma_relations[ii] = [word_tuple for word_tuple in set(lemma_relations[ii]) if word_tuple in
                                           semcor_words and word_tuple != word]
                print(lemma_relations)
                # Get lemmas out of synsets
                # hypernyms = [hyper for hyper in syn.hypernyms() if (hyper, hyper.label().synset()) in semcor_words and hyper != lemma]
                # hyponyms = [hypo for hypo in syn.hyponyms() if (hypo, hypo.label().synset()) in semcor_words and hypo != lemma]
                # holonyms = [holo for holo in syn.member_holonyms() + syn.substance_holonyms() + syn.part_holonyms()
                #             if
                #             (holo, holo.label().synset()) in semcor_words and holo != lemma]
                # meronyms = [mero for mero in syn.member_meronyms() + syn.substance_meronyms() + syn.part_meronyms()
                #             if
                #             (mero, mero.label().synset()) in semcor_words and mero != lemma]
                # attributes = [attr for attr in syn.attributes() if (attr, attr.label().synset()) in semcor_words and attr != lemma]
                # entailments = [entail for entail in syn.entailments() if (entail, entail.label().synset()) in semcor_words and entail != lemma]
                # causes = [cause for cause in syn.causes() if (cause, cause.label().synset()) in semcor_words and cause != lemma]
                # also_sees = [also_see for also_see in syn.also_sees() if
                #              (also_see, also_see.label().synset()) in semcor_words and also_see != lemma]
                # verb_groups = [verb for verb in syn.verb_groups() if (verb, verb.label().synset()) in semcor_words and verb != lemma]
                # similar_tos = [sim for sim in syn.similar_tos() if (sim, sim.label().synset())  in semcor_words and sim != lemma]
                network.store(mem_id=word,
                              time=timer,
                              synonyms=synonyms,
                              hypernyms=lemma_relations[0],
                              hyponyms=lemma_relations[1],
                              holynyms=lemma_relations[2],
                              meronyms=lemma_relations[3],
                              attributes=lemma_relations[4],
                              entailments=lemma_relations[5],
                              causes=lemma_relations[6],
                              also_sees=lemma_relations[7],
                              verb_groups=lemma_relations[8],
                              similar_tos=lemma_relations[9])
                timer += 1
    else:
        for sentence in sentence_list:
            for word in sentence:
                network.store(mem_id=word, time=timer)
            timer += 1
    if time:
        return network, timer
    else:
        return network


def naive_predict_word_sense(sentence_list, word_sense_dict):
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
    # Getting sense-pair counts
    sense_pair_counts = {}
    for sentence in sentence_list:
        for word1_index in range(len(sentence)):
            for word2_index in range(word1_index + 1, len(sentence)):
                word1 = sentence[word1_index]
                word2 = sentence[word2_index]
                if word1[0] < word2[0] or (word1[0] == word2[0] and word1[1] <= word2[1]):
                    sense_key = (word1, word2)
                else:
                    sense_key = (word2, word1)
                if sense_key not in sense_pair_counts.keys():
                    sense_pair_counts[sense_key] = 0
                sense_pair_counts[sense_key] += 1

    for sentence in sentence_list:
        for word in sentence:
            sense_cooccurrence_dict = {}
            for lemma in word_sense_dict[word[0].name()]:
                for cooccur_word in sentence:
                    if cooccur_word != word:
                        if lemma not in sense_cooccurrence_dict.keys():
                            sense_cooccurrence_dict[lemma] = 0
                        if lemma[0] < cooccur_word[0] or (lemma[0] == cooccur_word[0] and lemma[1] <= cooccur_word[1]):
                            sense_pair = (lemma, cooccur_word)
                        else:
                            sense_pair = (cooccur_word, lemma)
                        #sense_cooccurrence_dict[lemma] += sem_network.get_cooccurrence(lemma, cooccur_word)
                        if sense_pair in sense_pair_counts.keys():
                            sense_cooccurrence_dict[lemma] += sense_pair_counts[sense_pair]
            word_sense_guess = max(zip(sense_cooccurrence_dict.values(), sense_cooccurrence_dict.keys()))[1]
            time += 1
            if word not in guess_list.keys():
                guess_list[word] = []
            if word_sense_guess == word:
                guess_list[word].append(True)
            else:
                guess_list[word].append(False)
    return guess_list


def naive_semantic_predict_word_sense(sem_network, sentence_list, word_sense_dict, iterations, time):
    timer = time
    guess_list = {}
    for iter in range(iterations):
        for sentence in sentence_list:
            for word in sentence:
                # senses = sem_network.sense_query(word=word[0], time=timer)
                senses = word_sense_dict[word[0].name()]
                max_act = -1
                for sense in senses:
                    sense_act = sem_network.get_activation(mem_id=sense, time=timer)
                    if sense_act > max_act:
                        max_act = sense_act
                        sense_guess = sense
                if word not in guess_list.keys():
                    guess_list[word] = []
                if sense_guess == word:
                    guess_list[word].append(True)
                else:
                    guess_list[word].append(False)
                sem_network.store(mem_id=word, time=timer)
                sem_network.store(mem_id=sense_guess, time=timer)
                timer += 1
    return guess_list


def senseless_predict_word_sense(sentence_list, word_sense_dict):
    guess_list = {}
    sense_pair_counts = {}
    for sentence in sentence_list:
        for word1_index in range(len(sentence)):
            for word2_index in range(word1_index + 1, len(sentence)):
                word1 = sentence[word1_index]
                word2 = sentence[word2_index]
                if word1[0] < word2[0] or (word1[0] == word2[0] and word1[1] <= word2[1]):
                    sense_key = (word1, word2)
                else:
                    sense_key = (word2, word1)
                if sense_key not in sense_pair_counts.keys():
                    sense_pair_counts[sense_key] = 0
                sense_pair_counts[sense_key] += 1

    for sentence in sentence_list:
        for word in sentence:
            sense_cooccurrence_dict = {}
            for word_sense in word_sense_dict[word[0].name()]:
                for cooccur_word in sentence:
                    if cooccur_word != word:
                        for cooccur_sense in word_sense_dict[cooccur_word[0].name()]:
                            if word_sense[0] < cooccur_sense[0] or (word_sense[0] == cooccur_sense[0] and word_sense[1] <= cooccur_sense[1]):
                                sense_key = (word_sense, cooccur_sense)
                            else:
                                sense_key = (cooccur_sense, word_sense)
                            if word_sense not in sense_cooccurrence_dict.keys():
                                sense_cooccurrence_dict[word_sense] = 0
                            if sense_key in sense_pair_counts.keys():
                                sense_cooccurrence_dict[word_sense] += sense_pair_counts[sense_key]
                            #sense_cooccurrence_dict[word_sense] += sem_network.get_cooccurrence(word_sense,
                                                                                                #cooccur_sense)
            word_sense_guess = max(zip(sense_cooccurrence_dict.values(), sense_cooccurrence_dict.keys()))[1]
            if word not in guess_list.keys():
                guess_list[word] = []
            if word_sense_guess == word:
                guess_list[word].append(True)
            else:
                guess_list[word].append(False)
    return guess_list


def frequency_predict_word_sense(sentence_list, word_sense_dict):
    absolute_sense_counts = {}
    for sentence in sentence_list:
        for word in sentence:
            if word not in absolute_sense_counts.keys():
                absolute_sense_counts[word] = 0
            absolute_sense_counts[word] += 1
    guess_list = {}
    for sentence in sentence_list:
        for word in sentence:
            senses = word_sense_dict[word[0].name()]
            max_sense_count = 0
            for sense in senses:
                if absolute_sense_counts[sense] > max_sense_count:
                    max_sense = sense
                    max_sense_count = absolute_sense_counts[sense]
            if word not in guess_list.keys():
                guess_list[word] = []
            if max_sense == word:
                guess_list[word].append(True)
            else:
                guess_list[word].append(False)
    return guess_list


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
# print(extract_sentences()[1])
# print(create_sem_network(extract_sentences()[0], 2, 0.05, 0))
# print(run_wsd(2, 0.05, 0, predict_word_sense="senseless"))
# print(get_corpus_stats())
#print(run_wsd(predict_word_sense="naive_sem_spreading"))
#print(run_wsd(predict_word_sense="naive"))
