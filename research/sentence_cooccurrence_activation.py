import math
from research import ActivationDynamics
from collections import defaultdict


class SentenceCooccurrenceActivation(ActivationDynamics):
    """Activation functions to calculate the base level activation of objects with pairwise cooccurrence"""


    def __init__(self, ltm, constant_offset=0, activation_base=2, decay_parameter=0.05, **kwargs):
        """Initialize the ActivationDynamics.
        Parameters:
            ltm (LongTermMemory): The LongTermMemory that will be using this activation.
        """
        super().__init__(ltm, **kwargs)
        self.activations = defaultdict(lambda: list())
        self.constant_offset = constant_offset
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter
        # Last activated elements is a list of lists where each list refers
        # to a cooccurrent element, s.t. ["mem_id", #Times Cooccurred]
        #self.last_activated_elements = []
        self.last_activated_time = 0
        # Cooccurrent elements is a dictionary where the key for each one is the mem_id and the value
            # is how many times the element has cooccurred with the element in question.
        #self.cooccurrent_elements = {}


    def activate_pair(self, word1, word2, time):



     """
    def activate_sentence(self, sentence_words, time):
        Activate the element with the given ID using spreading activation.
        Parameters:
            sentence_words (list): A nested list with words in each sentence that need to be activated, in the form
                ["mem_id", "word_sense"]
            time (int): The time of activation. Optional.

        for word1 in sentence_words:
            for word2 in sentence_words:
                if word1 != word2:
                    if word1 < word2:
                        key = word1 + "+" + word2 # (word1, word2) tuple, named tuple
                    else:
                        key = word2 + "+" + word1
                    if key in self.cooccurrent_elements:
                        curr_cooccurrence_value = self.cooccurrent_elements.get(key) + 1
                        self.cooccurrent_elements[key] = curr_cooccurrence_value
                        element_pair_ratio = 1 + (curr_cooccurrence_value / (len(self.activations[word1]) + 1))
                    else:
                        self.cooccurrent_elements[key] = 1
                        element_pair_ratio = 1 + (1 / (len(self.activations[word1]) + 1))
                    self.activations[word1].append([time, 1, element_pair_ratio])
        # Spreading activation for each word.
        for word in sentence_words:
            curr_act_candidates = list(self.ltm.knowledge.get(word))
            next_act_candidates = list(self.ltm.knowledge.get(word))
            graph_units = 1
            while curr_act_candidates != []:
                curr_act_candidates = next_act_candidates
                next_act_candidates = []
                for element in curr_act_candidates:
                    if element != []:
                        connection = list(element)[1]
                        self.activations[connection].append([time, self.activation_base ** (-graph_units), 1])
                        new_links = list(self.ltm.knowledge.get(connection))
                    for link in new_links:
                        if type(link) == list:
                            for item in link:
                                next_act_candidates.append(item)
                        else:
                            next_act_candidates.append(link)
                graph_units += 1


    def activate_guess(self, mem_id, sentence_words, time):
            for word in sentence_words:
                if word != mem_id:
                    if word + "+" + mem_id in self.cooccurrent_elements:
                        curr_cooccurrence_value = self.cooccurrent_elements.get(word + "+" + mem_id) + 1
                        self.cooccurrent_elements[word + "+" + mem_id] = curr_cooccurrence_value
                        element_pair_ratio = 1 + (curr_cooccurrence_value / (len(self.activations[mem_id]) + 1))
                    elif mem_id + "+" + word in self.cooccurrent_elements:
                        curr_cooccurrence_value = self.cooccurrent_elements.get(mem_id + "+" + word) + 1
                        self.cooccurrent_elements[mem_id + "+" + word] = curr_cooccurrence_value
                        element_pair_ratio = 1 + (curr_cooccurrence_value / (len(self.activations[mem_id]) + 1))
                    else:
                        self.cooccurrent_elements[word + "+" + mem_id] = 1
                        element_pair_ratio = 1 + (1 / (len(self.activations[mem_id]) + 1))
                    self.activations[mem_id].append([time, 1, element_pair_ratio])
                    # Spreading activation for each word.
                    curr_act_candidates = list(self.ltm.knowledge.get(mem_id))
                    next_act_candidates = list(self.ltm.knowledge.get(mem_id))
                    graph_units = 1
                    while curr_act_candidates != []:
                        curr_act_candidates = next_act_candidates
                        next_act_candidates = []
                        for element in curr_act_candidates:
                            if element != []:
                                connection = list(element)[1]
                                self.activations[connection].append([time, self.activation_base ** (-graph_units), 1])
                                new_links = list(self.ltm.knowledge.get(connection))
                            for link in new_links:
                                if type(link) == list:
                                    for item in link:
                                        next_act_candidates.append(item)
                                else:
                                    next_act_candidates.append(link)
                        graph_units += 1

                """

    def get_activation(self, mem_id, time):
        """Get the activation of the element with the given ID.
        Parameters:
            mem_id (any): The ID of the desired element.
            time (int): The time of activation. Optional.
        Returns:
            float: The activation of the element.
        """
        act_times_list = self.activations[mem_id]
        if act_times_list == [] or (len(act_times_list) == 1 and act_times_list[0][0] == 0):
            return None
        time_since_last_act_list = [[time - time_spreading_pair[0], time_spreading_pair[1]] for time_spreading_pair
                                    in act_times_list]
        base_act_sum_term = 0
        for retrieval_pair in range(len(time_since_last_act_list)):
            if (act_times_list[retrieval_pair][0] > 0):
                base_act_sum_term = act_times_list[retrieval_pair][2] * (time_since_last_act_list[retrieval_pair][1] * (
                        time_since_last_act_list[retrieval_pair][0] ** (-self.decay_parameter))) + base_act_sum_term

        base_level_activation = self.constant_offset + math.log(base_act_sum_term)
        return base_level_activation


    def get_word_sense_cooccurrence(self, word_to_guess, sentence_words):
        # Sentence words cannot contain the word of interest.
        sense_cooccurrence = {}
        for key in self.cooccurrent_elements.keys():
            if word_to_guess + "." in key:
                for sent_word in sentence_words:
                    if key in sent_word:
                        start_index = key.rfind(word_to_guess + ".")
                        word_sense = key[start_index + 1: start_index + 2]
                        sense_cooccurrence[word_sense] = sense_cooccurrence[word_sense] + self.cooccurrent_elements[key]
        return sense_cooccurrence
    # FIXME Make another dictionary when looping through and training that has all of the word senses for each word
    #activation takes in the sense and a sentence, and LTM does for sense in word... and loops through each sentence


