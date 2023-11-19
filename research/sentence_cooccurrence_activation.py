import math
from research import ActivationDynamics
from collections import defaultdict


class SentenceCooccurrenceActivation(ActivationDynamics):
    """Activation functions to calculate the base level activation of objects with pairwise cooccurrence"""

    def __init__(self, ltm, constant_offset=0, activation_base=2, decay_parameter=0.05, **kwargs):
        """Initialize the SentenceCooccurrenceActivation.
        Parameters:
            ltm (LongTermMemory): The LongTermMemory that will be using this activation.
            activation_base (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            constant_offset (float): A parameter in the activation equation.
        """
        super().__init__(ltm, **kwargs)
        self.activations = defaultdict(lambda: list())
        self.constant_offset = constant_offset
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter

    def simple_activate(self, mem_id, element_pair_ratio=1, time=0):
        """
        Activates a given element and its neighbors via spreading activation.
        Parameters:
            mem_id (any): The ID of the desired element.
            element_pair_ratio (float): The ratio of the number of times a pair of elements have cooccurred over the number
                of times the element of interest has been activated. (optional)
            time (float): The time of retrieval (for activation) (optional)
        Returns:
            True: If completed.
        """
        self.activations[mem_id].append([time, 1, element_pair_ratio])
        prev_act_candidates = set()
        curr_act_candidates = list(self.ltm.knowledge.get(mem_id))
        graph_units = 1
        while curr_act_candidates != []:
            next_act_candidates = []
            for element in curr_act_candidates:
                if element != []:
                    connection = list(element)[1]
                    self.activations[connection].append([time, self.activation_base ** (-graph_units), 1])
                    new_links = list(self.ltm.knowledge.get(connection))
                    for link in new_links:
                        if type(link) == list:
                            for item in link:
                                if item not in next_act_candidates:
                                    next_act_candidates.append(item)
                        else:
                            if link not in next_act_candidates:
                                next_act_candidates.append(link)
            graph_units = graph_units + 1
            prev_act_candidates.update(curr_act_candidates)
            curr_act_candidates = [x for x in next_act_candidates if x not in prev_act_candidates]
            if curr_act_candidates == [] or curr_act_candidates is None:
                break
        return True

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
        #print(base_act_sum_term)
        base_level_activation = self.constant_offset + math.log(base_act_sum_term)
        return base_level_activation

