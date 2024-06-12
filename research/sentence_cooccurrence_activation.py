import math
from research import ActivationDynamics
from collections import defaultdict


class SentenceCooccurrenceActivation(ActivationDynamics):
    """Activation functions to calculate the base level activation of objects with pairwise cooccurrence. Many of the
    cooccurrence functionalities are no longer used."""

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

    def simple_activate(self, mem_id, spread_depth=-1, time=0):
        """
        Activates a given element and its neighbors via spreading activation.
        Parameters:
            mem_id (any): The ID of the desired element.
            spread_depth (int): The depth of connections to activate when a given element is activated. Serves mainly to
                allow the same setup for non-spreading and spreading scenarios.
            time (float): The time of retrieval (for activation) (optional)
        Returns:
            True: If completed.
        """
        self.activations[mem_id].append([time, 1])
        prev_act_candidates = set([mem_id]) # Candidates that have been activated before (prevents infinite looping)
        # Candidates to activate next (for spreading)
        curr_act_candidates = [list(element)[1] for element in list(self.ltm.knowledge.get(mem_id))]
        if spread_depth != 0:  # if spreading is allowed...
            graph_units = 1 # distance from originally activated node
            while curr_act_candidates: # checking that spreading still needs to be done
                next_act_candidates = []
                for element in curr_act_candidates:
                    if element is not None:
                        # activate element
                        self.activations[element].append([time, self.activation_base ** (-graph_units)])
                        # Add to next things to activate, the connections of the element we just activated
                        new_links = list(self.ltm.knowledge.get(element))
                        for link in new_links:
                            if type(link) == list:
                                for item in link:
                                    if item not in next_act_candidates:
                                        next_act_candidates.append(list(item)[1])
                            else:
                                if link not in next_act_candidates:
                                    next_act_candidates.append(list(link)[1])
                # If we don't want to spread farther - stop going through connections and activating them
                if graph_units == spread_depth:
                    break
                graph_units = graph_units + 1 # Moving to the next "round" of connections
                prev_act_candidates.update(curr_act_candidates)
                # Making sure that we haven't already activated the elements we are to "spread" to next...
                curr_act_candidates = [x for x in next_act_candidates if x not in prev_act_candidates]
                # If there's no more elements to activate, done!
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
        # We first create a nested list where each entry is a list: [time since last activation,
        # "graph distance multiplier"] where the "graph distance multiplier" is a previously calculated indicator of how
        # far away from the originally activated word the word being activated is.
        time_since_last_act_list = [[time - time_spreading_pair[0], time_spreading_pair[1]] for time_spreading_pair
                                    in act_times_list]
        base_act_sum_term = 0
        # For every activation...
        for retrieval_pair in range(len(time_since_last_act_list)):
            # Using the base activation equation to calculate the term referring to each activation
            if (act_times_list[retrieval_pair][0] > 0):
                base_act_sum_term = (time_since_last_act_list[retrieval_pair][1] * (
                        time_since_last_act_list[retrieval_pair][0] ** (-self.decay_parameter))) + base_act_sum_term
        # To finish calculating the activation of the element in question, adding the constant offset (normally 0) and
        # then taking the log of the aggregate term calculated in the for loop above.
        base_level_activation = self.constant_offset + math.log(base_act_sum_term)
        return base_level_activation

