# Create sem network from joint SWOWEN & SF Free Association Norms lists
import json
from collections import defaultdict
from research import sentence_long_term_memory



def make_combined_dict(swowen_link, sffan_link):
    """
    Combines preprocessed swowen and sffan dictionaries (with word as key and all of its connections/associations as values)
        into a single dictionary
    """
    swowen_dict = json.load(open(swowen_link))
    sffan_dict = json.load(open(sffan_link))
    combined_dict = defaultdict(set)
    for key in swowen_dict.keys():
        combined_dict[key].update(swowen_dict[key])
    for key in sffan_dict.keys():
        combined_dict[key].update(sffan_dict[key])
    for key in combined_dict.keys():
        combined_dict[key] = list(combined_dict[key])
    return combined_dict


def create_combined_sem_network(combined_dict, spreading=True, activation_base=2, decay_parameter=0.05,
                                constant_offset=0):
    """
    Builds a semantic network with each key word in the SWOWEN and South Florida Free Association Norms (SFFAN).
        Note that all words are stored at time 1.
    Parameters:
        SWOWEN_link (string): link to the SWOWEN preprocessed dictionary
        SFFAN_link (string): link to the SFFAN preprocessed dictionary
        spreading (bool): Whether to include the effects of spreading in creating the semantic network.
        activation_base (float): A parameter in the activation equation.
        decay_parameter (float): A parameter in the activation equation.
        constant_offset (float): A parameter in the activation equation.
        partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking at
            sentences 10000 - 14999.
    Returns:
        network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
    """
    if spreading:
        spread_depth = -1
    else:
        spread_depth = 0
    network = sentence_long_term_memory.sentenceLTM(
        activation_cls=(lambda ltm:
                        sentence_long_term_memory.SentenceCooccurrenceActivation(
                            ltm,
                            activation_base=activation_base,
                            constant_offset=constant_offset,
                            decay_parameter=decay_parameter
                        )))
    combined_keys = list(combined_dict.keys())
    for word in combined_keys:
        assocs = combined_dict[word]
        network.store(mem_id=word,
                      time=1,
                      spread_depth=spread_depth,
                      assocs=assocs)
    return network
