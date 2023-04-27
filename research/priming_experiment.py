import random
from research import NaiveDictLTM
from base_level_activation import BaseLevelActivation
from pairwise_cooccurrence_activation import PairwiseCooccurrenceActivation


def activation_fn(sem_network, mem_id, time):
    """Original Activation Function
    Parameters:
        sem_network (NaiveDictLTM): The semantic network containing the element to activate.
        mem_id (any): The ID of the desired element.
        time (int): The time of activation. Not used here.
    Returns:
        float: The activation of the element.
            """
    itemAct = sem_network.activations.get(mem_id)
    if itemAct != None:
        sem_network.activations.update({mem_id: itemAct + 1})
        itemAct += 1
    return None


def create_word_list(cooccur_1_freq, cooccur_2_freq, target_freq, num_word_pairs):
    """Creates the list of 100 word pairs.
    Parameters:
        cooccur_1_freq (float): The frequency from 0 to 1 of cooccur_1.
        cooccur_2_freq (float): The frequency from 0 to 1 of cooccur_2.
        target_freq (float): The frequency from 0 to 1 of the target word.
        num_word_pairs (int): The number of word pairs in the list.
    Returns:
        list: The list of word pairs.
            """
    word_pair_list = list()
    num_target_cooccur_1 = round(target_freq * cooccur_1_freq * num_word_pairs)
    num_target_cooccur_2 = round(target_freq * cooccur_2_freq * num_word_pairs)
    num_filler = num_word_pairs - num_target_cooccur_1 - num_target_cooccur_2
    for i in range(1, num_target_cooccur_1 + 1):
        word_pair_list.append(['cooccur_1', 'target'])
    for i in range(1, num_target_cooccur_2 + 1):
        word_pair_list.append(['cooccur_2', 'target'])
    for i in range(1, num_filler + 1):
        word_pair_list.append(['filler1', 'filler2'])
    return word_pair_list


def create_sem_network(target_distance, constant_offset, decay_parameter, activation_base, auto_storage=True):
    """Get the activation of the element with the given ID.
    Parameters:
        target_distance (int): The number of edges between the target word and the prime word.
        constant_offset (float): A parameter in the activation equation.
        decay_parameter (float): A parameter in the activation equation.
        activation_base (float): A parameter in the activation equation.
        auto_storage (Boolean): Determines which activation_dynamics class is used. If true, BaseLevelActivation is
            used. If false, PairwiseCooccurrenceActivation.
    Returns:
        NaiveDictLTM: A network containing the elements necessary for the experiment
            """
    if auto_storage:
        network = NaiveDictLTM(activation_cls=(lambda ltm:
                                               BaseLevelActivation(ltm,
                                                                   activation_base=activation_base,
                                                                   constant_offset=constant_offset,
                                                                   decay_parameter=decay_parameter)
                                               ))
    else:
        network = NaiveDictLTM(activation_cls=(lambda ltm:
                                               PairwiseCooccurrenceActivation(ltm,
                                                                              activation_base=activation_base,
                                                                              constant_offset=constant_offset,
                                                                              decay_parameter=decay_parameter)
                                               ))
    for word in ['filler_1',
                 'filler_2',
                 'control',
                 'cooccur_1',
                 'cooccur_2']:
        network.store(mem_id=word, time=0)
    prime_target_list = []
    if target_distance == 1:
        prime_target_list.append(['prime', 'target'])
    else:
        prime_target_list.append(['node1', 'target'])
        last_node = 'node1'
        curr_node = 'node1'
        for i in range(2, target_distance):
            curr_node = 'node' + str(i)
            prime_target_list.append([curr_node, last_node])
            last_node = curr_node
        prime_target_list.append(['prime', curr_node])
    for key, value in prime_target_list:
        network.store(mem_id=key, links_to=value, time=0)
    return network


def run_trial(word_pair_list, sem_network, semantic, cooccurrence, cooccur_num=1):
    """
    Runs each trial of the experiment by activating the words in the word pair list then running the experiment.
    Parameters:
        word_pair_list (list): The list of shuffled word pairs.
        sem_network (NaiveDictLTM): The semantic network to be used in the experiment.
        semantic (Boolean): If true, uses "prime" as the prime word. If false and cooccurrence == false, uses "control"
            as the prime word.
        cooccurrence (Boolean): If true, uses a cooccurrence element as the prime word. If false and semantic == false,
            uses "control" as the prime word.
        cooccur_num (int): Determines which cooccurrence element to use as the prime word when cooccurrence == true and
            semantic == false. If it equals 1, "cooccur_1" is used. If 2, "cooccur_2" is used.
    Returns:
        float: The difference between the target and comparison activation in each experiment.
    """
    if semantic and not cooccurrence:
        first_word_presented = 'prime'
    elif not semantic and cooccurrence and cooccur_num == 1:
        first_word_presented = 'cooccur_1'
    elif not semantic and cooccurrence and cooccur_num == 2:
        first_word_presented = 'cooccur_2'
    elif not semantic and not cooccurrence:
        first_word_presented = 'control'

    timer = 1
    for pair in word_pair_list:
        sem_network.retrieve(mem_id=pair[0], time=timer)
        sem_network.retrieve(mem_id=pair[1], time=timer)
        sem_network.store(mem_id=pair[0] + "+" + pair[1], time=timer, word_1=pair[0], word_2=pair[1])
        if sem_network.activation_dynamics.activations[pair[0] + "+" + pair[1]] == []:
            sem_network.retrieve(mem_id=pair[0] + "+" + pair[1], time=timer)
        timer += 1
    sem_network.retrieve(mem_id=first_word_presented, time=timer)
    sem_network.retrieve(mem_id="target", time=timer + 0.5)
    target_activation = sem_network.get_activation("target", time=timer + 1)
    return target_activation


def run_experiment(constant_offset, decay_parameter, activation_base, target_distance, num_trials, cooccur_1_freq,
                   cooccur_2_freq, target_freq, num_word_pairs, semantic, cooccurrence, cooccur_num=1,
                   auto_storage=True):
    """
        Runs the experiment, calling the run_trial function for each trial.
        Parameters:
            constant_offset (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            activation_base (float): A parameter in the activation equation.
            target_distance (int): The number of edges between the target word and the prime word.
            num_trials (int): How many trials the experiment should go through.
            cooccur_1_freq (float): The frequency from 0 to 1 of cooccur_1.
            cooccur_2_freq (float): The frequency from 0 to 1 of cooccur_2.
            target_freq (float): The frequency from 0 to 1 of the target word.
            num_word_pairs (int): The number of word pairs in the list.
            semantic (Boolean): If true, uses "prime" as the prime word. If false and cooccurrence == false, uses
                "control" as the prime word.
            cooccurrence (Boolean): If true, uses a cooccurrence element as the prime word. If false and
                semantic == false, uses "control" as the prime word.
            cooccur_num (int): Determines which cooccurrence element to use as the prime word when cooccurrence == true
                and semantic == false. If it equals 1, "cooccur_1" is used. If 2, "cooccur_2" is used.
            auto_storage (Boolean): Determines which activation_dynamics class is used. If true, BaseLevelActivation is
                used. If false, PairwiseCooccurrenceActivation.
        Returns:
            list: The differences between target and comparison activation for each item in the list.
                """
    word_pair_list = create_word_list(cooccur_1_freq, cooccur_2_freq, target_freq, num_word_pairs)
    target_act_list = []
    for i in range(num_trials):
        shuffle_word_pair_list = word_pair_list.copy()
        sem_network = create_sem_network(target_distance, constant_offset, decay_parameter, activation_base,
                                         auto_storage)
        random.shuffle(shuffle_word_pair_list)
        result = run_trial(shuffle_word_pair_list, sem_network, semantic, cooccurrence, cooccur_num)
        target_act_list.append(result)
    return target_act_list
