import random
from research import NaiveDictLTM
from BaseLevelActivation import BaseLevelActivation


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


def create_word_list(occ_control, occ_target, num_word_pairs):
    """Creates the list of 100 word pairs. Assumes occ_control + occ_target < 1.00
    Parameters:
        occ_control (float): The frequency from 0 to 1 of the control element.
        occ_target (float): The frequency from 0 to 1 of the control element.
    Returns:
        list: The list of word pairs.
            """
    word_pair_list = list()
    num_target = round(occ_target * num_word_pairs)
    num_control = round(occ_control * num_word_pairs)
    num_filler = num_word_pairs - num_target - num_control
    for i in range(1, num_target + 1):
        word_pair_list.append(['root', 'target'])
    for i in range(1, num_control + 1):
        word_pair_list.append(['root', 'control'])
    for i in range(1, num_filler + 1):
        word_pair_list.append(['filler1', 'filler2'])
    return word_pair_list




def create_sem_network(target_distance, constant_offset, decay_parameter, activation_base, spread_act_setting, coocc_setting):
    """Get the activation of the element with the given ID.
    Parameters:
        target_distance (int): The distance between the target word and the root word, elements will be placed
            in between the target and root words corresponding to this quantity.
    Returns:
        NaiveDictLTM: A network containing the elements necessary for the experiment
            """
    network = NaiveDictLTM(activation_cls=BaseLevelActivation)
    BaseLevelActivation.activation_base = activation_base
    BaseLevelActivation.constant_offset = constant_offset
    BaseLevelActivation.decay_parameter = decay_parameter

    if coocc_setting == True and spread_act_setting == False:
        for word in ['root', 'target', 'prime', 'control', 'filler3', 'filler2', 'filler1']:
            network.store(mem_id=word, time=0)

    if coocc_setting:
        for pair in [['filler3', 'root'], ['prime', 'root'], ['root', 'target'], ['root', 'control'], ['filler1', 'filler2']]:
            network.store(mem_id=pair[0]+"+"+pair[1],
                          word1=pair[0],
                          word2=pair[1],
                          time=0)

    if spread_act_setting:
        sem_network = [['root', 'dummyR'],
                   ['root', 'control'],
                   ['root', 'target'],
                   ['control', 'dummyC'],
                   ['filler1', 'dummy1'],
                   ['filler2', 'dummy2'],
                   ['filler3', 'dummy3']]
        if target_distance == 1:
            sem_network.append(['prime', 'target'])
        else:
            sem_network.append(['node1','target'])
            last_node = 'node1'
            curr_node = 'node1'
            for i in range(2, target_distance):
                curr_node = 'node' + str(i)
                sem_network.append([curr_node, last_node])
                last_node = curr_node
            sem_network.append(['prime', curr_node])
        for key, value in sem_network:
            network.store(mem_id=key, links_to=value, time=0)

    return network



def run_trial(word_pair_list, sem_network, control, coocc_setting):
    """
    Runs each trial of the experiment by activating the words in the word pair list then running the experiment.
    Parameters:
        word_pair_list (list): The list of 100 shuffled word pairs.
        sem_network (NaiveDictLTM): The semantic network to be used in the experiment
        control (Boolean): Determines whether 'filler3', the control condition (true), or 'prime', the
            experimental condition (false), should be presented before the root word.
    Returns:
        float: The difference between the target and control activation in each experiment.
    """
    timer = 1
    for pair in word_pair_list:
        sem_network.retrieve(mem_id=pair[0], time=timer)
        sem_network.retrieve(mem_id=pair[1], time=timer)
        timer += 1

    if coocc_setting == True:
        timer = 1
        for pair in word_pair_list:
            sem_network.retrieve(mem_id=pair[0]+"+"+pair[1], time=timer)
            timer += 1

    if (control == True):
        first_word_presented = 'filler3'
    else:
        first_word_presented = 'prime'
    sem_network.retrieve(mem_id=first_word_presented, time=timer)
    sem_network.retrieve(mem_id="root", time=timer)
    if coocc_setting == True:
        sem_network.retrieve(mem_id=pair[0] + "+" + pair[1], time=timer)
    # FIXME Do I want root presented at same time as filler3/prime for each trial?
    control_activation = sem_network.get_activation("control", time=timer + 2)
    target_activation = sem_network.get_activation("target", time=timer + 2)
    return target_activation, control_activation



def run_experiment(target_distance, num_trials, occ_control, occ_target, num_word_pairs, control, spread_act_setting,
                   coocc_setting, constant_offset, decay_parameter, activation_base):
    """
        Runs the experiment, calling the run_trial function for each trial.
        Parameters:
            target_distance (int): The distance between the target word and the root word, elements will be placed
                in between the target and root words corresponding to this quantity.
            num_trials (int): The number of trials to run and average.
            occ_control (float): The frequency from 0 to 1 of the control element.
            occ_target (float): The frequency from 0 to 1 of the control element.
            control (Boolean): Determines whether 'filler3', the control condition (true), or 'prime', the
                experimental condition (false), should be presented before the root word.
        Returns:
            list: The differences between target and control activation for each item in the list.
                """
    word_pair_list = create_word_list(occ_control, occ_target, num_word_pairs)
    target_act_list = []
    control_act_list = []
    for i in range(num_trials):
        shuffle_word_pair_list = word_pair_list.copy()
        sem_network = create_sem_network(target_distance, constant_offset, decay_parameter, activation_base, spread_act_setting, coocc_setting)
        random.shuffle(shuffle_word_pair_list)
        result = run_trial(shuffle_word_pair_list, sem_network, control, coocc_setting)
        target_act_list.append(result[0])
        control_act_list.append(result[1])
    return target_act_list


run_experiment(3, 1, 0.3, 0.3, 100, True, False, True, 0, 0.05, 2)
print()

run_experiment(3, 1, 0.3, 0.3, 100, True, True, False, 0, 0.05, 2)