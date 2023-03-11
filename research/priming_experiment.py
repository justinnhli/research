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


def create_word_list(occ_comparison, occ_target, num_word_pairs):
    """Creates the list of 100 word pairs. Assumes occ_comparison + occ_target < 1.00
    Parameters:
        occ_comparison (float): The frequency from 0 to 1 of the comparison element.
        occ_target (float): The frequency from 0 to 1 of the comparison element.
    Returns:
        list: The list of word pairs.
            """
    word_pair_list = list()
    num_target = round(occ_target * num_word_pairs)
    num_comparison = round(occ_comparison * num_word_pairs)
    num_filler = num_word_pairs - num_target - num_comparison
    for i in range(1, num_target + 1):
        word_pair_list.append(['shared', 'target'])
    for i in range(1, num_comparison + 1):
        word_pair_list.append(['shared', 'comparison'])
    for i in range(1, num_filler + 1):
        word_pair_list.append(['filler1', 'filler2'])
    return word_pair_list




def create_sem_network(target_distance, constant_offset, decay_parameter, activation_base):
    """Get the activation of the element with the given ID.
    Parameters:
        target_distance (int): The distance between the target word and the shared word, elements will be placed
            in between the target and shared words corresponding to this quantity.
    Returns:
        NaiveDictLTM: A network containing the elements necessary for the experiment
            """

    network = NaiveDictLTM(activation_cls=(lambda ltm:
        BaseLevelActivation(ltm, activation_base, constant_offset, decay_parameter)
    ))
    #network.activation_dynamics.activation_base =


    for word in ['shared',
                 'comparison',
                 'not_prime',
                 'filler2',
                 'filler1',
                 'dummy_not_prime',
                 'dummy_comparison',
                 'dummy_shared',
                 'dummy_filler1',
                 'dummy_filler2']:
        network.store(mem_id=word, time=0)
    prime_target_list = []
    if target_distance == 1:
        prime_target_list.append(['prime', 'target'])
    else:
        prime_target_list.append(['node1','target'])
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



def run_trial(word_pair_list, sem_network, link, prime):
    """
    Runs each trial of the experiment by activating the words in the word pair list then running the experiment.
    Parameters:
        word_pair_list (list): The list of 100 shuffled word pairs.
        sem_network (NaiveDictLTM): The semantic network to be used in the experiment
        comparison (Boolean): Determines whether 'not_prime', the comparison condition (true), or 'prime', the
            experimental condition (false), should be presented before the shared word.
    Returns:
        float: The difference between the target and comparison activation in each experiment.
    """
    timer = 1
    for pair in word_pair_list:
        sem_network.retrieve(mem_id=pair[0], time=timer)
        sem_network.retrieve(mem_id=pair[1], time=timer)
        #FIXME delete link parameter
        if link:
            if not sem_network.retrievable(mem_id=pair[0]+"+"+pair[1]):
                sem_network.store(mem_id=pair[0]+"+"+pair[1], time=timer, word_1=pair[0], word_2=pair[1])
            sem_network.retrieve(mem_id=pair[0]+"+"+pair[1], time=timer)
        timer += 1

    if prime:
        first_word_presented = 'prime'
    else:
        first_word_presented = 'not_prime'
    sem_network.retrieve(mem_id=first_word_presented, time=timer)
    sem_network.retrieve(mem_id="shared", time=timer)

    #FIXME Take out comparison_activation
    comparison_activation = sem_network.get_activation("comparison", time=timer + 2)
    target_activation = sem_network.get_activation("target", time=timer + 2)
    return target_activation, comparison_activation



def run_experiment(target_distance, num_trials, occ_comparison, occ_target, num_word_pairs, prime,
                   link, constant_offset, decay_parameter, activation_base):
    """
        Runs the experiment, calling the run_trial function for each trial.
        Parameters:
            target_distance (int): The distance between the target word and the shared word, elements will be placed
                in between the target and shared words corresponding to this quantity.
            num_trials (int): The number of trials to run and average.
            occ_comparison (float): The frequency from 0 to 1 of the comparison element.
            occ_target (float): The frequency from 0 to 1 of the comparison element.
            comparison (Boolean): Determines whether 'not_prime', the comparison condition (true), or 'prime', the
                experimental condition (false), should be presented before the shared word.
        Returns:
            list: The differences between target and comparison activation for each item in the list.
                """
    word_pair_list = create_word_list(occ_comparison, occ_target, num_word_pairs)
    target_act_list = []
    comparison_act_list = []
    for i in range(num_trials):
        shuffle_word_pair_list = word_pair_list.copy()
        sem_network = create_sem_network(target_distance, constant_offset, decay_parameter, activation_base)
        random.shuffle(shuffle_word_pair_list)
        result = run_trial(shuffle_word_pair_list, sem_network, link, prime)
        target_act_list.append(result[0])
        comparison_act_list.append(result[1])
    return target_act_list
