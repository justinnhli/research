from priming_experiment import *

""" Test Case #1
Testing BaseLevelActivation Class

Gets and compares activations at time 2, time 4, time 6, and time 8 for each of four elements. 
Element one is activated at time 1, element two at time 3, element three at time 5, element four at time 7

"""
def case_test_base_level_act_class():
    test1_ltm = NaiveDictLTM(activation_cls=BaseLevelActivation)
    test1_ltm.store(mem_id='elem1', links_to='elem2', time=0)
    test1_ltm.store(mem_id='elem1', links_to='elem4', time=0)
    test1_ltm.store(mem_id='elem2', links_to='elem3', time=0)
    test1_ltm.store(mem_id='elem3', time=0)
    test1_ltm.store(mem_id='elem4', time=0)

    BaseLevelActivation.constant_offset = 0
    BaseLevelActivation.decay_parameter = 0.05
    BaseLevelActivation.activation_base = 2

    # Time 1: Activate Element #1
    test1_ltm.retrieve('elem1', time=1)

    # Time 2: Test Activations
    assert (test1_ltm.get_activation('elem1', time=2) == 0)
    assert (-0.694 < test1_ltm.get_activation('elem2', time=2) < -0.693)
    assert (-1.387 < test1_ltm.get_activation('elem3', time=2) < -1.386)
    assert (-0.694 < test1_ltm.get_activation('elem4', time=2) < -0.693)

    #Time 3: Activate Element #2
    test1_ltm.retrieve('elem2', time=3)

    #Time 4: Test Activations
    assert (-0.055 < test1_ltm.get_activation('elem1', time=4) < -0.054)
    assert (0.387 < test1_ltm.get_activation('elem2', time=4) < 0.388)
    assert (-0.306 < test1_ltm.get_activation('elem3', time=4) < -0.305)
    assert (-0.749 < test1_ltm.get_activation('elem4', time=4) < -0.748)

    #Time 5: Activate Element #3
    test1_ltm.retrieve('elem3', time=5)

    #Time 6: Test Activations
    assert (-0.081 < test1_ltm.get_activation('elem1', time=6) < -0.080)
    assert (0.342 < test1_ltm.get_activation('elem2', time=6) < 0.343)
    assert (0.532 < test1_ltm.get_activation('elem3', time=6) < 0.533)
    assert (-0.774 < test1_ltm.get_activation('elem4', time=6) < -0.773)

    # Time 7: Activate Element #4
    test1_ltm.retrieve('elem4', time=7)

    # Time 8: Test Activations
    assert (-0.098 < test1_ltm.get_activation('elem1', time=8) < -0.097)
    assert (0.319 < test1_ltm.get_activation('elem2', time=8) < 0.320)
    assert (0.491 < test1_ltm.get_activation('elem3', time=8) < 0.492)
    assert (0.374 < test1_ltm.get_activation('elem4', time=8) < 0.375)



""" Test Cases #2 - #4
Testing operation of the BaseLevelActivation class and experiment code. 
Uses same functions, but edits wordpair list to be a preset list of words, so that the order is not 
randomized

Function Below:
"""
def run_experiment_test_cases(word_pair_list, target_distance, control, constant_offset, decay_parameter, activation_base):
    """
        Runs the experiment, calling the run_trial function for each trial.
        Parameters:
            word_pair_list (list): List of word pairs to be created by the user.
            target_distance (int): The distance between the target word and the root word, elements will be placed
                in between the target and root words corresponding to this quantity.
            num_trials (int): The number of trials to run and average.
            control (Boolean): Determines whether 'filler3', the control condition (true), or 'prime', the
                experimental condition (false), should be presented before the root word.
        Returns:
            list: The differences between target and control activation for each item in the list.
                """
    sem_network = create_sem_network(target_distance, constant_offset, decay_parameter, activation_base)
    result = run_trial(word_pair_list, sem_network, control)
    return result

test_234_word_pair_list = [['filler1', 'filler2'], ['root', 'target'], ['root', 'control'],
                              ['filler1', 'filler2'], ['root', 'target'], ['root', 'control'],
                              ['filler1', 'filler2'], ['root', 'target'], ['root', 'control'],
                              ['filler1', 'filler2']]

"""
Test Case #2:
target_distance = 1
num_trials = 1
occ_control = 0.3
occ_target = 0.3
num_word_pairs = 1
control = False
constant_offset = 0
decay_parameter = 0.05
activation_base = 2
"""

def case_semantic_experimental_condition():
    assert (-0.004 < run_experiment_test_cases(test_234_word_pair_list, 1, True, 0, 0.05, 2) < -0.003)


"""
Test Case #3:
target_distance = 1
num_trials = 1
occ_control = 0.3
occ_target = 0.3
num_word_pairs = 1
control = True
constant_offset = 0
decay_parameter = 0.05
activation_base = 2
"""

def case_control_experimental_condition():
    assert (0.074 < run_experiment_test_cases(test_234_word_pair_list, 1, False, 0, 0.05, 2) < 0.075)


"""
Test Case #4:
target_distance = 3
num_trials = 1
occ_control = 0.3
occ_target = 0.3
num_word_pairs = 1
control = True
constant_offset = 0
decay_parameter = 0.05
activation_base = 2
"""

def case_nontrivial_target_distance():
    assert 0.016 < run_experiment_test_cases(test_234_word_pair_list, 3, False, 0, 0.05, 2) < 0.017


"""
Test Case #5:
Testing the BaseLevelActivation class to see if activation reaches all nodes in a "string". 
Here, I created a long chain of 10 nodes and activated the first node in the list. Then I checked the activation
at different intervals in the list to ensure it was matching up with prediction. 
"""

def case_base_activation_spreading_act():
    test5_ltm = NaiveDictLTM(activation_cls=BaseLevelActivation)
    BaseLevelActivation.constant_offset = 0
    BaseLevelActivation.decay_parameter = 0.05
    BaseLevelActivation.activation_base = 2
    test5_ltm.store(mem_id='elem0', links_to='elem1', time=0)
    test5_ltm.store(mem_id='elem1', links_to='elem2', time=0)
    test5_ltm.store(mem_id='elem2', links_to='elem3', time=0)
    test5_ltm.store(mem_id='elem3', links_to='elem4', time=0)
    test5_ltm.store(mem_id='elem4', links_to='elem5', time=0)
    test5_ltm.store(mem_id='elem5', links_to='elem6', time=0)
    test5_ltm.store(mem_id='elem6', links_to='elem7', time=0)
    test5_ltm.store(mem_id='elem7', links_to='elem8', time=0)
    test5_ltm.store(mem_id='elem8', links_to='elem9', time=0)
    test5_ltm.store(mem_id='elem9', time=0)
    test5_ltm.retrieve(mem_id='elem0', time=1)

    # Retrieve activations at different points down the chain
    assert (test5_ltm.get_activation('elem0', time=2) == 0)
    assert (-1.387 < test5_ltm.get_activation('elem2', time=2) < -1.386)
    assert (-2.773 < test5_ltm.get_activation('elem4', time=2) < -2.772)
    assert (-4.159 < test5_ltm.get_activation('elem6', time=2) < -4.158)
    assert (-4.853 < test5_ltm.get_activation('elem7', time=2) < -4.852)
    assert (-5.546 < test5_ltm.get_activation('elem8', time=2) < -5.545)
    assert (-6.239 < test5_ltm.get_activation('elem9', time=2) < -6.238)


#Run all test cases here:
case_test_base_level_act_class()
case_semantic_experimental_condition()
case_control_experimental_condition()
case_nontrivial_target_distance()
case_base_activation_spreading_act()



