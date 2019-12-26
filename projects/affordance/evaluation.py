"""Evaluate all scenarios."""

import sys
from os.path import dirname, realpath, join as join_path
from collections import namedtuple

# make sure research library code is available
ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)

from research.word_embedding import load_model # pylint: disable=wrong-import-position
from students.lijia.word2vec import possible_actions # pylint: disable=wrong-import-position

GOOGLE_NEWS_MODEL_PATH = join_path(ROOT_DIRECTORY, 'data/models/GoogleNews-vectors-negative300.bin')
SCENARIO_DIRECTORY = join_path(dirname(realpath(__file__)), 'test-data')

Scenario = namedtuple('Scenario', ['id', 'description', 'actions'])
TestResult = namedtuple('TestResult', ['scenario', 'actions', 'true_positives', 'false_positives', 'false_negatives'])


def object_verb_tool_key(action_str):
    """Key function for sorting actions.

    Arguments:
        action_str (str): The string representing an action.

    Returns:
        tuple[str]: A tuple containing the object, the verb, and optionally the
            tool

    Raises:
        ValueError: If the action_str cannot be parsed.
    """
    words = action_str.split()
    if len(words) == 2:
        # verb noun
        return (words[1], words[0])
    elif len(words) == 3:
        # verb preposition noun
        return (words[2], ' '.join(words[0:2]))
    elif len(words) == 4 and words[2] == 'with':
        # verb noun with tool
        return (words[1], words[0], words[3])
    else:
        raise ValueError('Cannot sort action: "{}"'.format(action_str))


def get_scenario(scene_num):
    """Test actions for a scenario.

    Arguments:
        test_num (int): The scenario id.

    Returns:
        Scenario: The full path of the scenario file.

    Raises:
        IOError: If the scenario file does not exist
    """
    scene_file = '{:03d}.txt'.format(scene_num)
    scene_path = join_path(SCENARIO_DIRECTORY, scene_file)
    with open(scene_path) as fd:
        lines = [('' if line.strip().startswith('#') else line) for line in fd.readlines()]
        description, actions = '\n'.join(lines).split('\n\n', maxsplit=1)
        actions = set([action for action in actions.split("\n") if action.split()])
        return Scenario(scene_num, description, actions)


def run_scenario(scenario, model):
    """Test actions for a scenario.

    Arguments:
        scenarios (Scenario): The scenario to test.
        model (Model): The gensim model to use.

    Returns:
        TestResult: The correct and generated actions for the scenario.
    """
    actions = possible_actions(model, scenario.description)
    true_positives = sorted(
        scenario.actions & set(actions),
        key=object_verb_tool_key,
    )
    false_positives = sorted(
        set(actions) - scenario.actions,
        key=object_verb_tool_key,
    )
    false_negatives = sorted(
        scenario.actions - set(actions),
        key=object_verb_tool_key,
    )
    return TestResult(scenario, actions, true_positives, false_positives, false_negatives)


def print_scenario_result(result):
    """Test actions for a scenario.

    Arguments:
        result (TestRest): The test results for a scenario.
    """
    print('Scenario {} ({}...)'.format(
        result.scenario.id,
        ' '.join(result.scenario.description.split()[:5]),
    ))
    print('    Expected Actions: {}'.format(result.scenario.actions))
    print('    True Positives: {} {}'.format(
        len(result.true_positives),
        result.true_positives,
    ))
    print('    False Positives: {} {}'.format(
        len(result.false_positives),
        result.false_positives,
    ))
    print('    False Negatives: {} {}'.format(
        len(result.false_negatives),
        result.false_negatives,
    ))


def main():
    """Evaluate all scenarios."""
    model = load_model(GOOGLE_NEWS_MODEL_PATH)
    for scene_num in range(1, 30):
        scenario = get_scenario(scene_num)
        result = run_scenario(scenario, model)
        print_scenario_result(result)


if __name__ == '__main__':
    main()
