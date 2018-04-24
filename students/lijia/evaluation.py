"""Evaluate all scenarios."""

import sys
from os.path import dirname, realpath, join as join_path
from collections import namedtuple

# make sure research library code is available
ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)

from research.word_embedding import load_model # pylint: disable=wrong-import-position
from ifai import possible_actions # pylint: disable=wrong-import-position

GOOGLE_NEWS_MODEL_PATH = join_path(ROOT_DIRECTORY, 'data/models/GoogleNews-vectors-negative300.bin')
SCENARIO_DIRECTORY = join_path(dirname(realpath(__file__)), 'test-data')

Scenario = namedtuple('Scenario', ['description', 'actions'])


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
        return (words[1], words[0])
    elif len(words) == 4 and words[2] == 'with':
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
        return Scenario(description, actions)


def run_scenario(scene_num, model):
    """Test actions for a scenario.

    Arguments:
        scene_num (int): The scenario id.
    """
    scenario = get_scenario(scene_num)
    actions = possible_actions(model, scenario.description)
    print('Scenario {}'.format(scene_num))
    true_positives = sorted(
        scenario.actions & set(actions),
        key=object_verb_tool_key,
    )
    print('    True Positives: {} {}'.format(len(true_positives), true_positives))
    false_positives = sorted(
        scenario.actions - set(actions),
        key=object_verb_tool_key,
    )
    print('    False Positives: {} {}'.format(len(false_positives), false_positives))
    false_negatives = sorted(
        set(actions) - scenario.actions,
        key=object_verb_tool_key,
    )
    print('    False Negatives: {} {}'.format(len(false_negatives), false_negatives))


def main():
    """Evaluate all scenarios."""
    model = load_model(GOOGLE_NEWS_MODEL_PATH)
    for scene_num in range(1, 29):
        run_scenario(scene_num, model)

if __name__ == '__main__':
    main()
