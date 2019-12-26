from collections import namedtuple

from students.lijia.extraction import extract_np, extract_vpo

TestCase = namedtuple('TestCase', ['sentence', 'expected_output'])
VPO = namedtuple('VPO', ('verb', 'prep', 'object'))
NP = namedtuple('NP', ['noun', 'adjectives'])

def test_extract_vpo():
    """ test cases for extracting vpo"""
    test_cases = [
        TestCase(
            "Jack hits me with ball",
            [VPO(verb='hit', prep='with', object='ball')],
        ),
        TestCase(
            "Jonney enjoys sewing the little blanket with needle and thread",
            [VPO(verb='sew', prep=None, object='blanket'), VPO(verb='sew', prep='with', object='needle'),
             VPO(verb='sew', prep='with', object='thread')]
        ),
    ]
    for test_case in test_cases:
        message = [
            "Parsing sentence: " + test_case.sentence,
            "    but failed to see expected result: " + str(test_case.expected_output),
        ]
        assert test_case.expected_output == extract_vpo(nlp(test_case.sentence)), "\n".join(message)


def test_extract_np():
    """ test cases for extracting vpo"""
    test_cases = [
        TestCase(
            "The tall and skinny girl kiss Jack",
            [NP(noun='girl', adjectives=['tall', 'skinny'])],
        ),
        TestCase(
            "The girl is tall and skinny.",
            [NP(noun='girl', adjectives=['tall', 'skinny'])],
        ),
    ]
    for test_case in test_cases:
        message = [
            "Parsing sentence: " + test_case.sentence,
            "    but failed to see expected result: " + str(test_case.expected_output),
        ]
        assert test_case.expected_output == extract_np(nlp(test_case.sentence)), "\n".join(message)


def main():  # pylint: disable= missing-docstring
    """implements test methods"""
    test_extract_vpo()
    test_extract_np()
    print("passed all test cases")

if __name__ == '__main__':
    main()