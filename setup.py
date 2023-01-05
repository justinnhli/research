"""Setup for package."""

import re
from collections import namedtuple
from urllib.parse import urlsplit
from typing import Tuple, List, Dict

from setuptools import setup

GitRequirement = namedtuple('GitRequirement', 'url, name')


def read_requirements():
    # type: () -> Tuple[List[str], Dict[str, GitRequirement]]
    """Read the requirements.txt for this project.

    Returns:
        List[str]: The PyPI requirements.
        Dict[str, GitRequirement]: The git requirements.

    Raises:
        ValueError: If parsing the requirements.txt causes an error.
    """
    with open('requirements.txt', encoding='utf-8') as fd:
        requirements = fd.read().splitlines()
    pypi_requirements = []
    git_requirements = {}
    for requirement in requirements:
        requirement = requirement.strip()
        if requirement.startswith('#'):
            continue
        requirement = re.sub(r'\s+#.*', '', requirement)
        if requirement.startswith('git+'):
            match = re.fullmatch(r'git\+(?P<url>[^#]*)(#egg=(?P<name>.*))?', requirement)
            if not match:
                raise ValueError(f'unable to parse requirement: {requirement}')
            match_dict = match.groupdict()
            if 'name' in match_dict:
                name = urlsplit(match.group('url')).path.split('/')[-1]
                if name.endswith('.git'):
                    name = name[:-4]
                match_dict['name'] = name
            git_requirements[match_dict['name']] = GitRequirement(**match_dict)
        else:
            pypi_requirements.append(requirement)
    return pypi_requirements, git_requirements


def main():
    # type: () -> None
    """Install the package."""
    pypi_requirements, git_requirements = read_requirements()
    setup(
        name='research',
        version='',
        description="Justin Li's main research code repository.",
        license='MIT',
        author='Justin Li',
        author_email='justinnhli@oxy.edu',
        url='https://github.com/justinnhli/research',
        packages=['research',],
        entry_points={
            'console_scripts': [
                'rdfsqlize = research.rdfsqlize:main',
            ],
        },
        install_requires=[
            *pypi_requirements,
            *(
                f'{requirement.name} @ git+{requirement.url}'
                for requirement in git_requirements.values()
            ),
        ],
        dependency_links=[
            *(
                f'{requirement.url}#egg={requirement.name}'
                for requirement in git_requirements.values()
            ),
        ],
    )


if __name__ == '__main__':
    main()
