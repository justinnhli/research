"""Setup for Justin's research code."""

from setuptools import setup

DEPENDENCIES = {
    'pydictionary': 'https://github.com/justinnhli/PyDictionary.git',
    'permspace': 'https://github.com/justinnhli/permspace.git',
    'clusterun': 'https://github.com/justinnhli/clusterun.git',
}


def get_dependency(package, location):
    if location == 'install':
        return f'{package} @ git+{DEPENDENCIES[package]}'
    elif location == 'link':
        return f'{DEPENDENCIES[package]}#egg={package}'
    else:
        raise ValueError(f'Unknown location: {location}')


setup(
    name='justinnhli-research',
    version='',

    description='',
    long_description='',
    license='MIT',

    author='Justin Li',
    author_email='justinnhli@oxy.edu',
    url='https://github.com/justinnhli/research',
    entry_points={
        'console_scripts': [
            'rdfsqlize = research.rdfsqlize:main',
        ]
    },
    install_requires=[
        # knowledge base packages
        'rdflib==4.2.2',
        'rdflib-sqlalchemy==0.4.0',
        'SPARQLWrapper==1.8.4',
        # word embedding packages
        'gensim==3.8.1',
        # NLP packages
        'spacy==2.2.3',
        'nltk==3.4.5',
        # Justin's less chatty fork of pydictionary
        get_dependency('pydictionary', location='install'),
        # jupyter notebook packages
        'jupyter==1.0.0',
        'bokeh==1.4.0',
        'pandas==0.25.3',
        'numpy==1.17.4',
        # utility packages
        'SQLAlchemy==1.3.11',
        'requests==2.22.0',
        'networkx==2.4',
        # experiment packages
        get_dependency('permspace', location='install'),
        get_dependency('clusterun', location='install'),
        # code quality packages
        'pylint==2.4.4',
        'pydocstyle==4.0.1',
        'pytest-cov==2.8.1',
    ],
    dependency_links=[
        get_dependency('pydictionary', location='link'),
        get_dependency('permspace', location='link'),
        get_dependency('clusterun', location='link'),
    ],
)
