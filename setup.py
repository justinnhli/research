from setuptools import setup

setup(
    name='research',
    version='',

    description='',
    long_description='',
    license='MIT',

    author='Justin Li',
    author_email='justinnhli@oxy.edu',
    url='https://github.com/justinnhli/research']
    entry_points={
        'console_scripts': [
            'rdfsqlize = research.rdfsqlize:main',
        ]
    },
)
