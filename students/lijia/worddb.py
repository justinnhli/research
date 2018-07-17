import sqlite3
from os.path import exists as file_exists


class CondProbDict:

    def __init__(self, path):
        self.path = path
        self._connection = None
        if not file_exists(self.path):
            self._create_table()

    @property
    def connection(self):
        if self._connection is None:
            self._connection = sqlite3.connect(self.path)
        return self._connection

    def _create_table(self):
        with self.connection as conn:
            conn.execute('''
                CREATE TABLE probabilities (
                    id INTEGER PRIMARY KEY,
                    given text,
                    variable text,
                    probability real,
                    CONSTRAINT unique_key UNIQUE (given, variable)
                )
            ''')
            conn.execute('''
                CREATE INDEX given_lookup ON probabilities (
                    given, variable
                )
            ''')
            conn.execute('''
                CREATE INDEX variable_lookup ON probabilities (
                    variable, given
                )
            ''')

    def __iter__(self):
        with self.connection as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT given, variable FROM probabilities')
            for given, variable in cursor:
                yield given, variable

    def add_probability(self, given, variable, probability, update=False):
        with self.connection as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT probability FROM probabilities WHERE given=? AND variable=?',
                (given, variable),
            )
            result = cursor.fetchone()
            if result is None:
                conn.execute(
                    '''
                        INSERT INTO probabilities (
                            given, variable, probability
                        ) VALUES (?, ?, ?)
                    ''',
                    (given, variable, probability),
                )
            elif update:
                old_probability = result[0]
                conn.execute(
                    '''
                        UPDATE probabilities SET probability = ?
                        WHERE given=? AND variable=?
                    ''',
                    (old_probability + probability, given, variable),
                )
            else:
                assert False, "entry already exist"

    def get_probability(self, given, variable):
        with self.connection as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT probability FROM probabilities WHERE given=? AND variable=? LIMIT 1',
                (given, variable),
            )
            return cursor.fetchone()[0]

    def get_given_dict(self, given):
        result = {}
        with self.connection as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT variable, probability FROM probabilities WHERE given=?',
                (given,),
            )
            for variable, probability in cursor:
                result[variable] = probability
        return result

    def get_variable_dict(self, variable):
        result = {}
        with self.connection as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT given, probability FROM probabilities WHERE variable=?',
                (variable,),
            )
            for given, probability in cursor:
                result[given] = probability
        return result


def test():
    from tempfile import mkdtemp
    from shutil import rmtree
    from os.path import join as join_path
    temp_dir = mkdtemp()
    temp_file = join_path(temp_dir, 'temp.sqlite')
    d = CondProbDict(temp_file)
    d.add_probability('hello', 'world', 0.25)
    d.add_probability('hello', 'world', 0.25, update=True)
    d.add_probability('disney', 'world', 0.25)
    expected = 0.5
    actual = d.get_probability('hello', 'world')
    assert expected == actual, f'expected {expected} but got {actual}'
    assert d.get_given_dict('hello') == {'world': 0.5}
    assert d.get_variable_dict('world') == {'hello': 0.5, 'disney': 0.25}
    rmtree(temp_dir)


if __name__ == '__main__':
    test()
