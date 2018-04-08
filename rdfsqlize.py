#!/usr/bin/env python3
"""A script to convert RDF files to SQL dumps."""

import sqlite3
from os import remove
from os.path import exists as file_exists
from hashlib import sha1
from textwrap import dedent

from .kb import URI

TRANSACTION_SQL_HEADER = '''
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
'''.strip()

CREATE_TABLES_SQL = '''
CREATE TABLE {interned_id}_asserted_statements (
	id INTEGER NOT NULL,
	subject TEXT NOT NULL,
	predicate TEXT NOT NULL,
	object TEXT NOT NULL,
	context TEXT NOT NULL,
	termcomb INTEGER NOT NULL,
	PRIMARY KEY (id)
);
CREATE TABLE {interned_id}_type_statements (
	id INTEGER NOT NULL,
	member TEXT NOT NULL,
	klass TEXT NOT NULL,
	context TEXT NOT NULL,
	termcomb INTEGER NOT NULL,
	PRIMARY KEY (id)
);
CREATE TABLE {interned_id}_literal_statements (
	id INTEGER NOT NULL,
	subject TEXT NOT NULL,
	predicate TEXT NOT NULL,
	object TEXT,
	context TEXT NOT NULL,
	termcomb INTEGER NOT NULL,
	objlanguage VARCHAR(255),
	objdatatype VARCHAR(255),
	PRIMARY KEY (id)
);
CREATE TABLE {interned_id}_quoted_statements (
	id INTEGER NOT NULL,
	subject TEXT NOT NULL,
	predicate TEXT NOT NULL,
	object TEXT,
	context TEXT NOT NULL,
	termcomb INTEGER NOT NULL,
	objlanguage VARCHAR(255),
	objdatatype VARCHAR(255),
	PRIMARY KEY (id)
);
CREATE TABLE {interned_id}_namespace_binds (
	prefix VARCHAR(20) NOT NULL,
	uri TEXT,
	PRIMARY KEY (prefix),
	UNIQUE (prefix)
);
'''.strip()

CREATE_INDICES_SQL = '''
CREATE INDEX "{interned_id}_A_p_index" ON {interned_id}_asserted_statements (predicate);
CREATE INDEX "{interned_id}_A_c_index" ON {interned_id}_asserted_statements (context);
CREATE INDEX "{interned_id}_A_s_index" ON {interned_id}_asserted_statements (subject);
CREATE INDEX "{interned_id}_A_termComb_index" ON {interned_id}_asserted_statements (termcomb);
CREATE INDEX "{interned_id}_A_o_index" ON {interned_id}_asserted_statements (object);
CREATE INDEX {interned_id}_member_index ON {interned_id}_type_statements (member);
CREATE INDEX {interned_id}_c_index ON {interned_id}_type_statements (context);
CREATE INDEX {interned_id}_klass_index ON {interned_id}_type_statements (klass);
CREATE INDEX "{interned_id}_T_termComb_index" ON {interned_id}_type_statements (termcomb);
CREATE INDEX "{interned_id}_L_p_index" ON {interned_id}_literal_statements (predicate);
CREATE INDEX "{interned_id}_L_termComb_index" ON {interned_id}_literal_statements (termcomb);
CREATE INDEX "{interned_id}_L_s_index" ON {interned_id}_literal_statements (subject);
CREATE INDEX "{interned_id}_L_c_index" ON {interned_id}_literal_statements (context);
CREATE INDEX "{interned_id}_Q_p_index" ON {interned_id}_quoted_statements (predicate);
CREATE INDEX "{interned_id}_Q_c_index" ON {interned_id}_quoted_statements (context);
CREATE INDEX "{interned_id}_Q_s_index" ON {interned_id}_quoted_statements (subject);
CREATE INDEX "{interned_id}_Q_o_index" ON {interned_id}_quoted_statements (object);
CREATE INDEX "{interned_id}_Q_termComb_index" ON {interned_id}_quoted_statements (termcomb);
CREATE INDEX {interned_id}_uri_index ON {interned_id}_namespace_binds (uri);
'''.strip()

TRANSACTION_SQL_FOOTER = '''
COMMIT;
'''.strip()

# Taken from rdflib_sqlalchemy.constants.py
INTERNED_PREFIX = "kb_"

def generate_interned_id(identifier):
    """Generate an ID for this KB.

    Taken from rdflib_sqlalchemy.store.py

    Arguments:
        identifier (str): The name given to this KB.

    Returns:
        str: A short hash of the identifier.
    """
    return "{prefix}{identifier_hash}".format(
        prefix=INTERNED_PREFIX,
        identifier_hash=sha1(identifier.encode('utf8')).hexdigest()[:10],
    )


def standardize_uri(uri):
    """Convert a bracketed/prefixed URI to long form.

    Arguments:
        uri (str): Any URI, optionally bracketed or prefixed.


    Returns:
        str: The URI in long form.
    """
    if uri.startswith('<') and uri.endswith('>'):
        uri = URI(uri[1:-1])
    elif ':' in uri:
        prefix, identifier = uri.split(':', maxsplit=1)
        uri = URI(identifier, prefix)
    return uri.uri


class RDFSQLizer:
    """A class to convert triples into a SQL dump."""

    def __init__(self):
        """Initialize and reset the function."""
        self.type_id = 1
        self.triple_id = 1
        self.kb_id = ''
        self.interned_id = ''
        self._reset()

    def sqlize(self, filepath, kb_id, sql_file):
        """Convert an RDF file into a SQL dump.

        Arguments:
            filepath (str): Path to the RDF file.
            kb_id (str): The name of the KB and the output filename.
            sql_file (str): The output filename to save to.
        """
        self._reset()
        self.kb_id = kb_id
        self.interned_id = generate_interned_id(self.kb_id)
        with open(sql_file, 'w') as out_fd:
            with open(filepath) as in_fd:
                out_fd.write(TRANSACTION_SQL_HEADER + '\n')
                out_fd.write('\n')
                out_fd.write(CREATE_TABLES_SQL.format(interned_id=self.interned_id) + '\n')
                out_fd.write('\n')
                out_fd.write(CREATE_INDICES_SQL.format(interned_id=self.interned_id) + '\n')
                out_fd.write('\n')
                for sql in self._populate_namespaces():
                    out_fd.write(sql + '\n')
                out_fd.write('\n')
                for line in in_fd.readlines():
                    out_fd.write(self._dispatch_nt_line(line.strip()) + '\n')
                out_fd.write('\n')
                out_fd.write(TRANSACTION_SQL_FOOTER + '\n')

    def _reset(self):
        """Reset the function."""
        self.type_id = 1
        self.triple_id = 1
        self.kb_id = ''
        self.interned_id = ''

    def _populate_namespaces(self):
        """Generate the SQL dump for namespaces.

        Yields:
            str: The SQL insert statement for a namespace.
        """
        sql_template = dedent('''
            INSERT INTO {interned_id}_namespace_binds
            VALUES({prefix},{uri});
        ''').strip().replace('\n', ' ')
        for prefix, uri in URI.PREFIXES.items():
            if not uri.startswith('http'):
                continue
            yield sql_template.format(
                interned_id=self.interned_id,
                prefix=repr(prefix),
                uri=repr(uri),
            )

    def _dispatch_nt_line(self, line):
        """Generate the SQL dump for statements.

        Arguments:
            line (str): The N3 line to convert.

        Returns:
            str: The SQL insert statements.
        """
        assert line.endswith(' .')
        line = line[:-2]
        parent, relation, child = line.split(' ', maxsplit=2)
        parent = standardize_uri(parent)
        child = standardize_uri(child)
        if relation == 'a':
            return self._sqlize_nt_type(parent, child)
        else:
            relation = standardize_uri(relation)
            return self._sqlize_nt_triple(parent, relation, child)

    def _sqlize_nt_type(self, instance, classname):
        """Generate the SQL dump for type statements.

        Arguments:
            instance (str): The instance URI.
            classname (str): The classname URI.

        Returns:
            str: The SQL insert statement.
        """
        sql_template = dedent('''
            INSERT INTO {interned_id}_type_statements
            VALUES({id},{instance},{classname},{identifier},0);
        ''').strip().replace('\n', ' ')
        result = sql_template.format(
            interned_id=self.interned_id,
            id=self.type_id,
            instance=repr(instance),
            classname=repr(classname),
            identifier=repr(self.kb_id),
        )
        self.type_id += 1
        return result

    def _sqlize_nt_triple(self, parent, relation, child):
        """Generate the SQL dump for triple statements.

        Arguments:
            parent (str): The parent URI.
            relation (str): The relation URI.
            child (str): The child URI.

        Returns:
            str: The SQL insert statement.
        """
        sql_template = dedent('''
            INSERT INTO {interned_id}_asserted_statements
            VALUES({id},{parent},{relation},{child},{identifier},0);
        ''').strip().replace('\n', ' ')
        result = sql_template.format(
            interned_id=self.interned_id,
            id=self.triple_id,
            parent=repr(parent),
            relation=repr(relation),
            child=repr(child),
            identifier=repr(self.kb_id),
        )
        self.triple_id += 1
        return result


def read_dump(sql_path, db_path):
    """Read SQL into a SQLite file.

    Arguments:
        sql_path (str): Path to the input SQL file.
        db_path (str): Path to the output SQLite file.
    """
    conn = sqlite3.connect(db_path)
    with open(sql_path) as fd:
        dump = fd.read()
    conn.executescript(dump)
    conn.commit()
    conn.close()


def sqlize(rdf_file, kb_name, binary=True):
    """Convert an RDF file into a SQL dump.

    Arguments:
        rdf_file (str): Path to the input RDF file.
        kb_name (str): The name of the KB. Used as the stem for the output filename.
        binary (bool): Whether to save to a binary SQLite file. Defaults to True.

    Returns:
        str: Filename of the output file.

    Raises:
        FileExistsError: If any intermediate files already exist.
    """
    sql_file = kb_name + '.sql'
    if file_exists(sql_file):
        raise FileExistsError(sql_file)
    RDFSQLizer().sqlize(rdf_file, kb_name, sql_file)
    if binary:
        rdfsqlite_file = kb_name + '.rdfsqlite'
        if file_exists(rdfsqlite_file):
            raise FileExistsError(sql_file)
        read_dump(sql_file, rdfsqlite_file)
        remove(sql_file)
        return rdfsqlite_file
    else:
        return sql_file


def main():
    """Provide a CLI command to convert RDF files."""
    import sys
    if len(sys.argv) not in [3, 4]:
        print('usage: {} [--sql] <rdf_file> <kb_name>')
        exit(1)
    if len(sys.argv) == 4 and sys.argv[1] != '--sql':
        print('usage: {} [--sql] <rdf_file> <kb_name>')
        exit(1)
    rdf_file = sys.argv[-2]
    kb_name = sys.argv[-1]
    sqlize(rdf_file, kb_name, binary=(len(sys.argv) != 4))


if __name__ == '__main__':
    main()
