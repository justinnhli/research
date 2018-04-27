#!/usr/bin/env python3

import logging
import re
from os.path import exists as file_exists, splitext as split_ext, expanduser, realpath
from textwrap import indent, dedent

from SPARQLWrapper import SPARQLWrapper2, N3
from rdflib import Graph, Literal, URIRef, plugin
from rdflib.plugins.sparql import prepareQuery
from rdflib.store import Store
from rdflib.util import guess_format
from rdflib_sqlalchemy import registerplugins
from rdflib_sqlalchemy.store import SQLAlchemy
from sqlalchemy import create_engine

registerplugins()


class URI:
    PREFIXES = {
        '_': '_',
        'db': 'http://dbpedia.org/',
        'dbo': 'http://dbpedia.org/ontology/',
        'dbp': 'http://dbpedia.org/property/',
        'dbr': 'http://dbpedia.org/resource/',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'foaf': 'http://xmlns.com/foaf/0.1/',
        'owl': 'http://www.w3.org/2002/07/owl#',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
        'res': 'http://www.w3.org/2005/sparql-results#',
        'skos': 'http://www.w3.org/2004/02/skos/core#',
        'xsd': 'http://www.w3.org/2001/XMLSchema#',
        'umbel-rc': 'http://umbel.org/umbel/rc/',
        'umbel': 'http://umbel.org/umbel#',
    }
    REVERSED = None

    def __init__(self, uri, prefix=None):
        """Example function with types documented in the docstring.

        Arguments:
            uri (str): Either the fully qualified URI, or the fragment that comes after the prefix
            prefix (str): The second parameter.
        """
        if prefix:
            self.prefix = prefix
            self.fragment = uri
            self.uri = URI.PREFIXES[prefix] + uri
        else:
            self.uri = uri
            if URI.REVERSED is None:
                URI.REVERSED = sorted(
                    ([namespace, prefix] for prefix, namespace in URI.PREFIXES.items()),
                    key=(lambda kv: -len(kv[0])),
                )
            for namespace, prefix in URI.REVERSED:
                if uri.startswith(namespace):
                    self.prefix = prefix
                    self.fragment = uri[len(namespace):]

    def __str__(self):
        return '<' + self.uri + '>'

    @property
    def short_str(self):
        return self.prefix + ':' + self.fragment


class Node:
    IS_VARIABLE = False

    def __init__(self, name, **props):
        self.name = name
        self.props = props

    def to_triple_list(self):
        my_triples = []
        child_triples = []
        for k, child in self.props.items():
            if not isinstance(child, Node):
                child = L(child)
            prop = URI(*reversed(k.split('__', maxsplit=1)))
            my_triples.append('{} {} {} .'.format(self, prop, child))
            child_triples.extend(child.to_triple_list())
        return my_triples + child_triples

    def variables(self):
        seen = set(self.name)
        variables = []
        if self.IS_VARIABLE:
            variables.append(str(self))
        for v in self.props.values():
            if isinstance(v, str):
                desc_vars = ['?{}'.format(v)]
            else:
                desc_vars = v.variables()
            desc_vars = [dv for dv in desc_vars if dv not in seen]
            seen |= set(desc_vars)
            variables.extend(desc_vars)
        return variables

    @staticmethod
    def from_str(string):
        if string[0] == '<' and string[-1] == '>':
            return U(string[1:-1])
        elif string.startswith('"'):
            while not (string[0] == '"' and string[-1] == '"'):
                string = re.sub(r'^(".*")@[a-z]*$', r'\1', string)
                string = re.sub(r'^(".*")\^\^[^"]*$', r'\1', string)
            return L(string[1:-1])
        else:
            try:
                return L(int(string))
            except ValueError:
                try:
                    return L(float(string))
                except ValueError:
                    return U(*reversed(string.split(':', maxsplit=1)))


class V(Node):
    IS_VARIABLE = True

    def __str__(self):
        return '?{}'.format(self.name)


class U(Node):

    def __init__(self, uri, prefix=None, **props):
        self.uri = URI(uri, prefix)
        super().__init__(self.uri.uri, **props)

    def __str__(self):
        return str(self.uri)

    @property
    def short_str(self):
        return self.uri.short_str


class L(Node):

    def __init__(self, val):
        super().__init__(val)
        assert isinstance(val, (int, float, str))

    def __str__(self):
        if isinstance(self.name, str):
            return '"{}"'.format(self.name.replace('"', '\\"'))
        else:
            return str(self.name)

    def variables(self):
        return []


class Query:

    def __init__(self, *paths):
        self.paths = paths

    def variables(self, bound_vars=None):
        if bound_vars is None:
            bound_vars = []
        result = []
        for path in self.paths:
            for variable in path.variables():
                if variable[1:] not in bound_vars:
                    result.append(variable)
        return result
        #return [variable for path in self.paths for variable in path.variables()]

    def variable_names(self, bound_vars=None):
        return [variable.strip('?') for variable in self.variables(bound_vars)]

    def to_triples_str(self, **bindings):
        triples = '\n'.join(statement for path in self.paths for statement in path.to_triple_list())
        for variable, value in bindings.items():
            assert isinstance(value, Node), value
            triples = triples.replace('?{} '.format(variable), '{} '.format(value))
        return triples

    def to_triples_list(self, **bindings):
        return self.to_triples_str(**bindings).splitlines()

    def to_select(self, *constraints, **bindings):
        template = dedent('''
            SELECT {variables}
            WHERE {{
            {triples}
            }}
            {constraints}
        ''').strip()
        variables = self.variables()
        for bound in bindings.keys():
            if '?' + bound in variables:
                variables.remove('?' + bound)
        return template.format(
            variables=' '.join(variables),
            triples=indent(self.to_triples_str(**bindings), '    '),
            constraints='\n'.join(constraints)
        )

    def as_prepared_query(self, *constraints, **bindings):
        return prepareQuery(self.to_select(*constraints, **bindings))


def create_sqlite_graph(path, create=True, identifier=None):
    """Creates a sqlite-backed graph at the given path

    Args:
        path (str): Either the fully qualified URI, or the fragment that comes after the prefix
        create (bool): If True, create the path if it doesn't exist. Defaults to True.
        identifier (str): The identifier of the graph. Defaults to 'rdflib_sqlalchemy_graph'

    Returns:
        an RDF Graph that uses the specified sqlite-db at the path
    """
    if identifier is None:
        identifier = 'rdflib_sqlalchemy_graph'
    identifier = URIRef(identifier)
    store = plugin.get("SQLAlchemy", Store)(identifier=identifier)
    graph = Graph(store, identifier=identifier)
    graph.open(Literal('sqlite:///' + realpath(expanduser(path))))
    return graph


class KnowledgeSource:

    def query_sparql(self, sparql):
        raise NotImplementedError

    def query(self, query, *constraints, **bindings):
        results = self.query_sparql(query.to_select(*constraints, **bindings))
        result_graph = Graph()
        result_graph.parse(data=results, format='n3')
        variable_paths = []
        for variable in query.variable_names():
            variable = variable.strip('?')
            variable_paths.append(
                V(
                    'solution',
                    res__binding=V(
                        variable + '_binding',
                        res__variable=L(variable),
                        res__value=V(variable)
                    ),
                )
            )
        res_qry = Query(V('root', res__solution=V('solution')), *variable_paths)
        triples = set()
        for values in result_graph.query(res_qry.to_select()):
            val_dict = dict(zip(res_qry.variable_names(), values))
            bindings = {}
            for variable, value in val_dict.items():
                value = value.n3()
                bindings[variable] = Node.from_str(value)
            triples |= set(query.to_triples_str(**bindings).split('\n'))
        triples = '\n'.join(sorted(triples))
        #self.graph.parse(data=prefixes + '\n' + triples, format='n3')
        return triples


class KnowledgeFile(KnowledgeSource):

    def __init__(self, source=None, kb_name=None, sqlize=True):
        super().__init__()
        if kb_name is None:
            ident = URIRef('rdflib_test')
        else:
            ident = URIRef(kb_name)
        store = plugin.get('SQLAlchemy', Store)(identifier=ident)
        self.graph = Graph(store, identifier=ident)
        if source is None:
            self.graph.open(Literal('sqlite://'))
            return
        source = realpath(expanduser(source))
        if not file_exists(source):
            raise OSError('Cannot find file {}'.format(source))
        filepath, ext = split_ext(source)
        rdf_format = guess_format(source)
        if rdf_format == 'turtle':
            rdf_format = 'nt'
        if rdf_format is not None:
            if sqlize:
                sql_uri = 'sqlite:///' + filepath + '.rdfsqlite'
            else:
                sql_uri = 'sqlite://'
            self.graph.open(Literal(sql_uri), create=True)
            self.graph.parse(source, format=rdf_format)
        elif ext[1:] in ['db', 'sqlite', 'rdfsqlite']:
            sql_uri = 'sqlite:///' + source
            self.graph.open(Literal(sql_uri))
        else:
            raise ValueError('Cannot determine format of {}'.format(source))

    def __del__(self):
        self.graph.commit()
        self.graph.close()

    def query_sparql(self, sparql):
        results = []
        for result in self.graph.query(sparql).bindings:
            results.append({str(variable):str(uri) for variable, uri in result.items()})
        return results

    def query(self, query, *constraints, **bindings):
        results = self.query_sparql(query.to_select(*constraints, **bindings))
        variables = query.variable_names(list(bindings.keys()))
        triples = set()
        for row in results:
            val_dict = dict(zip(variables, row))
            result_bindings = {}
            for variable, value in val_dict.items():
                result_bindings[variable] = Node.from_str(value.n3())
            triples |= set(query.to_triples_str(**bindings, **result_bindings).split('\n'))
        triples = '\n'.join(sorted(triples))
        return triples


class SparqlEndpoint(KnowledgeSource):

    def __init__(self, url):
        self.endpoint = SPARQLWrapper2(url)

    def query_sparql(self, sparql):
        self.endpoint.setQuery(sparql)
        results = []
        for bindings in self.endpoint.query().bindings:
            results.append({key:value.value for key, value in bindings.items()})
        return results

    def query(self, query, *constraints, **bindings):
        return self.query_sparql(query.to_select(*constraints, **bindings))
