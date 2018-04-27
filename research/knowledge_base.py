#!/usr/bin/env python3

from os.path import exists as file_exists, splitext as split_ext, expanduser, realpath

from SPARQLWrapper import SPARQLWrapper2
from rdflib import Graph, Literal, URIRef, plugin
from rdflib.store import Store
from rdflib.util import guess_format
from rdflib_sqlalchemy import registerplugins

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
    REVERSED = sorted( 
        ([namespace, prefix] for prefix, namespace in PREFIXES.items()), 
        key=(lambda kv: -len(kv[0])), 
    ) 
 
    def __init__(self, uri, prefix=None): 
        if prefix: 
            self.prefix = prefix 
            self.fragment = uri 
            self.uri = URI.PREFIXES[prefix] + uri 
        else: 
            self.uri = uri 
            for namespace, alias in URI.REVERSED: 
                if uri.startswith(namespace): 
                    self.prefix = alias 
                    self.fragment = uri[len(namespace):] 
 
    def __str__(self): 
        return '<' + self.uri + '>' 
 
    @property 
    def short_str(self): 
        return self.prefix + ':' + self.fragment 


def create_sqlite_graph(path, create=True, identifier=None):
    """Creates a sqlite-backed graph at the given path

    Args:
        path (str): Either the fully qualified URI, or the fragment that comes after the prefix
        create (bool): If True, create the path if it doesn't exist. Defaults to True.
        identifier (str): The identifier of the graph. Defaults to 'rdflib_sqlalchemy_graph'

    Returns:
        Graph: An RDF Graph that uses the specified sqlite-db at the path
    """
    if identifier is None:
        identifier = 'rdflib_sqlalchemy_graph'
    identifier = URIRef(identifier)
    store = plugin.get("SQLAlchemy", Store)(identifier=identifier)
    graph = Graph(store, identifier=identifier)
    graph.open(Literal('sqlite:///' + realpath(expanduser(path))))
    return graph


class KnowledgeSource:
    """Abstract class to represent a knowledge source."""
    # pylint: disable=redundant-returns-doc, missing-raises-doc

    def query_sparql(self, sparql):
        """Query the KB with SPARQL.

        Arguments:
            sparql (str): The SPARQL query.

        Returns:
            list[dict[str:str]]: A list of variable bindings.
        """
        raise NotImplementedError


class KnowledgeFile(KnowledgeSource):

    def __init__(self, source=None, kb_name='rdflib_test', sqlize=True):
        super().__init__()
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


class SparqlEndpoint(KnowledgeSource):

    def __init__(self, url):
        self.endpoint = SPARQLWrapper2(url)

    def query_sparql(self, sparql):
        self.endpoint.setQuery(sparql)
        results = []
        for bindings in self.endpoint.query().bindings:
            results.append({key:value.value for key, value in bindings.items()})
        return results
