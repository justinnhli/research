"""A module to handle local and remote knowledge bases."""

from os.path import exists as file_exists, splitext as split_ext, expanduser, realpath

from SPARQLWrapper import SPARQLWrapper2
from rdflib import Graph, Literal, URIRef, plugin
from rdflib.store import Store
from rdflib.util import guess_format
from rdflib_sqlalchemy import registerplugins

registerplugins()


class URI:
    """A class to represent URIs and their namespaces."""

    NAMESPACES = {
        '_': '_',
        'db': 'http://dbpedia.org/',
        'dbc': 'http://dbpedia.org/resource/Category:',
        'dbo': 'http://dbpedia.org/ontology/',
        'dbp': 'http://dbpedia.org/property/',
        'dbr': 'http://dbpedia.org/resource/',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'dct': 'http://purl.org/dc/terms/',
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
    PREFIXES = sorted(
        ([namespace, prefix] for prefix, namespace in NAMESPACES.items()),
        key=(lambda kv: -len(kv[0])),
    )

    def __init__(self, uri, namespace=None):
        """Construct the URI.

        Arguments:
            uri (str): The full URI, or the after-prefix fragment.
            namespace (str): The namespace. Defaults to None.
        """
        if namespace:
            self.namespace = namespace
            self.prefix = URI.NAMESPACES[namespace]
            self.fragment = uri
            self.uri = self.prefix + self.fragment
        else:
            self.namespace = None
            self.prefix = None
            self.fragment = None
            self.uri = uri
            prefix_order = sorted(
                URI.NAMESPACES.items(),
                key=(lambda kv: len(kv[1])),
                reverse=True,
            )
            # pylint: disable = redefined-argument-from-local
            for namespace, prefix in prefix_order:
                if uri.startswith(prefix):
                    self.namespace = namespace
                    self.prefix = prefix
                    self.fragment = uri[len(prefix):]
                    break

    def __str__(self):
        return self.uri

    @property
    def short_str(self):
        """Get the namespace:fragment representation of this URI.

        Returns:
            str: The prefixed string.

        Raises:
            ValueError: If no namespace was specified or found.
        """
        if self.namespace:
            return self.namespace + ':' + self.fragment
        else:
            raise ValueError('No namespace found for URI: ' + self.uri)


class KnowledgeSource:
    """Abstract class to represent a knowledge source."""

    def query_sparql(self, sparql):
        """Query the KB with SPARQL.

        Arguments:
            sparql (str): The SPARQL query.

        Returns:
            list[dict[str:str]]: A list of variable bindings.
        """
        raise NotImplementedError()


class KnowledgeFile(KnowledgeSource):
    """A knowledge base in a local file."""

    def __init__(self, source=None, kb_name='rdflib_test', sqlize=True):
        """Construct the KnowledgeFile.

        Arguments:
            source (str): Path to the knowledge base. If None, an in-memory
                knowledge base will be created. Defaults to None.
            kb_name (str): The name of the knowledge base. This must match the
                name used to create the knowledge base, if the source is an
                rdfsqlite file. Defaults to 'rdflib_test'.
            sqlize (bool): Whether to create a sqlite version of the knowledge
                base for faster future access. Defaults to True.

        Raises:
            FileNotFoundError: If the specified source is not found.
            ValueError: If the format of the source cannot be determined.
        """
        super().__init__()
        ident = URIRef(kb_name)
        store = plugin.get('SQLAlchemy', Store)(identifier=ident)
        self.graph = Graph(store, identifier=ident)
        if source is None:
            self.graph.open(Literal('sqlite://'))
            return
        source = realpath(expanduser(source))
        if not file_exists(source):
            raise FileNotFoundError(source)
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

    def query_sparql(self, sparql): # noqa: D102
        results = []
        for result in self.graph.query(sparql).bindings:
            results.append({str(variable):str(uri) for variable, uri in result.items()})
        return results


class SparqlEndpoint(KnowledgeSource):
    """A knowledge base from a remote SPARQL endpoint."""

    def __init__(self, url):
        """Construct the SparqlEndpoint.

        Arguments:
            url (str): The URL to the SPARQL endpoint.
        """
        self.endpoint = SPARQLWrapper2(url)

    def query_sparql(self, sparql): # noqa: D102
        self.endpoint.setQuery(sparql)
        results = []
        for bindings in self.endpoint.query().bindings:
            results.append({key:value.value for key, value in bindings.items()})
        return results
