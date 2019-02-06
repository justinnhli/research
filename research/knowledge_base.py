"""A module to handle local and remote knowledge bases."""

from os.path import exists as file_exists, splitext as split_ext, expanduser, realpath

from SPARQLWrapper import SPARQLWrapper2
from SPARQLWrapper.SmartWrapper import Value as SparqlValue
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed
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
        """Initialize the URI.

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
            list[dict[str:Value]]: A list of variable bindings.
        """
        raise NotImplementedError()


class KnowledgeFile(KnowledgeSource):
    """A knowledge base in a local file."""

    def __init__(self, source=None, kb_name='rdflib_test', sqlize=True):
        """Initialize the KnowledgeFile.

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
            results.append({str(variable): str(uri) for variable, uri in result.items()})
        return results


class Value:
    """Wrapper around SPARQLWrapper.SmartWrapper.Value."""

    def __init__(self, sparql_value):
        """Initialize a Value.

        Arguments:
            sparql_value (SPARQLWrapper.SmartWrapper.Value):
                The original value.
        """
        self.sparql_value = sparql_value

    @property
    def is_uri(self):
        """Whether this is a URI node.

        Returns:
            bool: True if this is a URI node.
        """
        return self.sparql_value.type == SparqlValue.URI

    @property
    def is_literal(self):
        """Whether this is a literal node.

        Returns:
            bool: True if this is a literal node.
        """
        return self.sparql_value.type in (SparqlValue.Literal, SparqlValue.TypedLiteral)

    @property
    def is_blank(self):
        """Whether this is a blank node.

        Returns:
            bool: True if this is a blank node.
        """
        return self.sparql_value.type == SparqlValue.BNODE

    @property
    def uri(self):
        """The URI of this URI node.

        Returns:
            str: The URI of this URI node.

        Raises:
            ValueError: If this is not a URI node.
        """
        if not self.is_uri:
            raise ValueError('Value is not a URI')
        return self.sparql_value.value

    @property
    def value(self):
        """The value of this literal node.

        Returns:
            Union[int,float,str]: The value of this literal node.

        Raises:
            ValueError: If this is not a literal node.
        """
        if not self.is_literal:
            raise ValueError('Value is not a literal')
        return self.sparql_value.value

    @property
    def datatype(self):
        """The datatype of this literal node.

        Returns:
            str: The datatype of this literal node.

        Raises:
            ValueError: If this is not a literal node.
        """
        if not self.is_literal:
            raise ValueError('Value is not a literal')
        return self.sparql_value.datatype

    @property
    def lang(self):
        """The language of this literal node.

        Returns:
            str: The language of this literal node.

        Raises:
            ValueError: If this is not a literal node.
        """
        if not self.is_literal:
            raise ValueError('Value is not a literal')
        return self.sparql_value.lang

    @property
    def rdf_format(self):
        """Convert this node into RDF format.

        Returns:
            str: This node as an RDF/SPARQL string.

        Raises:
            ValueError: If this is a blank node.
        """
        if self.is_uri:
            return f'<{self.sparql_value.value}>'
        elif self.is_literal:
            # FIXME there may be issues with escaping quote here
            result = f'"{self.sparql_value.value}"'
            if self.lang:
                result += '@{self.lang}'
            if self.datatype:
                result += f'^^<{self.datatype}>'
            return result
        raise ValueError(repr(self.sparql_value))


class SparqlEndpoint(KnowledgeSource):
    """A knowledge base from a remote SPARQL endpoint."""

    def __init__(self, url):
        """Initialize the SparqlEndpoint.

        Arguments:
            url (str): The URL to the SPARQL endpoint.
        """
        self.endpoint = SPARQLWrapper2(url)

    def query_sparql(self, sparql): # noqa: D102
        self.endpoint.setQuery(sparql)
        try:
            query_bindings = self.endpoint.query().bindings
        except QueryBadFormed as qbf:
            message = '\n'.join([
                'Failed to parse SPARQL',
                sparql,
                'Original error:',
                str(qbf),
            ])
            raise ValueError(message)
        results = []
        for bindings in query_bindings:
            results.append({key: Value(value) for key, value in bindings.items()})
        return results
