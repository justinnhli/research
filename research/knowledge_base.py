"""A module to handle local and remote knowledge bases."""

import re
from ast import literal_eval
from enum import Enum, unique
from os.path import exists as file_exists, splitext as split_ext, expanduser, realpath
from time import sleep
from  urllib.error import URLError

from SPARQLWrapper import SPARQLWrapper2
from SPARQLWrapper.SmartWrapper import Value as SparqlValue
from SPARQLWrapper.SPARQLExceptions import EndPointNotFound
from rdflib import Graph, Literal, URIRef, plugin
from rdflib.store import Store
from rdflib.util import guess_format
from rdflib_sqlalchemy import registerplugins

registerplugins()


class Value:
    """Wrapper around SPARQLWrapper.SmartWrapper.Value."""

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
    PREFIXES = {value: key for key, value in NAMESPACES.items()}

    @unique
    class ValueType(Enum):
        """An enum for the possible value types."""

        URI = 'uri'
        LITERAL = 'literal'

    def __init__(self, value, value_type, lang=None, datatype=None):
        """Initialize a Value.

        Arguments:
            value (str): The value.
            value_type (ValueType): The type of value.
            lang (str): The language of a literal. Optional.
            datatype (str): The datatype of a literal. Optional.
        """
        self._value = value
        self.value_type = value_type
        self._lang = lang
        if datatype is None:
            self._datatype = None
        else:
            self._datatype = Value.from_uri(datatype)

    def __str__(self):
        return self.rdf_format

    @property
    def is_uri(self):
        """Whether this is a URI node.

        Returns:
            bool: True if this is a URI node.
        """
        return self.value_type == self.ValueType.URI

    @property
    def is_literal(self):
        """Whether this is a literal node.

        Returns:
            bool: True if this is a literal node.
        """
        return self.value_type == self.ValueType.LITERAL

    @property
    def uri(self):
        """Get the URI of this URI node.

        Returns:
            str: The URI of this URI node.

        Raises:
            ValueError: If this is not a URI node.
        """
        if not self.is_uri:
            raise ValueError('Value is not a URI')
        return self._value

    @property
    def namespace_fragment(self):
        """Get the URI of this URI node in namespace:fragment format.

        Returns:
            str: The URI of this URI node.

        Raises:
            ValueError: If this is not a URI node.
        """
        if not self.is_uri:
            raise ValueError('Value is not a URI')
        namespace = self.namespace
        if not namespace:
            return self.uri
        return namespace + ':' + self.fragment

    @property
    def prefix(self):
        """Get the prefix of this URI node.

        Returns:
            str: The prefix of this URI node.

        Raises:
            ValueError: If this is not a URI node.
        """
        if not self.is_uri:
            raise ValueError('Value is not a URI')
        candidates = [
            prefix for prefix in self.PREFIXES
            if self._value.startswith(prefix)
        ]
        if not candidates:
            return None
        return max(candidates, key=len)

    @property
    def namespace(self):
        """Get the namespace of this URI node.

        Returns:
            str: The namespace of this URI node.
        """
        prefix = self.prefix
        if not prefix:
            return None
        return self.PREFIXES[prefix]

    @property
    def fragment(self):
        """Get the namespace of this URI node.

        Returns:
            str: The namespace of this URI node.
        """
        prefix = self.prefix
        if not prefix:
            return None
        return self._value[len(prefix):]

    @property
    def literal_value(self):
        """Get the value of this literal node.

        Returns:
            Union[bool,int,float,str]: The value of this literal node.

        Raises:
            ValueError: If this is not a literal node.
        """
        if not self.is_literal:
            raise ValueError('Value is not a literal')
        return self._value

    @property
    def datatype(self):
        """Get the datatype of this literal node.

        Returns:
            str: The datatype of this literal node.

        Raises:
            ValueError: If this is not a literal node.
        """
        if not self.is_literal:
            raise ValueError('Value is not a literal')
        return self._datatype

    @property
    def lang(self):
        """Get the language of this literal node.

        Returns:
            str: The language of this literal node.

        Raises:
            ValueError: If this is not a literal node.
        """
        if not self.is_literal:
            raise ValueError('Value is not a literal')
        return self._lang

    @property
    def rdf_format(self):
        """Convert this node into RDF format.

        Returns:
            str: This node as an RDF/SPARQL string.

        Raises:
            ValueError: If this is a blank node.
        """
        if self.is_uri:
            return f'<{self._value}>'
        elif isinstance(self._value, bool):
            return str(self._value).lower()
        elif isinstance(self._value, (int, float)):
            return str(self._value)
        else:
            result = self._value.replace('"', r'\"')
            if '\n' in self._value:
                result = f'"""{result}"""'
            else:
                result = f'"{result}"'
            if self._lang:
                result += f'@{self._lang}'
            if self._datatype:
                result += f'^^{self._datatype}'
            return result
        raise ValueError(repr(self._value))

    @staticmethod
    def from_sparqlwrapper(sparql_value):
        """Create a Value from a SPARQLWrapper value.

        Arguments:
            sparql_value (SPARQLWrapper.SmartWrapper.Value):
                The SPARQLWrapper value.

        Returns:
            Value: The resulting value.

        Raises:
            ValueError: If the SPARQLWrapper value is not a URI or a literal.
        """
        if sparql_value.type == SparqlValue.URI:
            return Value(sparql_value.value, Value.ValueType.URI)
        elif sparql_value.type in (SparqlValue.Literal, SparqlValue.TypedLiteral):
            return Value(
                sparql_value.value,
                Value.ValueType.LITERAL,
                sparql_value.lang,
                sparql_value.datatype,
            )
        else:
            raise ValueError(f'sparql_value is neither a URI nor a literal: {sparql_value}')

    @staticmethod
    def from_python_literal(literal):
        """Create a Value from a literal.

        Arguments:
            literal (Union[bool,int,float,str]):
                An RDF representation of a literal.

        Returns:
            Value: The resulting value.

        Raises:
            ValueError: If the literal is not of an appropriate type.
        """
        if isinstance(literal, str):
            return Value(literal, Value.ValueType.LITERAL, None, None)
        elif isinstance(literal, bool):
            datatype = 'http://www.w3.org/2001/XMLSchema#boolean'
        elif isinstance(literal, int):
            datatype = 'http://www.w3.org/2001/XMLSchema#integer'
        elif isinstance(literal, float):
            datatype = 'http://www.w3.org/2001/XMLSchema#double'
        return Value(literal, Value.ValueType.LITERAL, None, datatype)

    @staticmethod
    def from_literal(literal):
        """Create a Value from a literal.

        Arguments:
            literal (str): An RDF representation of a literal.

        Returns:
            Value: The resulting value.

        Raises:
            ValueError: If the literal fails to parse.
        """
        try:
            if literal in ('true', 'false'):
                evaled = literal_eval(literal.title())
            else:
                evaled = literal_eval(literal)
            return Value.from_python_literal(evaled)
        except SyntaxError:
            pass
        match = re.fullmatch(
            (
                '(?P<value>(?P<quote>["\']).*(?P=quote))'
                '(?P<lang>(@[a-z]+)?)'
                r'(?P<datatype>(\^\^<.*>)?)'
            ),
            literal,
        )
        if not match:
            raise ValueError('failed to parse literal: {literal}')
        if match.group('lang'):
            lang = match.group('lang')[1:]
        else:
            lang = None
        if match.group('datatype'):
            datatype = match.group('datatype')[2:]
        else:
            datatype = None
        return Value(
            literal_eval(match.group('value')),
            Value.ValueType.LITERAL,
            lang,
            datatype,
        )

    @staticmethod
    def from_uri(uri):
        """Create a Value from a URI.

        Arguments:
            uri (str): The value.

        Returns:
            Value: The resulting value.
        """
        if uri[0] == '<' and uri[-1] == '>':
            uri = uri[1:-1]
        return Value(uri, Value.ValueType.URI)

    @staticmethod
    def from_namespace_fragment(namespace, fragment):
        """Create a Value from a URI in namespace:fragment format.

        Arguments:
            namespace (str): The URI namespace.
            fragment (str): The URI fragment.

        Returns:
            Value: The resulting value.
        """
        return Value.from_uri(Value.NAMESPACES[namespace] + fragment)


class KnowledgeSource:
    """Abstract class to represent a knowledge source."""

    def query_sparql(self, sparql):
        """Query the KB with SPARQL.

        Arguments:
            sparql (str): The SPARQL query.

        Yields:
            Dict[str, Value]: A dictionary of variable bindings.
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
            # FIXME need to add prefix definitions for other formats
            preamble = ''
            if rdf_format == 'n3':
                preamble = '\n'.join(
                    f'@prefix {prefix}: <{url}>.'
                    for prefix, url in Value.NAMESPACES.items()
                    if url != '_'
                )
            with open(source) as fd:
                data = fd.read()
            if preamble:
                data = preamble + '\n' + data
            if sqlize:
                sql_uri = 'sqlite:///' + filepath + '.rdfsqlite'
            else:
                sql_uri = 'sqlite://'
            self.graph.open(Literal(sql_uri), create=True)
            self.graph.parse(data=data, format=rdf_format)
        elif ext[1:] in ['db', 'sqlite', 'rdfsqlite']:
            sql_uri = 'sqlite:///' + source
            self.graph.open(Literal(sql_uri))
        else:
            raise ValueError('Cannot determine format of {}'.format(source))

    def __del__(self):
        self.graph.commit()
        self.graph.close()

    def query_sparql(self, sparql): # noqa: D102
        for result in self.graph.query(sparql).bindings:
            yield {str(variable): str(uri) for variable, uri in result.items()}


class SparqlEndpoint(KnowledgeSource):
    """A knowledge base from a remote SPARQL endpoint."""

    NUM_CONNECTION_ATTEMPTS = 10

    def __init__(self, url):
        """Initialize the SparqlEndpoint.

        Arguments:
            url (str): The URL to the SPARQL endpoint.
        """
        self.endpoint = SPARQLWrapper2(url)

    def query_sparql(self, sparql): # noqa: D102
        self.endpoint.setQuery(sparql)
        query_bindings = None
        for _ in range(self.NUM_CONNECTION_ATTEMPTS):
            try:
                query_bindings = self.endpoint.query().bindings
                break
            except (EndPointNotFound, URLError):
                sleep(3)
        if query_bindings is None:
            raise EndPointNotFound(
                f'Tried to connect {self.NUM_CONNECTION_ATTEMPTS} times and failed'
            )
        for bindings in query_bindings:
            if any(value.type == SparqlValue.BNODE for value in bindings.values()):
                continue
            yield {
                key: Value.from_sparqlwrapper(value)
                for key, value in bindings.items()
            }
