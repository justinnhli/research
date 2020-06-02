"""Long-term memory interface and implementations."""

from collections import namedtuple, defaultdict
from collections.abc import Hashable
from uuid import uuid4 as uuid

from networkx import MultiDiGraph

from .data_structures import AVLTree

class LongTermMemory:
    """Generic interface to a knowledge base."""

    def clear(self):
        """Remove all knowledge from the KB."""
        raise NotImplementedError()

    def store(self, mem_id=None, **kwargs):
        """Add knowledge to the KB.

        Arguments:
            mem_id (any): The ID of the element. Defaults to None.
            **kwargs: Attributes and values of the element to add.

        Returns:
            bool: True if the add was successful.
        """
        raise NotImplementedError()

    def retrieve(self, mem_id):
        """Retrieve the element with the given ID.

        Arguments:
            mem_id (any): The ID of the desired element.

        Returns:
            AVLTree: The desired element, or None.
        """
        raise NotImplementedError()

    def query(self, attr_vals):
        """Search the KB for elements with the given attributes.

        Arguments:
            attr_vals (Mapping[str, Any]): Attributes and values of the desired element.

        Returns:
            AVLTree: A search result, or None.
        """
        raise NotImplementedError()

    @property
    def has_prev_result(self):
        """Determine if a previous query result is available.

        Returns:
            bool: True if there is a previous result.
        """
        raise NotImplementedError()

    def prev_result(self):
        """Get the prev element that matches the most recent search.

        Returns:
            AVLTree: A search result, or None.
        """
        raise NotImplementedError()

    @property
    def has_next_result(self):
        """Determine if a next query result is available.

        Returns:
            bool: True if there is a next result.
        """
        raise NotImplementedError()

    def next_result(self):
        """Get the next element that matches the most recent search.

        Returns:
            AVLTree: A search result, or None.
        """
        raise NotImplementedError()

    @staticmethod
    def retrievable(mem_id):
        """Determine if an object is a retrievable memory ID.

        Arguments:
            mem_id (any): The object to check.

        Returns:
            bool: True if the object is a retrievable memory ID.
        """
        raise NotImplementedError()


class NaiveDictKB(LongTermMemory):
    """A list-of-dictionary implementation of a knowledge store."""

    def __init__(self):
        """Initialize the NaiveDictKB."""
        self.knowledge = []
        self.query_index = None
        self.query_matches = []

    def clear(self): # noqa: D102
        self.knowledge = []
        self.query_index = None
        self.query_matches = []

    def store(self, mem_id=None, **kwargs): # noqa: D102
        self.knowledge.append(AVLTree.from_dict(kwargs))
        return True

    def retrieve(self, mem_id): # noqa: D102
        raise NotImplementedError()

    def query(self, attr_vals): # noqa: D102
        candidates = []
        for candidate in self.knowledge:
            match = all(
                attr in candidate and candidate[attr] == val
                for attr, val in attr_vals.items()
            )
            if match:
                candidates.append(candidate)
        if candidates:
            # if the current retrieved item still matches the new query
            # leave it there but update the cached matches and index
            if self.query_index is not None:
                curr_retrieved = self.query_matches[self.query_index]
            else:
                curr_retrieved = None
            self.query_matches = sorted(candidates, key=(lambda candidate: tuple(candidate.items())))
            # use the ValueError from list.index() to determine if the query still matches
            try:
                self.query_index = self.query_matches.index(curr_retrieved)
            except ValueError:
                self.query_index = 0
            return self.query_matches[self.query_index]
        self.query_index = None
        self.query_matches = []
        return None

    @property
    def has_prev_result(self): # noqa: D102
        return True

    def prev_result(self): # noqa: D102
        self.query_index = (self.query_index - 1) % len(self.query_matches)
        return self.query_matches[self.query_index]

    @property
    def has_next_result(self): # noqa: D102
        return True

    def next_result(self): # noqa: D102
        self.query_index = (self.query_index + 1) % len(self.query_matches)
        return self.query_matches[self.query_index]

    @staticmethod
    def retrievable(mem_id): # noqa: D102
        return False


class NetworkXKB(LongTermMemory):
    """A NetworkX implementation of a knowledge store."""

    def __init__(self, activation_fn=None):
        """Initialize the NetworkXKB."""
        # parameters
        if activation_fn is None:
            activation_fn = (lambda graph, mem_id: None)
        self.activation_fn = activation_fn
        # variables
        self.graph = MultiDiGraph()
        self.inverted_index = defaultdict(set)
        self.query_results = None
        self.result_index = None
        self.clear()

    def clear(self): # noqa: D102
        self.graph.clear()
        self.inverted_index.clear()
        self.query_results = None
        self.result_index = None

    def store(self, mem_id=None, **kwargs): # noqa: D102
        if mem_id is None:
            mem_id = uuid()
        if mem_id not in self.graph:
            self.graph.add_node(mem_id, activation=0)
        else:
            self.activation_fn(self.graph, mem_id)
        for attribute, value in kwargs.items():
            if value not in self.graph:
                self.graph.add_node(value, activation=0)
            self.graph.add_edge(mem_id, value, attribute=attribute)
            self.inverted_index[attribute].add(mem_id)
        return True

    def _activate_and_return(self, mem_id):
        self.activation_fn(self.graph, mem_id)
        result = AVLTree()
        for _, value, data in self.graph.out_edges(mem_id, data=True):
            result[data['attribute']] = value
        return result

    def retrieve(self, mem_id): # noqa: D102
        if mem_id not in self.graph:
            return None
        return self._activate_and_return(mem_id)

    def query(self, attr_vals): # noqa: D102
        # first pass: get candidates with all the attributes
        candidates = set.intersection(*(
            self.inverted_index[attribute] for attribute in attr_vals.keys()
        ))
        # second pass: get candidates with the correct values
        candidates = set(
            candidate for candidate in candidates
            if all((
                (candidate, value) in self.graph.edges
                and self.graph.get_edge_data(candidate, value)[0]['attribute'] == attribute
            ) for attribute, value in attr_vals.items())
        )
        # quit early if there are no results
        if not candidates:
            self.query_results = None
            self.result_index = None
            return None
        # final pass: sort results by activation
        self.query_results = sorted(
            candidates,
            key=(lambda mem_id: self.graph.nodes[mem_id]['activation']),
            reverse=True,
        )
        self.result_index = 0
        return self._activate_and_return(self.query_results[self.result_index])

    @property
    def has_prev_result(self): # noqa: D102
        return (
            self.query_results is not None
            and self.result_index > 0
        )

    def prev_result(self): # noqa: D102
        self.result_index -= 1
        return self._activate_and_return(self.query_results[self.result_index])

    @property
    def has_next_result(self): # noqa: D102
        return (
            self.query_results is not None
            and self.result_index < len(self.query_results) - 1
        )

    def next_result(self): # noqa: D102
        self.result_index += 1
        return self._activate_and_return(self.query_results[self.result_index])

    @staticmethod
    def retrievable(mem_id): # noqa: D102
        return isinstance(mem_id, Hashable)


class SparqlKB(LongTermMemory):
    """An adaptor for RL agents to use KnowledgeSources."""

    # FIXME arguably this should be abstracted and moved to LongTermMemory
    Augment = namedtuple('Augment', 'old_attrs, transform')

    BAD_VALUES = set([
        '"NAN"^^<http://www.w3.org/2001/XMLSchema#double>',
        '"NAN"^^<http://www.w3.org/2001/XMLSchema#float>',
    ])

    def __init__(self, knowledge_source, augments=None):
        """Initialize a SparqlKB.

        Arguments:
            knowledge_source (KnowledgeSource): A SPARQL knowledge source.
            augments (Sequence[Augment]): Additional values to add to results.
        """
        # parameters
        self.source = knowledge_source
        if augments is None:
            augments = []
        self.augments = list(augments)
        # variables
        self.prev_query = None
        self.query_offset = 0
        # cache
        self.retrieve_cache = {}
        self.query_cache = {}

    def clear(self): # noqa: D102
        raise NotImplementedError()

    def store(self, mem_id=None, **kwargs): # noqa: D102
        raise NotImplementedError()

    def retrieve(self, mem_id): # noqa: D102
        valid_mem_id = (
            isinstance(mem_id, str)
            and mem_id.startswith('<http')
            and mem_id.endswith('>')
        )
        if not valid_mem_id:
            raise ValueError(
                f'mem_id should be a str of the form "<http:.*>", '
                f'but got: {mem_id}'
            )
        if mem_id not in self.retrieve_cache:
            result = self._true_retrieve(mem_id)
            for augment in self.augments:
                if all(attr in result for attr in augment.old_attrs):
                    new_prop_val = augment.transform(result)
                    if new_prop_val is not None:
                        new_prop, new_val = new_prop_val
                        result[new_prop] = new_val
            self.retrieve_cache[mem_id] = AVLTree.from_dict(result)
        result = self.retrieve_cache[mem_id]
        self.prev_query = None
        self.query_offset = 0
        return result

    def _true_retrieve(self, mem_id):
        query = f'''
        SELECT DISTINCT ?attr ?value WHERE {{
            {mem_id} ?attr ?value .
        }}
        '''
        results = self.source.query_sparql(query)
        # FIXME HACK to avoid dealing with multi-valued attributes,
        # we only return the "largest" value for each attribute
        result = defaultdict(set)
        for binding in results:
            val = binding['value'].rdf_format
            if val in self.BAD_VALUES:
                continue
            result[binding['attr'].rdf_format].add(val)
        return {attr: max(vals) for attr, vals in result.items()}

    def query(self, attr_vals): # noqa: D102
        query_terms = tuple((k, v) for k, v in sorted(attr_vals.items()))
        if query_terms not in self.query_cache:
            mem_id = self._true_query(attr_vals)
            self.query_cache[query_terms] = mem_id
        mem_id = self.query_cache[query_terms]
        self.query_offset = 0
        if mem_id is None:
            self.prev_query = None
            return AVLTree()
        else:
            self.prev_query = attr_vals
            return self.retrieve(mem_id)

    def _true_query(self, attr_vals, offset=0):
        condition = ' ; '.join(
            f'{attr} {val}' for attr, val in attr_vals.items()
        )
        query = f'''
        SELECT DISTINCT ?concept WHERE {{
            ?concept {condition} ;
                     <http://xmlns.com/foaf/0.1/name> ?__name__ .
        }} ORDER BY ?__name__ LIMIT 1 OFFSET {offset}
        '''
        results = self.source.query_sparql(query)
        try:
            return next(iter(results))['concept'].rdf_format
        except StopIteration:
            return None

    @property
    def has_prev_result(self): # noqa: D102
        return self.prev_query is not None and self.query_offset > 0

    def prev_result(self): # noqa: D102
        if not self.has_prev_result:
            return None
        self.query_offset -= 1
        return self._true_query(self.prev_query, offset=self.query_offset)

    @property
    def has_next_result(self): # noqa: D102
        return self.prev_query is not None

    def next_result(self): # noqa: D102
        if not self.has_next_result:
            return None
        self.query_offset += 1
        return self._true_query(self.prev_query, offset=self.query_offset)

    @staticmethod
    def retrievable(mem_id): # noqa: D102
        return isinstance(mem_id, str) and mem_id.startswith('<http')
