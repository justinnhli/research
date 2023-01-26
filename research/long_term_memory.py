"""Long-term memory interface and implementations."""

from collections import namedtuple, defaultdict
from collections.abc import Hashable
from typing import Any, Optional, Sequence, AbstractSet, Tuple, List, Set, Dict
from uuid import uuid4 as uuid

from networkx import MultiDiGraph

from .data_structures import AVLTree
from .rl_environments import AttrVal
from .knowledge_base import KnowledgeSource


class ActivationDynamics:
    """Generic interface for activation functions."""

    def __init__(self, ltm, **kwargs):
        # type: (LongTermMemory) -> None
        """Initialize the ActivationDynamics.

        Parameters:
            ltm (LongTermMemory): The LongTermMemory that will be using this activation.
        """
        super().__init__(**kwargs)
        self.ltm = ltm

    def activate(self, mem_id, time):
        # type: (Hashable, int) -> None
        # pylint: disable = unused-argument
        """Activation the element with the given ID.

        Parameters:
            mem_id (any): The ID of the element to activate.
            time (int): The time of activation. Optional.
        """
        return

    def get_activation(self, mem_id, time):
        # type: (Hashable, int) -> float
        # pylint: disable = unused-argument
        """Get the activation of the element with the given ID.

        Parameters:
            mem_id (any): The ID of the desired element.
            time (int): The time of activation. Optional.

        Returns:
            float: The activation of the element.
        """
        return 0


class FrequencyActivation(ActivationDynamics):
    """Activation functions that only care about the frequency of activations."""

    def __init__(self, ltm, **kwargs):
        super().__init__(ltm, **kwargs)
        self.activations = defaultdict(int)

    def activate(self, mem_id, time):
        self.activations[mem_id] += 1

    def get_activation(self, mem_id, time):
        return self.activations[mem_id]


class RecencyActivation(ActivationDynamics):
    """Activation functions that only care about the recency of activations."""

    def __init__(self, ltm, **kwargs):
        super().__init__(ltm, **kwargs)
        self.activations = defaultdict(int)

    def activate(self, mem_id, time):
        self.activations[mem_id] = time

    def get_activation(self, mem_id, time):
        return self.activations[mem_id]


class LongTermMemory:
    """Generic interface to a knowledge base."""

    def __init__(self, activation_cls=None, **kwargs):
        # type: (type[ActivationDynamics], **kwargs) -> None
        """Initialize the LongTermMemory.

        Parameters:
            activation_cls (ActivationDynamics):
                The activation function to call when storing/retrieving. Optional.
            **kwargs (Any): Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        if activation_cls is None:
            self.activation_dynamics = ActivationDynamics(self)
        else:
            self.activation_dynamics = activation_cls(self)

    def clear(self):
        # type: () -> None
        """Remove all knowledge from the LTM."""
        raise NotImplementedError()

    def store(self, mem_id=None, time=0, **kwargs):
        # type: (Hashable, int, **Any) -> bool
        """Add knowledge to the LTM.

        Parameters:
            mem_id (Hashable): The ID of the element. Defaults to None.
            time (int): The time of storage (for activation). Optional.
            **kwargs: Attributes and values of the element to add.

        Returns:
            bool: True if the add was successful.
        """
        raise NotImplementedError()

    def activate(self, mem_id, time=0):
        # type: (Hashable, int) -> Optional[AVLTree]
        """Activation the element with the given ID.

        Parameters:
            mem_id (any): The ID of the element to activate.
            time (int): The time of activation. Optional.
        """
        self.activation_dynamics.activate(mem_id, time)

    def get_activation(self, mem_id, time=0):
        # type: (Hashable, int) -> float
        """Get the activation of the element with the given ID.

        Parameters:
            mem_id (any): The ID of the desired element.
            time (int): The time of activation. Optional.

        Returns:
            float: The activation of the element.
        """
        return self.activation_dynamics.get_activation(mem_id, time)

    def retrieve(self, mem_id, time=0):
        # type: (Hashable, int) -> Optional[AVLTree]
        """Retrieve the element with the given ID.

        Parameters:
            mem_id (any): The ID of the desired element.
            time (int): The time of retrieval (for activation). Optional.

        Returns:
            AVLTree: The desired element, or None.
        """
        raise NotImplementedError()

    def query(self, attr_vals, time=0):
        # type: (AbstractSet[AttrVal], int) -> Optional[AVLTree]
        """Query the LTM for elements with the given attributes.

        Parameters:
            attr_vals (AbstractSet[AttrVal]): Attributes and values of the desired element.
            time (int): The time of query (for activation). Optional.

        Returns:
            AVLTree: A search result, or None.
        """
        raise NotImplementedError()

    @property
    def has_prev_result(self):
        # type: () -> bool
        """Determine if a previous query result is available.

        Returns:
            bool: True if there is a previous result.
        """
        raise NotImplementedError()

    def prev_result(self, time=0):
        # type: (int) -> Optional[AVLTree]
        """Get the prev element that matches the most recent search.

        Parameters:
            time (int): The time of getting the previous result (for activation). Optional.

        Returns:
            AVLTree: A search result, or None.
        """
        raise NotImplementedError()

    @property
    def has_next_result(self):
        # type: () -> bool
        """Determine if a next query result is available.

        Returns:
            bool: True if there is a next result.
        """
        raise NotImplementedError()

    def next_result(self, time=0):
        # type: (int) -> Optional[AVLTree]
        """Get the next element that matches the most recent search.

        Parameters:
            time (int): The time of getting the next result (for activation). Optional.

        Returns:
            AVLTree: A search result, or None.
        """
        raise NotImplementedError()

    @staticmethod
    def retrievable(mem_id):
        # type: (Any) -> bool
        """Determine if an object is a retrievable memory ID.

        Parameters:
            mem_id (any): The object to check.

        Returns:
            bool: True if the object is a retrievable memory ID.
        """
        raise NotImplementedError()


class NaiveDictLTM(LongTermMemory):
    """A list-of-dictionary implementation of LTM."""

    def __init__(self, **kwargs):
        # type: (**Any) -> None
        """Initialize the NaiveDictLTM.

        Parameters:
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.knowledge = {} # type: Dict[Hashable, AbstractSet[AttrVal]]
        self.query_index = -1
        self.query_matches = [] # type: List[Hashable]

    def clear(self): # noqa: D102
        # type: () -> None
        self.knowledge = {}
        self.query_index = -1
        self.query_matches = []

    def store(self, mem_id=None, time=0, **kwargs): # noqa: D102
        # type: (Hashable, int, **Any) -> bool
        if mem_id is None:
            mem_id = uuid()
        if mem_id not in self.knowledge:
            self.knowledge[mem_id] = AVLTree()
        else:
            self.activate(mem_id, time)
        for attr, val in kwargs.items():
            if val not in self.knowledge:
                self.knowledge[val] = AVLTree()
            self.knowledge[mem_id].add(AttrVal(attr, val))
        return True

    def _activate_and_return(self, mem_id, time):
        self.activate(mem_id, time)
        return self.knowledge[mem_id]

    def retrieve(self, mem_id, time=0): # noqa: D102
        # type: (Hashable, int) -> Optional[AVLTree]
        if mem_id not in self.knowledge:
            return None
        return self._activate_and_return(mem_id, time=time)

    def query(self, attr_vals, time=0): # noqa: D102
        # type: (AbstractSet[AttrVal], int) -> Optional[AVLTree]
        candidates = []
        for candidate, cand_attr_vals in self.knowledge.items():
            if cand_attr_vals.is_superset(attr_vals):
                candidates.append(candidate)
        if candidates:
            # if the current retrieved item still matches the new query
            # leave it there but update the cached matches and index
            if self.query_index != -1:
                curr_retrieved = self.query_matches[self.query_index]
            else:
                curr_retrieved = None
            self.query_matches = sorted(
                candidates,
                key=self.get_activation,
                reverse=True,
            )
            # use the ValueError from list.index() to determine if the query still matches
            try:
                self.query_index = self.query_matches.index(curr_retrieved)
            except ValueError:
                self.query_index = 0
            return self._activate_and_return(self.query_matches[self.query_index], time=time)
        self.query_index = -1
        self.query_matches = []
        return None

    @property
    def has_prev_result(self): # noqa: D102
        # type: () -> bool
        return bool(self.query_matches) and self.query_index > 0

    def prev_result(self, time=0): # noqa: D102
        # type: (int) -> Optional[AVLTree]
        self.query_index = (self.query_index - 1) % len(self.query_matches)
        return self._activate_and_return(self.query_matches[self.query_index], time=time)

    @property
    def has_next_result(self): # noqa: D102
        # type: () -> bool
        return bool(self.query_matches) and -1 < self.query_index < len(self.query_matches) - 1

    def next_result(self, time=0): # noqa: D102
        # type: (int) -> Optional[AVLTree]
        self.query_index = (self.query_index + 1) % len(self.query_matches)
        return self._activate_and_return(self.query_matches[self.query_index], time=time)

    @staticmethod
    def retrievable(mem_id): # noqa: D102
        # type: (Any) -> bool
        return mem_id is not None and isinstance(mem_id, Hashable)


class NetworkXLTM(LongTermMemory):
    """A NetworkX implementation of LTM."""

    def __init__(self, **kwargs):
        # type: (**Any) -> None
        """Initialize the NetworkXLTM.

        Parameters:
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.graph = MultiDiGraph()
        self.inverted_index = defaultdict(set) # type: Dict[Hashable, Set[Hashable]]
        self.query_results = [] # type: List[Hashable]
        self.result_index = -1
        self.clear()

    def clear(self): # noqa: D102
        # type: () -> None
        self.graph.clear()
        self.inverted_index.clear()
        self.query_results = []
        self.result_index = -1

    def store(self, mem_id=None, time=0, **kwargs): # noqa: D102
        # type: (Hashable, int, **Any) -> bool
        if mem_id is None:
            mem_id = uuid()
        if mem_id not in self.graph:
            self.graph.add_node(mem_id, activation=0)
        else:
            self.activate(mem_id, time)
        for attribute, value in kwargs.items():
            if value not in self.graph:
                self.graph.add_node(value, activation=0)
            self.graph.add_edge(mem_id, value, attribute=attribute)
            self.inverted_index[attribute].add(mem_id)
        return True

    def _activate_and_return(self, mem_id, time):
        # type: (Hashable, int) -> AVLTree
        self.activate(mem_id, time)
        result = AVLTree()
        for _, value, data in self.graph.out_edges(mem_id, data=True):
            result.add(AttrVal(data['attribute'], value))
        return result

    def retrieve(self, mem_id, time=0): # noqa: D102
        # type: (Hashable, int) -> Optional[AVLTree]
        if mem_id not in self.graph:
            return None
        return self._activate_and_return(mem_id, time=time)

    def query(self, attr_vals, time=0): # noqa: D102
        # type: (AbstractSet[AttrVal], int) -> Optional[AVLTree]
        # first pass: get candidates with all the attributes
        attrs = set(attr for attr, _ in attr_vals)
        candidates = set.intersection(*(
            self.inverted_index[attribute] for attribute in attrs
        ))
        # second pass: get candidates with the correct values
        candidates = set(
            candidate for candidate in candidates
            if all((
                (candidate, val) in self.graph.edges
                and any(
                    attr_dict['attribute'] == attr
                    for attr_dict in self.graph.get_edge_data(candidate, val).values()
                )
            ) for attr, val in attr_vals)
        )
        # quit early if there are no results
        if not candidates:
            self.query_results = []
            self.result_index = -1
            return None
        # final pass: sort results by activation
        self.query_results = sorted(
            candidates,
            key=self.get_activation,
            reverse=True,
        )
        self.result_index = 0
        return self._activate_and_return(self.query_results[self.result_index], time=time)

    @property
    def has_prev_result(self): # noqa: D102
        # type: () -> bool
        return bool(self.query_results) and self.result_index > 0

    def prev_result(self, time=0): # noqa: D102
        # type: (int) -> Optional[AVLTree]
        self.result_index -= 1
        return self._activate_and_return(self.query_results[self.result_index], time=time)

    @property
    def has_next_result(self): # noqa: D102
        # type: () -> bool
        return bool(self.query_results) and -1 < self.result_index < len(self.query_results) - 1

    def next_result(self, time=0): # noqa: D102
        # type: (int) -> Optional[AVLTree]
        self.result_index += 1
        return self._activate_and_return(self.query_results[self.result_index], time=time)

    @staticmethod
    def retrievable(mem_id): # noqa: D102
        # type: (Any) -> bool
        return mem_id is not None and isinstance(mem_id, Hashable)


class SparqlLTM(LongTermMemory):
    """An adaptor for RL agents to use KnowledgeSources."""

    # FIXME arguably this should be abstracted and moved to LongTermMemory
    Augment = namedtuple('Augment', 'old_attrs, transform')

    BAD_VALUES = set([
        '"NAN"^^<http://www.w3.org/2001/XMLSchema#double>',
        '"NAN"^^<http://www.w3.org/2001/XMLSchema#float>',
    ])

    def __init__(self, knowledge_source, augments=None, **kwargs):
        # type: (KnowledgeSource, Sequence[Augment], **Any) -> None
        """Initialize a SparqlLTM.

        Parameters:
            knowledge_source (KnowledgeSource): A SPARQL knowledge source.
            augments (Sequence[Augment]): Additional values to add to results.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        # parameters
        self.source = knowledge_source
        if augments is None:
            augments = []
        self.augments = list(augments)
        # variables
        self.prev_query = None # type: Optional[AbstractSet[AttrVal]]
        self.query_offset = 0
        # cache
        self.retrieve_cache = {} # type: Dict[Hashable, AVLTree]
        self.query_cache = {} # type: Dict[Tuple[AttrVal, ...], Optional[Hashable]]

    def clear(self): # noqa: D102
        # type: () -> None
        raise NotImplementedError()

    def store(self, mem_id=None, time=0, **kwargs): # noqa: D102
        # type: (Hashable, int, **Any) -> bool
        raise NotImplementedError()

    def retrieve(self, mem_id, time=0): # noqa: D102
        # type: (Hashable, int) -> Optional[AVLTree]
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
                attr_dict = dict(result)
                if all(attr in attr_dict for attr in augment.old_attrs):
                    new_prop_val = augment.transform(attr_dict)
                    if new_prop_val is not None:
                        new_prop, new_val = new_prop_val
                        result.add(AttrVal(new_prop, new_val))
            self.retrieve_cache[mem_id] = AVLTree.from_dict(result)
        result = self.retrieve_cache[mem_id]
        self.prev_query = None
        self.query_offset = 0
        return result

    def _true_retrieve(self, mem_id):
        # type: (Hashable) -> Optional[AVLTree]
        query = f'''
        SELECT DISTINCT ?attr ?value WHERE {{
            {mem_id} ?attr ?value .
        }}
        '''
        results = self.source.query_sparql(query)
        # FIXME HACK to avoid dealing with multi-valued attributes,
        # we only return the "largest" value for each attribute
        result = AVLTree()
        for binding in results:
            val = binding['value'].rdf_format
            if val not in self.BAD_VALUES:
                result.add(AttrVal(
                    binding['attr'].rdf_format,
                    val,
                ))
        return result

    def query(self, attr_vals, time=0): # noqa: D102
        # type: (AbstractSet[AttrVal], int) -> Optional[AVLTree]
        query_terms = tuple(sorted(attr_vals))
        if query_terms not in self.query_cache:
            mem_id = self._true_query(attr_vals)
            self.query_cache[query_terms] = mem_id
        mem_id = self.query_cache[query_terms]
        self.query_offset = 0
        if mem_id is None:
            self.prev_query = None
            return None
        else:
            self.prev_query = attr_vals
            return self.retrieve(mem_id, time=time)

    def _true_query(self, attr_vals, offset=0):
        # type: (AbstractSet[AttrVal], int) -> Optional[Hashable]
        condition = ' ; '.join(
            f'{attr} {val}' for attr, val in attr_vals
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
        # type: () -> bool
        return self.prev_query is not None and self.query_offset > 0

    def prev_result(self, time=0): # noqa: D102
        # type: (int) -> Optional[AVLTree]
        if not self.has_prev_result:
            return None
        self.query_offset -= 1
        mem_id = self._true_query(self.prev_query, offset=self.query_offset)
        if mem_id is None:
            return None
        else:
            return self.retrieve(mem_id, time=time)

    @property
    def has_next_result(self): # noqa: D102
        # type: () -> bool
        return self.prev_query is not None

    def next_result(self, time=0): # noqa: D102
        # type: (int) -> Optional[AVLTree]
        if not self.has_next_result:
            return None
        self.query_offset += 1
        mem_id = self._true_query(self.prev_query, offset=self.query_offset)
        if mem_id is None:
            return None
        else:
            return self.retrieve(mem_id, time=time)

    @staticmethod
    def retrievable(mem_id): # noqa: D102
        # type: (Any) -> bool
        return isinstance(mem_id, str) and mem_id.startswith('<http')
