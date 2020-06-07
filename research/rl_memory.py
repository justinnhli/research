"""Memory architecture for reinforcement learning."""

from collections import namedtuple, defaultdict
from copy import deepcopy

from .data_structures import AVLTree
from .rl_environments import State, Action, Environment
from .long_term_memory import AttrValPair


def memory_architecture(cls):
    """Decorate an Environment to become a memory architecture.

    Arguments:
        cls (class): The Environment superclass.

    Returns:
        class: A subclass with a memory architecture.
    """
    # pylint: disable = too-many-statements
    assert issubclass(cls, Environment)

    # pylint: disable = invalid-name
    BufferProperties = namedtuple('BufferProperties', ['copyable', 'writable'])

    class MemoryArchitectureMetaEnvironment(cls):
        """A subclass to add a long-term memory to an Environment."""

        BUFFERS = {
            'perceptual': BufferProperties(
                copyable=True,
                writable=False,
            ),
            'query': BufferProperties(
                copyable=False,
                writable=True,
            ),
            'retrieval': BufferProperties(
                copyable=True,
                writable=False,
            ),
            'scratch': BufferProperties(
                copyable=True,
                writable=True,
            ),
        }

        def __init__(
                self, ltm,
                buf_ignore=None, internal_reward=-0.1, max_internal_actions=None,
                **kwargs,
        ): # noqa: D102
            """Initialize a memory architecture.

            Arguments:
                ltm (LongTermMemory): The memory model to use.
                buf_ignore (Iterable[str]): Buffers that should not be created.
                internal_reward (float): Reward for internal actions. Defaults to -0.1.
                max_internal_actions (int): Max number of consecutive internal actions. Defaults to None.
                **kwargs: Arbitrary keyword arguments.
            """
            # pylint: disable = keyword-arg-before-vararg
            # parameters
            if buf_ignore is None:
                buf_ignore = set()
            self.buf_ignore = set(buf_ignore)
            self.internal_reward = internal_reward
            self.max_internal_actions = max_internal_actions
            # infrastructure
            self.ltm = ltm
            # variables
            self.buffers = {}
            self.internal_action_count = 0
            self._state = State()
            self._buffer_changes = defaultdict(int)
            # initialization
            self._clear_all_buffers()
            super().__init__(**kwargs)

        @property
        def slots(self):
            """Yield all values of all attributes in all buffers.

            Yields:
                Tuple[str, str, Any]: A tuple of buffer, attribute, and value.
            """
            for buf, attrs in self.buffers.items():
                for (attr, val) in attrs:
                    yield buf, attr, val

        def to_dict(self):
            """Convert the state into a dictionary."""
            return {buf + '_' + attr: val for buf, attr, val in self.slots}

        def get_state(self): # noqa: D102
            # pylint: disable = missing-docstring
            if self._buffer_changes:
                self._state = deepcopy(self._state)
                for (buf, attr, val), count in self._buffer_changes.items():
                    if count < 0:
                        del self._state[buf + '_' + attr]
                    elif count > 0:
                        self._state[buf + '_' + attr] = val
            self._buffer_changes = defaultdict(int)
            return self._state

        def get_observation(self): # noqa: D102
            # pylint: disable = missing-docstring
            return self.get_state()

        def reset(self): # noqa: D102
            # pylint: disable = missing-docstring
            super().reset()
            self._clear_all_buffers()

        def _track_adds(self, buf, attr, val):
            key = (buf, attr, val)
            if self._buffer_changes[key] == -1:
                del self._buffer_changes[key]
            else:
                self._buffer_changes[key] += 1

        def _track_dels(self, buf, attr, val):
            key = (buf, attr, val)
            if self._buffer_changes[key] == 1:
                del self._buffer_changes[key]
            else:
                self._buffer_changes[key] -= 1

        def _add_attr(self, buf, attr, val):
            self.buffers[buf][attr] = val
            self._track_adds(buf, attr, val)

        def _set_buffer(self, buf, contents):
            for attr, val in self.buffers[buf].items():
                self._track_dels(buf, attr, val)
            for attr, val in contents.items():
                self._track_adds(buf, attr, val)
            self.buffers[buf] = contents

        def _del_attr(self, buf, attr, val):
            self._track_dels(buf, attr, val)
            del self.buffers[buf][attr]

        def _clear_buffer(self, buf):
            for attr, val in self.buffers[buf].items():
                self._track_dels(buf, attr, val)
            self.buffers[buf].clear()

        def _clear_all_buffers(self):
            self.buffers = {}
            for buf, _ in self.BUFFERS.items():
                if buf in self.buf_ignore:
                    continue
                self.buffers[buf] = AVLTree()
            self._state = State()
            self._buffer_changes = defaultdict(int)

        def _clear_ltm_buffers(self):
            self._clear_buffer('query')
            self._clear_buffer('retrieval')

        def start_new_episode(self): # noqa: D102
            # pylint: disable = missing-docstring
            super().start_new_episode()
            self._clear_all_buffers()
            self._sync_input_buffers()
            self.internal_action_count = 0

        def get_actions(self): # noqa: D102
            # pylint: disable = missing-docstring
            actions = super().get_actions()
            if actions == []:
                return actions
            actions = set(actions)
            allow_internal_actions = (
                self.max_internal_actions is None
                or self.internal_action_count < self.max_internal_actions
            )
            if allow_internal_actions:
                actions.update(self._generate_copy_actions())
                actions.update(self._generate_delete_actions())
                actions.update(self._generate_retrieve_actions())
                actions.update(self._generate_cursor_actions())
            return sorted(actions)

        def _generate_copy_actions(self):
            actions = []
            for src_buf, src_props in self.BUFFERS.items():
                if src_buf in self.buf_ignore or not src_props.copyable:
                    continue
                for (attr, val) in self.buffers[src_buf]:
                    for dst_buf, dst_prop in self.BUFFERS.items():
                        copy_okay = (
                            src_buf != dst_buf
                            and dst_buf not in self.buf_ignore
                            and dst_prop.writable
                            and not (src_buf == 'perceptual' and dst_buf == 'scratch')
                            and AttrValPair(attr, val) not in self.buffers[dst_buf]
                        )
                        if not copy_okay:
                            continue
                        actions.append(Action(
                            'copy',
                            src_buf=src_buf,
                            src_attr=attr,
                            dst_buf=dst_buf,
                            dst_attr=attr,
                            dst_val=val,
                        ))
            return actions

        def _generate_delete_actions(self):
            actions = []
            for buf, prop in self.BUFFERS.items():
                if buf in self.buf_ignore or not prop.writable:
                    continue
                for (attr, val) in self.buffers[buf]:
                    actions.append(Action(
                        'delete',
                        buf=buf,
                        attr=attr,
                        val=val,
                    ))
            return actions

        def _generate_retrieve_actions(self):
            actions = []
            for buf, buf_props in self.BUFFERS.items():
                if buf in self.buf_ignore or not buf_props.copyable:
                    continue
                for attr, val in self.buffers[buf].items():
                    if self.ltm.retrievable(val):
                        actions.append(Action('retrieve', val=val))
            return actions

        def _generate_cursor_actions(self):
            actions = []
            if self.buffers['retrieval']:
                if self.ltm.has_prev_result:
                    actions.append(Action('prev-result'))
                if self.ltm.has_next_result:
                    actions.append(Action('next-result'))
            return actions

        def react(self, action): # noqa: D102
            # pylint: disable = missing-docstring
            # handle internal actions and update internal buffers
            external_action = self._process_internal_actions(action)
            if external_action:
                reward = super().react(action)
                self.internal_action_count = 0
            else:
                reward = self.internal_reward
                self.internal_action_count += 1
            self._sync_input_buffers()
            return reward

        def _process_internal_actions(self, action):
            """Process internal actions, if appropriate.

            Arguments:
                action (Action): The action, which may or may not be internal.

            Returns:
                bool: Whether the action was external.
            """
            if action.name == 'copy':
                self._add_attr(
                    action.dst_buf,
                    action.dst_attr,
                    action.dst_val,
                )
                if action.dst_buf == 'query':
                    self._query_ltm()
            elif action.name == 'delete':
                self._del_attr(action.buf, action.attr, action.val)
                if action.buf == 'query':
                    self._query_ltm()
            elif action.name == 'retrieve':
                result = self.ltm.retrieve(action.val)
                self._clear_buffer('query')
                if result is None:
                    self._clear_buffer('retrieval')
                else:
                    self._set_buffer('retrieval', result)
            elif action.name == 'prev-result':
                self._set_buffer('retrieval', self.ltm.prev_result())
            elif action.name == 'next-result':
                self._set_buffer('retrieval', self.ltm.next_result())
            else:
                return True
            return False

        def _query_ltm(self):
            if not self.buffers['query']:
                self._clear_buffer('retrieval')
                return
            result = self.ltm.query(self.buffers['query'])
            if result is None:
                self._clear_buffer('retrieval')
            else:
                self._set_buffer('retrieval', result)

        def _sync_input_buffers(self):
            # update input buffers
            self._set_buffer('perceptual', super().get_observation())

        def add_to_ltm(self, **kwargs):
            """Add a memory element to long-term memory.

            Arguments:
                **kwargs: The key-value pairs of the memory element.
            """
            self.ltm.store(**kwargs)

    return MemoryArchitectureMetaEnvironment
