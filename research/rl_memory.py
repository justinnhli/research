"""Memory architecture for reinforcement learning."""

from collections import namedtuple

from .data_structures import AVLTree
from .rl_environments import State, Action, Environment


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
                *args, **kwargs,
        ): # noqa: D102
            """Initialize a memory architecture.

            Arguments:
                ltm (LongTermMemory): The memory model to use.
                buf_ignore (Iterable[str]): Buffers that should not be created.
                internal_reward (float): Reward for internal actions. Defaults to -0.1.
                max_internal_actions (int): Max number of consecutive internal actions. Defaults to None.
                *args: Arbitrary positional arguments.
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
            # initialization
            self._clear_buffers()
            super().__init__(*args, **kwargs)

        @property
        def slots(self):
            """Yield all values of all attributes in all buffers.

            Yields:
                Tuple[str, str, Any]: A tuple of buffer, attribute, and value.
            """
            for buf, attrs in sorted(self.buffers.items()):
                for attr, val in attrs.items():
                    yield buf, attr, val

        def to_dict(self):
            """Convert the state into a dictionary."""
            return {buf + '_' + attr: val for buf, attr, val in self.slots}

        def get_state(self): # noqa: D102
            # pylint: disable = missing-docstring
            return State(**self.to_dict())

        def get_observation(self): # noqa: D102
            # pylint: disable = missing-docstring
            return State(**self.to_dict())

        def reset(self): # noqa: D102
            # pylint: disable = missing-docstring
            super().reset()
            self._clear_buffers()

        def _clear_buffers(self):
            self.buffers = {}
            for buf, _ in self.BUFFERS.items():
                if buf in self.buf_ignore:
                    continue
                self.buffers[buf] = AVLTree()

        def _clear_ltm_buffers(self):
            self.buffers['query'].clear()
            self.buffers['retrieval'].clear()

        def start_new_episode(self): # noqa: D102
            # pylint: disable = missing-docstring
            super().start_new_episode()
            self._clear_buffers()
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
                for attr in self.buffers[src_buf]:
                    for dst_buf, dst_prop in self.BUFFERS.items():
                        copyable = (
                            src_buf != dst_buf
                            and dst_buf not in self.buf_ignore
                            and dst_prop.writable
                            and not (src_buf == 'perceptual' and dst_buf == 'scratch')
                            and self.buffers[src_buf][attr] != self.buffers[dst_buf].get(attr, None)
                        )
                        if not copyable:
                            continue
                        actions.append(Action(
                            'copy',
                            src_buf=src_buf,
                            src_attr=attr,
                            dst_buf=dst_buf,
                            dst_attr=attr,
                        ))
            return actions

        def _generate_delete_actions(self):
            actions = []
            for buf, prop in self.BUFFERS.items():
                if buf in self.buf_ignore or not prop.writable:
                    continue
                for attr in self.buffers[buf]:
                    actions.append(Action(
                        'delete',
                        buf=buf,
                        attr=attr,
                    ))
            return actions

        def _generate_retrieve_actions(self):
            actions = []
            for buf, buf_props in self.BUFFERS.items():
                if buf in self.buf_ignore or not buf_props.copyable:
                    continue
                for attr, value in self.buffers[buf].items():
                    if self.ltm.retrievable(value):
                        actions.append(Action('retrieve', buf=buf, attr=attr))
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
            assert action in self.get_actions(), f'{action} not in {self.get_actions()}'
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
                val = self.buffers[action.src_buf][action.src_attr]
                self.buffers[action.dst_buf][action.dst_attr] = val
                if action.dst_buf == 'query':
                    self._query_ltm()
            elif action.name == 'delete':
                del self.buffers[action.buf][action.attr]
                if action.buf == 'query':
                    self._query_ltm()
            elif action.name == 'retrieve':
                result = self.ltm.retrieve(self.buffers[action.buf][action.attr])
                self.buffers['query'].clear()
                if result is None:
                    self.buffers['retrieval'].clear()
                else:
                    self.buffers['retrieval'] = result
            elif action.name == 'prev-result':
                self.buffers['retrieval'] = self.ltm.prev_result()
            elif action.name == 'next-result':
                self.buffers['retrieval'] = self.ltm.next_result()
            else:
                return True
            return False

        def _query_ltm(self):
            if not self.buffers['query']:
                self.buffers['retrieval'].clear()
                return
            result = self.ltm.query(self.buffers['query'])
            if result is None:
                self.buffers['retrieval'].clear()
            else:
                self.buffers['retrieval'] = result

        def _sync_input_buffers(self):
            # update input buffers
            self.buffers['perceptual'] = super().get_observation()

        def add_to_ltm(self, **kwargs):
            """Add a memory element to long-term memory.

            Arguments:
                **kwargs: The key-value pairs of the memory element.
            """
            self.ltm.store(**kwargs)

    return MemoryArchitectureMetaEnvironment
