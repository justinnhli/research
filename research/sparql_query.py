import re
from textwrap import indent, dedent

from rdflib.plugins.sparql import prepareQuery

from .knowledge_base import URI


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
        for value in self.props.values():
            if isinstance(value, str):
                desc_vars = ['?{}'.format(value)]
            else:
                desc_vars = value.variables()
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
        raise ValueError('unknown URI format: ' + string)


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
        for bound in bindings:
            if '?' + bound in variables:
                variables.remove('?' + bound)
        return template.format(
            variables=' '.join(variables),
            triples=indent(self.to_triples_str(**bindings), '    '),
            constraints='\n'.join(constraints)
        )

    def as_prepared_query(self, *constraints, **bindings):
        return prepareQuery(self.to_select(*constraints, **bindings))
