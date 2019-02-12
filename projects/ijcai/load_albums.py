import re
from collections import defaultdict
from pathlib import Path

from research.knowledge_base import SparqlEndpoint


def write_to_file(schema, data):
    with Path(__file__).parent.joinpath('schemas', schema.name).open('w') as fd:
        for key, values in data.items():
            fd.write('\t'.join([
                *key,
                *(max(values[cat]) for cat in schema.categories),
            ]))
            fd.write('\n')


def get_schema_attr(schema, var):
    matches = [
        re.search(r'(?P<attr><[^>]*>) \?' + var + ' [;.]', line)
        for line in schema.sparql.splitlines()
    ]
    matches = [match for match in matches if match is not None]
    assert len(matches) == 1
    return matches[0].group('attr')


def fetch_data(schema):
    endpoint = SparqlEndpoint('http://162.233.132.179:8890/sparql')
    limit = 10
    offset = 0
    data = defaultdict(lambda: defaultdict(set))
    query = schema.sparql + f' LIMIT {limit} OFFSET {offset}'
    results = endpoint.query_sparql(query)
    while results:
        for result in results:
            key = tuple([result[clue].rdf_format for clue in schema.clues])
            for cat in schema.categories:
                data[key][cat].add(result[cat].rdf_format)
        offset += limit
        print(f'processed {offset} results; total albums = {len(data)}')
        if offset % 1000 == 0:
            write_to_file(schema, data)
        query = schema.sparql + f' LIMIT {limit} OFFSET {offset}'
        results = endpoint.query_sparql(query)
    print(f'processed {offset} results; total albums = {len(data)}')
    write_to_file(schema, data)

if __name__ == '__main__':
    from experiment2 import TITLE_COUNTRY
    fetch_data(TITLE_COUNTRY)
