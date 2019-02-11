from collections import defaultdict
from pathlib import Path

from research.knowledge_base import SparqlEndpoint

def make_select_statement(limit, offset):
    return f'''
        SELECT DISTINCT ?title ?release_date WHERE {{
            ?album <http://xmlns.com/foaf/0.1/name> ?title ;
                   <http://wikidata.dbpedia.org/ontology/releaseDate> ?release_date .
            FILTER ( lang(?title) = "en" )
        }} LIMIT {limit} OFFSET {offset}
    '''

def write_to_file(release_dates, filename):
    with Path(__file__).parent.joinpath(filename).open('w') as fd:
        for title, release_date_set in release_dates.items():
            last_date = max(release_date_set)
            fd.write(f'{title}\t{last_date}\n')

def main():
    endpoint = SparqlEndpoint('http://162.233.132.179:8890/sparql')
    limit = 100
    offset = 0
    titles = set()
    release_dates = defaultdict(set)
    results = endpoint.query_sparql(make_select_statement(limit, offset))
    while results:
        for result in results:
            title = result['title'].rdf_format
            release_date = result['release_date'].rdf_format
            release_dates[title].add(release_date)
        print(f'processed {offset + limit} results; total albums = {len(release_dates)}')
        offset += limit
        if offset % 1000 == 0:
            write_to_file( release_dates, 'albums')
        results = endpoint.query_sparql(make_select_statement(limit, offset))
    write_to_file(release_dates, 'albums')

if __name__ == '__main__':
    main()
