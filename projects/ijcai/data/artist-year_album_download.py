import re
from textwrap import dedent

from research.knowledge_base import SparqlEndpoint
from research.rl_memory import SparqlKB

QUERY = dedent('''
    SELECT DISTINCT ?album_uri WHERE {
        ?track_uri <http://wikidata.dbpedia.org/ontology/album> ?album_uri .
        ?album_uri <http://xmlns.com/foaf/0.1/name> ?album_name ;
                   <http://wikidata.dbpedia.org/ontology/artist> ?artist_uri ;
                   <http://wikidata.dbpedia.org/ontology/releaseDate> ?release_date .
        FILTER ( lang(?album_name) = "en" )
    }
''').strip()


def first_letter(literal):
    match = re.fullmatch('"[^a-z]*([a-z]).*"(([@^][^"]*)?)', literal, flags=re.IGNORECASE)
    if match:
        initial = match.group(1).upper()
        metadata = match.group(2)
        return f'"{initial}"{metadata}'
    else:
        return None


def main():
    endpoint = SparqlEndpoint('http://162.233.132.179:8890/sparql')
    kb_store = SparqlKB(endpoint)

    album_uris = set()

    limit = 100
    offset = 0
    query = QUERY + f' LIMIT {limit} OFFSET {offset}'
    results = endpoint.query_sparql(query)
    while results:
        for result in results:
            album_uris.add(result['album_uri'].rdf_format)
        offset += limit
        print(f'processed {offset} results; total albums = {len(album_uris)}')
        query = QUERY + f' LIMIT {limit} OFFSET {offset}'
        results = endpoint.query_sparql(query)
    print(f'found {len(album_uris)} albums')

    filename = 'artist-year_album'
    name_pred = '<http://xmlns.com/foaf/0.1/name>'
    artist_pred = '<http://wikidata.dbpedia.org/ontology/artist>'
    date_pred = '<http://wikidata.dbpedia.org/ontology/releaseDate>'

    with open(filename, 'w') as fd:
        fd.write('(\n')

    for i, album_uri in enumerate(album_uris, start=1):
        result = kb_store.retrieve(album_uri).as_dict()
        album_name = result[name_pred]

        results = kb_store.retrieve(album_uri).as_dict()
        predicates = [name_pred, date_pred, artist_pred]
        if any(predicate not in result for predicate in predicates):
            continue
        artist_uri = result[artist_pred]
        release_date = result[date_pred]
        album_initial = first_letter(album_name)
        if artist_initial is None:
            continue

        result = kb_store.query({
            artist_pred: artist_uri,
            date_pred: release_date
        }).as_dict()
        if name_pred not in result or result[name_pred] != album_name:
            continue

        with open(filename, 'a') as fd:
            question = {
                artist_pred: artist_uri,
                date_pred: release_date,
            }
            answer = (
                album_initial,
            )
            qna = (question, answer)
            fd.write('    ' + repr(qna))
            fd.write(',\n')

        if i % limit == 0:
            print(f'processed {i} albums')

    with open(filename, 'a') as fd:
        fd.write(')\n')


if __name__ == '__main__':
    main()
