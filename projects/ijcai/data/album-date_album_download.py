import re
from textwrap import dedent

from research.knowledge_base import SparqlEndpoint
from research.rl_memory import SparqlKB

QUERY = dedent('''
    SELECT DISTINCT ?artist_uri WHERE {
        ?track <http://wikidata.dbpedia.org/ontology/album> ?album_uri .
        ?album_uri <http://xmlns.com/foaf/0.1/name> ?album_name ;
                    <http://wikidata.dbpedia.org/ontology/releaseDate> ?release_date ;
                    <http://wikidata.dbpedia.org/ontology/artist> ?artist_uri .
        ?other_track <http://wikidata.dbpedia.org/ontology/album> ?other_album_uri .
        ?other_album_uri <http://xmlns.com/foaf/0.1/name> ?other_album_name ;
                          <http://wikidata.dbpedia.org/ontology/releaseDate> ?other_release_date ;
                          <http://wikidata.dbpedia.org/ontology/artist> ?artist_uri .
        ?artist_uri <http://xmlns.com/foaf/0.1/name> ?artist_name .
        FILTER ( ?album_uri != ?other_album_uri )
        FILTER ( lang(?album_name) = "en" )
        FILTER ( lang(?other_album_name) = "en" )
    }
''').strip()

ALBUM_QUERY = dedent('''
    SELECT DISTINCT ?album_uri WHERE {{
        ?track <http://wikidata.dbpedia.org/ontology/album> ?album_uri .
        ?album_uri <http://xmlns.com/foaf/0.1/name> ?album_name ;
                    <http://wikidata.dbpedia.org/ontology/releaseDate> ?release_date ;
                    <http://wikidata.dbpedia.org/ontology/artist> {} .
        FILTER ( lang(?album_name) = "en" )
    }}
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

    artist_uris = set()

    limit = 100
    offset = 0
    query = QUERY + f' LIMIT {limit} OFFSET {offset}'
    results = endpoint.query_sparql(query)
    while results:
        for result in results:
            artist_uris.add(result['artist_uri'].rdf_format)
        offset += limit
        print(f'processed {offset} results; total artists = {len(artist_uris)}')
        query = QUERY + f' LIMIT {limit} OFFSET {offset}'
        results = endpoint.query_sparql(query)
    print(f'found {len(artist_uris)} artists')

    filename = 'album-date_album'
    name_pred = '<http://xmlns.com/foaf/0.1/name>'
    artist_pred = '<http://wikidata.dbpedia.org/ontology/artist>'
    date_pred = '<http://wikidata.dbpedia.org/ontology/releaseDate>'

    with open(filename, 'w') as fd:
        fd.write('(\n')

    for i, artist_uri in enumerate(artist_uris, start=1):

        result = kb_store.query({
            artist_pred: artist_uri,
        }).as_dict()
        predicates = [name_pred, date_pred, artist_pred]
        if any(predicate not in result for predicate in predicates):
            continue
        album_name = result[name_pred]

        results = endpoint.query_sparql(ALBUM_QUERY.format(artist_uri))
        for result in results:
            album_uri = result['album_uri'].rdf_format
            other_album = kb_store.retrieve(album_uri)
            other_album_name = other_album[name_pred]
            if other_album_name == album_name:
                continue

            release_date = other_album[date_pred]

            trial_result = kb_store.query({
                artist_pred: artist_uri,
                date_pred: release_date,
            })
            if trial_result[name_pred] != other_album_name:
                continue

            album_initial = first_letter(other_album_name)
            if album_initial is None:
                continue

            with open(filename, 'a') as fd:
                question = {
                    name_pred: album_name,
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
