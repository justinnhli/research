import re
from textwrap import dedent

from research.knowledge_base import SparqlEndpoint
from research.rl_memory import SparqlKB

QUERY = dedent('''
    SELECT DISTINCT ?album_uri WHERE {
        ?track_uri <http://wikidata.dbpedia.org/ontology/album> ?album_uri .
        ?album_uri <http://xmlns.com/foaf/0.1/name> ?album_name ;
                   <http://wikidata.dbpedia.org/ontology/artist> ?artist_uri .
        ?artist_uri <http://xmlns.com/foaf/0.1/name> ?artist_name .
        FILTER ( lang(?album_name) = "en" )
        FILTER ( lang(?artist_name) = "en" )
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

    filename = 'title_artist'
    name_prop = '<http://xmlns.com/foaf/0.1/name>'

    with open(filename, 'w') as fd:
        fd.write('(\n')

    artists = {}
    for i, album_uri in enumerate(album_uris, start=1):
        result = kb_store.retrieve(album_uri).as_dict()
        album_name = result[name_prop]
        artist_uri = result['<http://wikidata.dbpedia.org/ontology/artist>']

        if artist_uri not in artists:
            result = kb_store.retrieve(artist_uri).as_dict()
            if '<http://xmlns.com/foaf/0.1/name>' not in result:
                continue
            artist_name = result['<http://xmlns.com/foaf/0.1/name>']
            artists[artist_uri] = artist_name
        else:
            artist_name = artists[artist_uri]
        artist_initial = first_letter(artist_name)

        with open(filename, 'a') as fd:
            question = {name_prop: album_name,}
            answer = (artist_initial,)
            qna = (question, answer)
            fd.write('    ' + repr(qna))
            fd.write(',\n')

        if i % limit == 0:
            print(f'processed {i} albums')

    with open(filename, 'a') as fd:
        fd.write(')\n')


if __name__ == '__main__':
    main()
