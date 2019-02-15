from textwrap import dedent

from research.knowledge_base import SparqlEndpoint
from research.rl_memory import SparqlKB

QUERY = dedent('''
    SELECT DISTINCT ?album_uri WHERE {
        ?track_uri <http://wikidata.dbpedia.org/ontology/album> ?album_uri .
        ?album_uri <http://xmlns.com/foaf/0.1/name> ?album_name ;
                   <http://wikidata.dbpedia.org/ontology/artist> ?artist_uri .
        ?artist_uri <http://wikidata.dbpedia.org/ontology/hometown> ?hometown_uri .
        ?hometown_uri <http://wikidata.dbpedia.org/ontology/country> ?country_uri .
        ?country_uri <http://xmlns.com/foaf/0.1/name> ?country_name .
        FILTER ( lang(?album_name) = "en" )
        FILTER ( lang(?country_name) = "en" )
    }
''').strip()


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

    filename = 'title_country'
    name_prop = '<http://xmlns.com/foaf/0.1/name>'

    with open(filename, 'w') as fd:
        fd.write('(\n')

    artists = {}
    hometowns = {}
    countries = {}
    for i, album_uri in enumerate(album_uris, start=1):
        result = kb_store.retrieve(album_uri).as_dict()
        album_name = result[name_prop]
        artist_uri = result['<http://wikidata.dbpedia.org/ontology/artist>']

        if artist_uri not in artists:
            result = kb_store.retrieve(artist_uri).as_dict()
            if '<http://wikidata.dbpedia.org/ontology/hometown>' not in result:
                continue
            hometown_uri = result['<http://wikidata.dbpedia.org/ontology/hometown>']
            artists[artist_uri] = hometown_uri
        else:
            hometown_uri = artists[artist_uri]

        if hometown_uri not in hometowns:
            result = kb_store.retrieve(hometown_uri).as_dict()
            if '<http://wikidata.dbpedia.org/ontology/country>' not in result:
                continue
            country_uri = result['<http://wikidata.dbpedia.org/ontology/country>']
            hometowns[hometown_uri] = country_uri
        else:
            country_uri = hometowns[hometown_uri]

        if country_uri not in countries:
            result = kb_store.retrieve(country_uri).as_dict()
            if name_prop not in result:
                continue
            country_name = result[name_prop]
        else:
            country_name = countries[country_uri]

        with open(filename, 'a') as fd:
            question = {name_prop: album_name,}
            answer = (country_name,)
            qna = (question, answer)
            fd.write('    ' + repr(qna))
            fd.write(',\n')

        if i % limit == 0:
            print(f'processed {i} albums')

    with open(filename, 'a') as fd:
        fd.write(')\n')


if __name__ == '__main__':
    main()
