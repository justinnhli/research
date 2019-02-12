from collections import namedtuple
from textwrap import dedent

Schema = namedtuple('Schema', 'name sparql clues categories')

SCHEMAS = {
    'title_year': Schema(
        'title_year',
        dedent('''
            SELECT DISTINCT ?title ?release_date WHERE {
                ?track <http://wikidata.dbpedia.org/ontology/album> ?album .
                ?album <http://xmlns.com/foaf/0.1/name> ?title ;
                       <http://wikidata.dbpedia.org/ontology/artist> ?artist_node ;
                       <http://wikidata.dbpedia.org/ontology/releaseDate> ?release_date .
                FILTER ( lang(?title) = "en" )
            }
        ''').strip(),
        ['title',],
        ['release_date',],
    ),
    'title_genre_decade': Schema(
        'title_genre_decade',
        dedent('''
            SELECT DISTINCT ?title ?genre ?release_date WHERE {
                ?track <http://wikidata.dbpedia.org/ontology/album> ?album .
                ?album <http://xmlns.com/foaf/0.1/name> ?title ;
                       <http://wikidata.dbpedia.org/ontology/artist> ?artist_node ;
                       <http://wikidata.dbpedia.org/ontology/genre> ?genre_node ;
                       <http://wikidata.dbpedia.org/ontology/releaseDate> ?release_date .
                ?genre_node <http://xmlns.com/foaf/0.1/name> ?genre .
                FILTER ( lang(?title) = "en" )
                FILTER ( lang(?genre) = "en" )
            }
        ''').strip(),
        ['title',],
        ['genre', 'release_date',],
    ),
    'title_country': Schema(
        'title_country',
        dedent('''
            SELECT DISTINCT ?title ?country WHERE {
                ?track <http://wikidata.dbpedia.org/ontology/album> ?album .
                ?album <http://xmlns.com/foaf/0.1/name> ?title ;
                            <http://wikidata.dbpedia.org/ontology/artist> ?artist .
                ?artist <http://wikidata.dbpedia.org/ontology/hometown> ?hometown .
                ?hometown <http://wikidata.dbpedia.org/ontology/country> ?country_node .
                ?country_node <http://xmlns.com/foaf/0.1/name> ?country .
                FILTER ( lang(?title) = "en" )
                FILTER ( lang(?country) = "en" )
            }
        ''').strip(),
        ['title',],
        ['country',],
    ),
}
