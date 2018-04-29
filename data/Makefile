.PHONY: directories

directories: kbs models

kbs/umbel-ontology.n3: kbs
	wget --output-document $@ 'https://raw.githubusercontent.com/structureddynamics/UMBEL/master/Ontology/umbel.n3'

kbs/umbel-animals-typology.n3: kbs
	wget --output-document $@ 'https://raw.githubusercontent.com/structureddynamics/UMBEL/master/Typologies/Animals-typology.n3'

kbs/umbel-concepts-typology.n3: kbs
	wget --output-document $@ 'https://raw.githubusercontent.com/structureddynamics/UMBEL/master/Typologies/Concepts-typology.n3'

kbs/wordnet-3.1.n3: kbs
	wget --output-document wn31.nt.gz 'https://raw.githubusercontent.com/jrvosse/wordnet-3.0-rdf/master/rdf/rdf/wn31.nt.gz'
	gunzip wn31.nt.gz
	sort wn31.nt | uniq > $@
	rm -f wn31.nt.gz wn31.nt

kbs/mappingbased_objects_en.ttl: kbs
	wget --output-docoument $@.bz2 'http://downloads.dbpedia.org/2016-10/core-i18n/en/mappingbased_objects_en.ttl.bz2'
	bunzip2 $@.bz2

kbs/mappingbased_literals_en.ttl: kbs
	wget --output-docoument $@.bz2 'http://downloads.dbpedia.org/2016-10/core-i18n/en/mappingbased_literals_en.ttl.bz2'
	bunzip2 $@.bz2

models/GoogleNews-vectors-negative300.bin: models
	wget --output-document $@.gz 'https://github.com/mmihaltz/word2vec-GoogleNews-vectors/raw/master/GoogleNews-vectors-negative300.bin.gz'
	gunzip $@.gz

models:
	if [ ! -d 'models' ]; then mkdir 'models'; fi

kbs:
	if [ ! -d 'kbs' ]; then mkdir 'kbs'; fi

# opencyc-latest.owl
