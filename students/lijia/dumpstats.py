"""DumpStats is a class that organize and compute all the statistics of a dump (given extraction)"""

from os.path import join as join_path, exists as file_exists
from collections import defaultdict, Counter
from students.lijia.worddb import CondProbDict  # pylint disable=import-error
from students.lijia.utils import get_filename_from_folder  # pylint disable=import-error


class DumpStats:
    """a class for organizing all the statistics for a dump """
    def __init__(self, dump_dir, stat_dir):
        self.dump_dir = dump_dir
        self.stats_dir = stat_dir

        self.AN_extract_dir = join_path(self.stats_dir, "adj_noun")  # pylint: disable=invalid-name
        self.VN_extract_dir = join_path(self.stats_dir, "verb_noun")  # pylint: disable=invalid-name

        self.prob_verb_noun_file = join_path(self.stats_dir, 'prob_verb_noun.sqlite')
        self.prob_noun_verb_file = join_path(self.stats_dir, 'prob_verb_noun.sqlite')
        self.prob_noun_adj_file = join_path(self.stats_dir, 'prob_noun_adj.sqlite')
        self.prob_adj_noun_file = join_path(self.stats_dir, 'prob_adj_noun.sqlite')
        self.prob_verb_adj_file = join_path(self.stats_dir, 'prob_verb_adj.sqlite')

        self._prob_verb_noun_db = None
        self._prob_noun_verb_db = None
        self._prob_noun_adj_db = None
        self._prob_adj_noun_db = None
        self._prob_verb_adj_db = None

    @property
    def prob_verb_noun_db(self):
        """
        return p(verb|noun) database if already exist, else initiate database and return
        Returns:
            a instance of CondProbDict for p(verb|noun)
        """
        if self._prob_verb_noun_db is not None:
            pass
        elif file_exists(self.prob_verb_noun_file):
            self._prob_verb_noun_db = CondProbDict(self.prob_verb_noun_file)
        else:
            self._prob_verb_noun_db = self.calculate_prob_from_extraction(
                self.VN_extract_dir, self.prob_verb_noun_file, 1, 0)
        return self._prob_verb_noun_db

    @property
    def prob_noun_verb_db(self):
        """
        return p(noun|verb) database if already exist, else initiate database and return
        Returns:
            a instance of CondProbDict for p(noun|verb)
        """
        if self._prob_noun_verb_db is not None:
            pass
        elif file_exists(self.prob_noun_verb_file):
            self._prob_noun_verb_db = CondProbDict(self.prob_noun_verb_file)
        else:
            self._prob_noun_verb_db = self.calculate_prob_from_extraction(
                self.VN_extract_dir, self.prob_noun_verb_file, 0, 1)
        return self._prob_noun_verb_db

    @property
    def prob_noun_adj_db(self):
        """
        return p(noun|adj) database if already exist, else initiate database and return
        Returns:
            a instance of CondProbDict for p(noun|adj)
        """
        if self._prob_noun_adj_db is not None:
            pass
        elif file_exists(self.prob_noun_adj_file):
            self._prob_noun_adj_db = CondProbDict(self.prob_noun_adj_file)
        else:
            self._prob_noun_adj_db = self.calculate_prob_from_extraction(
                self.AN_extract_dir, self.prob_noun_adj_file, 0, 1)
        return self._prob_noun_adj_db

    @property
    def prob_adj_noun_db(self):
        """
        return p(adj|noun) database if already exist, else initiate database and return
        Returns:
            a instance of CondProbDict for p(adj|noun)
        """
        if self._prob_adj_noun_db is not None:
            pass
        elif file_exists(self.prob_adj_noun_file):
            self._prob_adj_noun_db = CondProbDict(self.prob_adj_noun_file)
        else:
            self._prob_adj_noun_db = self.calculate_prob_from_extraction(
                self.AN_extract_dir, self.prob_adj_noun_file, 1, 0)
        return self._prob_adj_noun_db

    @property
    def prob_verb_adj_db(self):
        """
        return p(verb|adj) database if already exist, else initiate database and return
        Returns:
            a instance of CondProbDict for p(verb|adj)
        """
        if self._prob_verb_adj_db is not None:
            pass
        elif file_exists(self.prob_verb_adj_file):
            self._prob_verb_adj_db = CondProbDict(self.prob_verb_adj_file)
        else:
            verb_adj_set = self.compute_verb_adj_pair()
            self.init_prob_verb_adj(verb_adj_set)
        return self._prob_verb_adj_db

    def compute_verb_adj_pair(self):
        """
        Compute the unique "verb adj" pair of the dump and return sorted tuple of the set
        Returns:
            (tuple) sorted "verb adj" set
        """
        verb_adj_set = set()
        for obj, _ in self.prob_verb_noun_db:
            for adj in self.prob_noun_adj_db.get_variable_dict(obj):
                for verb in self.prob_verb_noun_db.get_given_dict(obj):
                    verb_adj_set.add("%s %s" % (adj, verb))
        verb_adj_set = sorted(verb_adj_set)
        # cache the set
        with open(join_path(self.stats_dir, "verb_adj_set.txt"), 'w', encoding='utf-8') as file:
            for pair in verb_adj_set:
                file.write(pair + "\n")
        print("finish writing verb adj pair")
        return verb_adj_set

    def init_prob_verb_adj(self, verb_adj_set):
        print("initiating p(verb|adj) database")
        self._prob_verb_adj_db = CondProbDict(self.prob_verb_adj_file)
        for pair in verb_adj_set:
            adj, verb = pair.split()
            prob = sum(
                # P(V_O) * P(O_A)
                self.prob_verb_noun_db.get_probability(obj, verb) *
                self.prob_noun_adj_db.get_probability(adj, obj)
                for obj in self.prob_verb_noun_db.get_variable_dict(verb)
            )
            if not 0 <= prob <= 1:
                continue
            self._prob_verb_adj_db.add_probability(adj, verb, prob)

    def cache_prob(self, db_name, dict_count):
        """
        cache probability of the dictionary_counter
        Args:
            db_name (str): database's name
            dict_count (dict): a dictionary of counter as in {condition: {variable: count}}

        Returns:
            an instance of CondProbDict
        """
        if file_exists(db_name):
            print("failed to cache beacause, the file %s already exist" % db_name)
            return CondProbDict(db_name)
        dict_db = CondProbDict(db_name)
        for cond in dict_count:
            counter = dict_count[cond]
            total = sum(counter.values())
            for var in dict_count[cond]:
                dict_db.add_probability(cond, var, float(counter[var] / total), update=True)
        return dict_db

    def calculate_prob_from_extraction(self, extract_dir, db_name, cond_idx, var_idx):
        """
        Read from extraction folder and calculate the appearance of variable given condition
        Args:
            extract_dir (str): the directory which stores extraction
            db_name (str): database's name
            cond_idx (int): condition's index
            var_idx (int): variable's index

        Returns:
            cache_prob function
        """
        # read from a extract_dir to a nested dictionary
        file_gen = get_filename_from_folder(extract_dir)
        dict_counter = defaultdict(Counter)
        for file in file_gen:
            lines = [line for line in open(join_path(extract_dir, file), 'r', encoding='utf-8')]
            # create defaultdict in {condition: {variable: count}} format
            for line in lines:
                cond = line.split()[cond_idx]
                var = line.split()[var_idx]
                if dict_counter[cond]:
                    dict_counter[cond].update([var])
                else:
                    dict_counter[cond] = Counter([var])

        cache_count(db_name, dict_counter)
        return self.cache_prob(db_name, dict_counter)


def cache_count(db_filename, dict_counter):
    """
    Cache the count of the extraction
    Args:
        db_filename: the file name for sql
        dict_counter: dictionary nested with a counter as in {condition: {variable: count}}

    Returns:
        write out a count file for the given dictionary-counter
    """
    filename = db_filename[:-7].replace("prob", "count") + ".txt"
    if file_exists(filename):
        print("failed to cache beacause the file %s already exist" % filename)
        return
    with open(filename, 'w', encoding='utf-8') as file:
        for cond in dict_counter:
            counter = dict_counter[cond]
            file.write("%s (%s)\n" % (cond, sum(counter.values())))
            for var in dict_counter[cond]:
                file.write("---%s (%s)\n" % (var, counter[var]))

