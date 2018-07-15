import re
import sys
from os.path import dirname, realpath, join as join_path, exists as file_exists
from collections import defaultdict, namedtuple, Counter
from students.lijia.worddb import CondProbDict
from students.lijia.utils import get_filename_from_folder

# todo(Lijia): looging

VPO = namedtuple('VPO', ('verb', 'prep', 'object'))
NP = namedtuple('NP', ['noun', 'adjectives'])


class DumpStats:
    """a class for organizing all the statistics for a dump """
    def __init__(self, dump_dir, stat_dir):
        self.dump_dir = dump_dir
        self.stat_dir = stat_dir # todo: how should dump_dir and stat_dir work?

        self.adj_noun_extract_folder = join_path(self.stat_dir, "adj_noun")
        self.verb_noun_extract_folder = join_path(self.stat_dir, "verb_noun")

        self.prob_verb_noun_file = join_path(self.stat_dir, 'prob_verb_noun.sqlite')
        self.prob_noun_verb_file = join_path(self.stat_dir, 'prob_verb_noun.sqlite')
        self.prob_noun_adj_file = join_path(self.stat_dir, 'prob_noun_adj.sqlite')
        self.prob_adj_noun_file = join_path(self.stat_dir, 'prob_adj_noun.sqlite')
        self.prob_verb_adj_file = join_path(self.stat_dir, 'prob_verb_adj.sqlite')

        self._prob_verb_noun_db = None
        self._prob_noun_verb_db = None
        self._prob_noun_adj_db = None
        self._prob_adj_noun_db = None


    @property
    def prob_verb_noun_db(self):
        if self._prob_verb_noun_db is not None:
            return self._prob_verb_noun_db
        else:
            self._prob_verb_noun_db = self.calculate_prob_from_extraction(
                self.verb_noun_extract_folder, self.prob_verb_noun_file, 1, 0)
            return self._prob_verb_noun_db

    @property
    def prob_noun_verb_db(self):
        if self._prob_noun_verb_db is not None:
            return self._prob_noun_verb_db
        else:
            self._prob_noun_verb_db = self.calculate_prob_from_extraction\
                (self.verb_noun_extract_folder, self.prob_noun_verb_file, 0, 1)
            return self._prob_noun_verb_db

    @property
    def prob_noun_adj_db(self):
        if self._prob_noun_adj_db is not None:
            return self._prob_noun_adj_db
        else:
            self._prob_noun_adj_db = self.calculate_prob_from_extraction(
                self.adj_noun_extract_folder, self.prob_noun_adj_file, 0, 1)
            return self._prob_noun_adj_db

    @property
    def prob_adj_noun_db(self):
        if self._prob_adj_noun_db is not None:
            return self._prob_adj_noun_db
        else:
            self._prob_adj_noun_db = self.calculate_prob_from_extraction(
                self.adj_noun_extract_folder, self.prob_adj_noun_file, 1, 0)
            return self._prob_adj_noun_db

    def cache_prob(self, filename, count_dict):
        if file_exists(filename):
            print("failed to cache beacause, the file %s already exist" % filename)
            return
        d = CondProbDict(self.prob_verb_noun_file)
        for condition in count_dict:
            c = count_dict[condition]
            total = sum(c.values())
            for variable in count_dict[condition]:
                d.add_probability(condition, variable, float(c[variable] / total))
        return d

    def calculate_prob_from_extraction(self, directory, db_name, outer_index, inner_index):
        """
        a generic function that calculate probability and count
        :param directory: the folder that contains extracted words
        :param output_filename
        :param outer_index: the index for condition (default dictionary's key)
        :param inner_index: the index for marginal (counter's key)
        :return:
        """
        # FIXME: check for existance
        # if file_exists(join_path(OUTPUT_DIR, "count_" + output_filename)) \
        #         and file_exists(join_path(OUTPUT_DIR, "prob_" + output_filename)):
        #     print("both count and probability for (%s) exist" % output_filename)
        #     return

        # read from a directory to a nested dictionary
        file_gen = get_filename_from_folder(directory)
        d = defaultdict(Counter)
        i = 0
        for file in file_gen:
            i += 1
            lines = [line for line in open(join_path(directory, file), 'r', encoding='utf-8')]
            # add to defaultdict with x as key and a counter as value
            # the counter counts y's appearance
            for line in lines:
                x = line.split()[outer_index]
                y = line.split()[inner_index]
                if d[x]:
                    d[x].update([y])
                else:
                    d[x] = Counter([y])

        cache_count(db_name, d)
        print("Finished writing count. Total file processed: ", i)
        return self.cache_prob(db_name, d)


def cache_count(filename, count_dict):
    """ write out counts"""
    filename = "count_" + filename[:-7] + ".txt"
    if file_exists(filename):
        print("failed to cache beacause, the file %s already exist" % filename)
        return
    with open(filename, 'w', encoding='utf-8') as f:
        for k in count_dict:
            c = count_dict[k]
            f.write("%s (%s)\n" % (k, sum(c.values())))
            for word in count_dict[k]:
                f.write("---%s (%s)\n" % (word, c[word]))