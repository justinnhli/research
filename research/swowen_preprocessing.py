# Preprocessing SWOWEN data to enter into a semantic network to conduct spreading
import csv
from collections import defaultdict
import json

file = open('/Users/lilygebhart/Downloads/SWOWEN_data/SWOW-EN.R100.csv', mode='r')
csvFile = csv.reader(file)
next(csvFile, None)
swowen_dict = defaultdict(set)
for entries in csvFile:
    word = entries[-4]
    for i in [-1,-2,-3]:
        if entries[i] != 'NA':
            swowen_dict[word].update([entries[i]])
for word in swowen_dict.keys():
    swowen_dict[word] = list(swowen_dict[word])

swowen_file = open('/Users/lilygebhart/Downloads/SWOWEN_data/SWOWEN_spreading_sample.json', 'w')
json.dump(swowen_dict, swowen_file)
swowen_file.close()