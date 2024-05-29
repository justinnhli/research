import csv
from collections import defaultdict
import json

"""
Script for making a sample dictionary to test my code on before running in the cluster. 
"""

sample_link = '/Users/lilygebhart/Downloads/south_florida_free_assoc_norms/sf_free_assoc_1.rtf'
sf_dict = defaultdict(set)
file = open(sample_link, mode='r')
csvFile = csv.reader(file)
counter = 0
for row in csvFile:
    print("row", row)
    counter +=1
    if counter > 10 and len(row) > 1:
        word = row[0].lower()
        sf_dict[word].update([row[1][1:].lower()])
        print(word)
        print(sf_dict[word])
for word in sf_dict.keys():
    sf_dict[word] = list(sf_dict[word])
sf_file = open('/Users/lilygebhart/Downloads/south_florida_free_assoc_norms/sf_spreading_sample.json', 'w')
json.dump(sf_dict, sf_file)
sf_file.close()