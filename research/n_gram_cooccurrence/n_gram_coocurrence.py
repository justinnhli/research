from collections import defaultdict


def clean_ngram_word(word):
    if not word.isalpha():
        if "_" not in word:
            return None
        word = word.split("_")[0]
    return word

def get_pairwise_combos(word_list):
    pairs = []
    for i in range(1, len(word_list)):
        for j in range(len(i)):
            pairs.append[word_list[i], word_list[j]]
    return pairs

def get_ngram_cooccurrences(dir_link):
    ngrams = open(dir_link)
    ngram_coocs = defaultdict(int)
    for line in ngrams:
        entries = line.split('\t')
        if len(entries) != 4:
            continue
        words = entries[0].split(' ')
        pairs = get_pairwise_combos(words)
        for pair in pairs:
            word1 = clean_ngram_word(pair[0])
            word2 = clean_ngram_word(pair[1])
            print(entries)
            if word1 is None or word2 is None or word1 == "" or word2 == "" or not word1.isalnum() or not word2.isalnum():
                continue
            ngram_coocs[tuple(sorted([word1, word2]))] += int(entries[3])
    return ngram_coocs

dict = get_ngram_cooccurrences()
print(dict.items())