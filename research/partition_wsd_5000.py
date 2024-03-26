# Gets WSD Accuracies for the following tasks for all six 5,000 word partitions available in the
# SemCor corpus:
# (1) Frequency
# (2) Context Sense
# (3) Context Word
# (4) Semantics (No Spreading)
# (5) Semantics (Spreading, Never Clear)
# (6) Semantics (Spreading, Clear After Sentence)
# (7) Semantics (Spreading, Clear After Word)
from wsd_task import *

num_sents = 5000

for partition in range(1,7):
    print(partition)
    print(run_wsd("frequency", num_sentences=num_sents, partition=partition))
    print(run_wsd("context_sense", num_sentences=num_sents, partition=partition))
    print(run_wsd("context_word", num_sentences=num_sents, partition=partition))
    print(run_wsd("naive_semantic", num_sentences=num_sents, partition=partition))
    print(run_wsd("naive_semantic_spreading", num_sentences=num_sents, partition=partition, clear_network="never"))
    print(run_wsd("naive_semantic_spreading", num_sentences=num_sents, partition=partition, clear_network="sentence"))
    print(run_wsd("naive_semantic_spreading", num_sentences=num_sents, partition=partition, clear_network="word"))
    print()