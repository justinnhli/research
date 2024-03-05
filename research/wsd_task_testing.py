from wsd_task import *

print("go 1")
print(run_wsd("frequency", iterations=2))
print(run_wsd("context_sense", iterations=2))
print(run_wsd("context_word", iterations=2))
print(run_wsd("naive_semantic", num_sentences=200, iterations=2))
print(run_wsd("naive_semantic_spreading", num_sentences=200, iterations=2))
