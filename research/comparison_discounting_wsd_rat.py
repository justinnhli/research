import matplotlib.pyplot as plt
import statistics
import json
import os
import matplotlib.ticker as mtick

def compute_discounted_accuracy_wsd(filename):
    # filename should be a directory filename, with 6 files in it.
    files = os.listdir(filename)
    cumulative_acc = 0
    num_words = 0
    for file in files:
        if "sents" in file:
            total_guesses = 0
            discounted_count = 0
            guess_list = json.load(open(filename + "/" + file))
            for word_guess in guess_list:
                guesses = word_guess[1]
                for guess in guesses:
                    total_guesses += 1
                    if any(guess):
                        discounted_count += 1 / len(guess)
            partition_words = len(guess_list)
            num_words += partition_words
            cumulative_acc += (discounted_count / total_guesses) * partition_words
    return cumulative_acc / num_words

def compute_discounted_accuracy_rat(filename):
    guess_list = json.load(open(filename))
    total_guesses = 0
    discounted_count = 0
    for word_guess in guess_list:
        guesses = word_guess[1]
        total_guesses += 1
        if guesses is None:
            continue
        if guesses.count(word_guess[0].upper()) > 0:
            discounted_count += 1/len(guesses)
    return discounted_count / total_guesses

#WSD accuracy
wsd_cooc_sense = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/cooc_sense"
wsd_cooc_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/cooc_word"

wsd_sem_nospread_sentence = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/sem_nospread_sentence"
wsd_sem_nospread_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/sem_nospread_word"
wsd_sem_nospread_never = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/sem_nospread_never"
wsd_sem_never = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/sem_spread_never"
wsd_sem_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/sem_spread_word"
wsd_sem_sentence = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/sem_spread_sentence"

wsd_freq = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/frequency"

wsd_stc_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/semthreshcooc_word"
wsd_stc_sense = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/semthreshcooc_sense"

wsd_cts_word_never = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/coocthreshsem_word_never"
wsd_cts_word_sentence = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/coocthreshsem_word_sentence"
wsd_cts_word_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/coocthreshsem_word_word"
wsd_cts_sense_never = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/coocthreshsem_sense_never"
wsd_cts_sense_sentence = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/coocthreshsem_sense_sentence"
wsd_cts_sense_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/coocthreshsem_sense_word"

wsd_cts_corpus_word_never = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/coocthreshsem_corpus_word_never"
wsd_cts_corpus_word_sentence = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/coocthreshsem_corpus_word_sentence"
wsd_cts_corpus_word_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/coocthreshsem_corpus_word_word"
wsd_cts_corpus_sense_never = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/coocthreshsem_corpus_sense_never"
wsd_cts_corpus_sense_sentence = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/coocthreshsem_corpus_sense_sentence"
wsd_cts_corpus_sense_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/coocthreshsem_corpus_sense_word"

wsd_joint_word_never = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointprob_word_never"
wsd_joint_word_sentence = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointprob_word_sentence"
wsd_joint_word_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointprob_word_word"
wsd_joint_sense_never = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointprob_sense_never"
wsd_joint_sense_sentence = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointprob_sense_sentence"
wsd_joint_sense_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointprob_sense_word"

wsd_varstdev_word_never = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointvarstdev_word_never"
wsd_varstdev_word_sentence = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointvarstdev_word_sentence"
wsd_varstdev_word_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointvarstdev_word_word"
wsd_varstdev_sense_never = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointvarstdev_sense_never"
wsd_varstdev_sense_sentence = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointvarstdev_sense_sentence"
wsd_varstdev_sense_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointvarstdev_sense_word"

wsd_varmaxdiff_word_never = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointvarmaxdiff_word_never"
wsd_varmaxdiff_word_sentence = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointvarmaxdiff_word_sentence"
wsd_varmaxdiff_word_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointvarmaxdiff_word_word"
wsd_varmaxdiff_sense_never = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointvarmaxdiff_sense_never"
wsd_varmaxdiff_sense_sentence = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointvarmaxdiff_sense_sentence"
wsd_varmaxdiff_sense_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/jointvarmaxdiff_sense_word"

wsd_addprob_word_never = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/additiveprob_word_word"
wsd_addprob_word_sentence = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/additiveprob_word_word"
wsd_addprob_word_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/additiveprob_word_word"
wsd_addprob_sense_never = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/additiveprob_sense_never"
wsd_addprob_sense_sentence = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/additiveprob_sense_sentence"
wsd_addprob_sense_word = "/Users/lilygebhart/Documents/GitHub/research/research/results_plot/WSD/additiveprob_sense_word"



# RAT Links
rat_cooc = "/Users/lilygebhart/Documents/GitHub/research/research/results/rat_test_results_swowen_cooc.txt"

rat_spread_comb = "/Users/lilygebhart/Documents/GitHub/research/research/results/rat_semantic_combined.json"
rat_spread_sffan = "/Users/lilygebhart/Documents/GitHub/research/research/results/rat_semantic_SFFAN.json"
rat_spread_swowen = "/Users/lilygebhart/Documents/GitHub/research/research/results/rat_semantic_SWOWEN.json"

rat_stc_comb = "/Users/lilygebhart/Documents/GitHub/research/research/results/rat_sem_thresh_cooc_combined.json"
rat_stc_sffan = "/Users/lilygebhart/Documents/GitHub/research/research/results/rat_test_results_semthreshcooc_sffan.txt"
rat_stc_swowen = "/Users/lilygebhart/Documents/GitHub/research/research/results/rat_test_results_swowen_semthreshcooc.txt"

rat_cts_comb = "/Users/lilygebhart/Documents/GitHub/research/research/results/rat_cooc_thresh_sem_combined.json"
rat_cts_sffan = "/Users/lilygebhart/Documents/GitHub/research/research/results/rat_cooc_thresh_sem_SFFAN.json"
rat_cts_swowen = "/Users/lilygebhart/Documents/GitHub/research/research/results/rat_cooc_thresh_sem_SWOWEN.json"

rat_joint_comb = ""
rat_joint_sffan = "/Users/lilygebhart/Documents/GitHub/research/research/results/rat_joint_probability_SFFAN.json"
rat_joint_swowen = ""

rat_varstdev_comb = ""
rat_varstdev_sffan = "/Users/lilygebhart/Documents/GitHub/research/research/results/rat_joint_variance_stdev_SFFAN.json"
rat_varstdev_swowen = ""

rat_varmaxdiff_comb = ""
rat_varmaxdiff_sffan = "/Users/lilygebhart/Documents/GitHub/research/research/results/rat_joint_variance_maxdiff_SFFAN.json"
rat_varmaxdiff_swowen = ""

rat_addprob_comb = ""
rat_addprob_sffan = "/Users/lilygebhart/Documents/GitHub/research/research/results/rat_additive_probability_SFFAN.json"
rat_addprob_swowen = ""


# Make plot...
wsd_results_lb = 0.3794224297
rat_results_lb = 0
wsd_results_ub = 0.98977029
rat_results_ub = 0.3450704225

fig, ax = plt.subplots()
ax.set_title("RAT vs. WSD Accuracy for Retrieval Mechanisms")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_ylabel("RAT Accuracy")
ax.set_xlabel("WSD Accuracy")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

ax.scatter(wsd_results_ub, rat_results_ub, marker='^', color="blue", s=80, label="Oracle")
ax.scatter(wsd_results_lb, rat_results_lb, marker='^', color="purple", s=80, label="Uniform Random")

print("cooccurrence")
rat_list = []
wsd_list = []
for rat in [rat_cooc]:
    for wsd in [wsd_cooc_sense, wsd_cooc_word]:
        rat_coord = compute_discounted_accuracy_rat(rat)
        wsd_coord = compute_discounted_accuracy_wsd(wsd)
        print("RAT", rat_coord, ", WSD", wsd_coord)
        rat_list.append(rat_coord)
        wsd_list.append(wsd_coord)
ax.scatter(wsd_list, rat_list, marker='o', color="tab:blue", label="Co-occurrence")

print("spreading")
rat_list = []
wsd_list = []
for rat in [rat_spread_sffan]:
    for wsd in [wsd_sem_sentence, wsd_sem_word, wsd_sem_never]:
        rat_coord = compute_discounted_accuracy_rat(rat)
        wsd_coord = compute_discounted_accuracy_wsd(wsd)
        print("RAT", rat_coord, ", WSD", wsd_coord)
        rat_list.append(rat_coord)
        wsd_list.append(wsd_coord)
ax.scatter(wsd_list, rat_list, marker='o', color="tab:red", label="Semantic Spreading")

print("frequency")
rat_list = []
wsd_list = []
for rat in [0]:
    for wsd in [wsd_freq]:
        rat_coord = 0
        wsd_coord = compute_discounted_accuracy_wsd(wsd)
        print("RAT", rat_coord, ", WSD", wsd_coord)
        rat_list.append(rat_coord)
        wsd_list.append(wsd_coord)
ax.scatter(wsd_list, rat_list, marker='o', color="peru", label="A Priori Frequency")

print("spreading - no spreading")
rat_list = []
wsd_list = []
for wsd in [wsd_sem_nospread_word, wsd_sem_nospread_sentence, wsd_sem_nospread_never]:
    rat_coord = 0
    wsd_coord = compute_discounted_accuracy_wsd(wsd)
    print("RAT", rat_coord, ", WSD", wsd_coord)
    rat_list.append(rat_coord)
    wsd_list.append(wsd_coord)
ax.scatter(wsd_list, rat_list, marker='o', color="tab:orange", label="Act-Based Frequency")

print("cts")
rat_list = []
wsd_list = []
for rat in [rat_cts_sffan]:
    for wsd in [wsd_cts_sense_never, wsd_cts_sense_sentence, wsd_cts_sense_word,
                wsd_cts_word_never, wsd_cts_word_sentence, wsd_cts_word_word,
                wsd_cts_corpus_sense_never, wsd_cts_corpus_sense_sentence, wsd_cts_corpus_sense_word,
                wsd_cts_corpus_word_never, wsd_cts_corpus_word_sentence, wsd_cts_corpus_word_word]:
        rat_coord = compute_discounted_accuracy_rat(rat)
        wsd_coord = compute_discounted_accuracy_wsd(wsd)
        print("RAT", rat_coord, ", WSD", wsd_coord)
        rat_list.append(rat_coord)
        wsd_list.append(wsd_coord)
ax.scatter(wsd_list, rat_list, marker='o', color="tab:pink", label="CTS")

print("stc")
rat_list = []
wsd_list = []
for rat in [rat_stc_sffan]:
    for wsd in [wsd_stc_sense, wsd_stc_word]:
        rat_coord = compute_discounted_accuracy_rat(rat)
        wsd_coord = compute_discounted_accuracy_wsd(wsd)
        print("RAT", rat_coord, ", WSD", wsd_coord)
        rat_list.append(rat_coord)
        wsd_list.append(wsd_coord)
ax.scatter(wsd_list, rat_list, marker='o', color="tab:purple", label="STC")


print("joint")
rat_list = []
wsd_list = []
for rat in [rat_joint_sffan]:
    for wsd in [wsd_joint_sense_never, wsd_joint_word_never,
                wsd_joint_word_sentence, wsd_joint_sense_sentence,
                wsd_joint_sense_word, wsd_joint_word_word]:
        rat_coord = compute_discounted_accuracy_rat(rat)
        wsd_coord = compute_discounted_accuracy_wsd(wsd)
        print("RAT", rat_coord, ", WSD", wsd_coord)
        rat_list.append(rat_coord)
        wsd_list.append(wsd_coord)
ax.scatter(wsd_list, rat_list, marker='o', color="tab:green", label="Joint")

print("joint variance: st dev")
rat_list = []
wsd_list = []
for rat in [rat_varstdev_sffan]:
    for wsd in [wsd_varstdev_sense_never, wsd_varstdev_word_never,
                wsd_varstdev_word_sentence, wsd_varstdev_sense_sentence,
                wsd_varstdev_sense_word, wsd_varstdev_word_word]:
        rat_coord = compute_discounted_accuracy_rat(rat)
        wsd_coord = compute_discounted_accuracy_wsd(wsd)
        print("RAT", rat_coord, ", WSD", wsd_coord)
        rat_list.append(rat_coord)
        wsd_list.append(wsd_coord)
ax.scatter(wsd_list, rat_list, marker='o', color="tab:olive", label="Variance - St Dev")


print("joint variance: max diff")
rat_list = []
wsd_list = []
for rat in [rat_varmaxdiff_sffan]:
    for wsd in [wsd_varmaxdiff_sense_never, wsd_varmaxdiff_word_never,
                wsd_varmaxdiff_word_sentence, wsd_varmaxdiff_sense_sentence,
                wsd_varmaxdiff_sense_word, wsd_varmaxdiff_word_word]:
        rat_coord = compute_discounted_accuracy_rat(rat)
        wsd_coord = compute_discounted_accuracy_wsd(wsd)
        print("RAT", rat_coord, ", WSD", wsd_coord)
        rat_list.append(rat_coord)
        wsd_list.append(wsd_coord)
ax.scatter(wsd_list, rat_list, marker='o', color="tab:cyan", label="Variance - Max Diff")

print("additive")
rat_list = []
wsd_list = []
for rat in [rat_addprob_sffan]:
    for wsd in [wsd_addprob_sense_never, wsd_addprob_word_never,
                wsd_addprob_word_sentence, wsd_addprob_sense_sentence,
                wsd_addprob_sense_word, wsd_addprob_word_word]:
        rat_coord = compute_discounted_accuracy_rat(rat)
        wsd_coord = compute_discounted_accuracy_wsd(wsd)
        print("RAT", rat_coord, ", WSD", wsd_coord)
        rat_list.append(rat_coord)
        wsd_list.append(wsd_coord)
ax.scatter(wsd_list, rat_list,  marker='o', color="tab:brown", label="Additive")

ax.legend(loc="upper left", prop={'size': 8})
plt.show()
