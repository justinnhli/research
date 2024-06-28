import matplotlib.pyplot as plt
import statistics
from matplotlib.patches import Rectangle

def plot_results(rat_dict, wsd_dict):
    for rat_key in rat_dict.keys():
        rat_result = rat_dict[rat_key]
        rat_point = statistics.mean(rat_result)
        rat_err = rat_result[1] - rat_point
        for wsd_key in wsd_dict.keys():
            wsd_result = wsd_dict[wsd_key]
            wsd_point = statistics.mean(wsd_result)
            wsd_err = wsd_result[1] - wsd_point
            plt.errorbar(x=rat_point, y=wsd_point, yerr=wsd_err, xerr=rat_err,
                         capsize=2, ecolor="black")
            plt.plot([rat_point], [wsd_point], marker='o', color="black")

def plot_max_results(rat_dict, wsd_dict):
    for rat_key in rat_dict.keys():
        rat_result = rat_dict[rat_key][1]
        for wsd_key in wsd_dict.keys():
            wsd_result = wsd_dict[wsd_key][1]
            plt.plot(rat_result, wsd_result, marker='o', color="black")

def plot_category_rectangle(rat_dict, wsd_dict, ax):
    rat_min = 1
    wsd_min = 1
    wsd_max = 0
    rat_max = 0
    for rat_key in rat_dict.keys():
        rat_result = rat_dict[rat_key]
        if rat_result[0] < rat_min:
            rat_min = rat_result[0]
        if rat_result[1] > rat_max:
            rat_max = rat_result[1]
    for wsd_key in wsd_dict.keys():
        wsd_result = wsd_dict[wsd_key]
        if wsd_result[0] < wsd_min:
            wsd_min = wsd_result[0]
        if wsd_result[1] > wsd_max:
            wsd_max = wsd_result[1]
    width = rat_max - rat_min
    if width == 0:
        width = 0.005
    height = wsd_max - wsd_min
    if height == 0:
        height = 0.005
    ax.add_patch(Rectangle((rat_min, wsd_min), width, height, fill=False))


wsd_results_cooc = {"cooc_sense": [0.9730733527, 0.978710402],
                    "cooc_word": [0.9602758899, 0.967843862]}
wsd_results_spread = {"spread_never_nospread": [0.59993609456, 0.6476250084],
                      "spread_never": [0.5357324982, 0.55535514],
                      "spread_word": [0.450781559, 0.634564881],
                      "spread_sentence": [0.4586118165, 0.6332476208]}
wsd_results_lb = 0.3794224297
wsd_results_ub = 0.98977029
wsd_results_cts = {"CTS_never_sense_partition": [0.5982593665, 0.6444288909],
                   "CTS_word_sense_partition": [0.3635861041, 0.9490967187],
                   "CTS_sent_sense_partition": [0.3870607868, 0.9415567259],
                   "CTS_never_word_partition": [0.5459793956, 0.5908585686],
                   "CTS_word_word_partition": [0.364303052, 0.834765724],
                   "CTS_sent_word_partition": [0.3787555978, 0.8330514432],
                   "CTS_never_sense_corpus": [0.5657012499, 0.6084375625],
                   "CTS_word_sense_corpus": [0.3979881083, 0.8371990546],
                   "CTS_sent_sense_corpus": [0.4117452476, 0.8354613437],
                   "CTS_never_word_corpus": [0.5406384418, 0.581230736],
                   "CTS_word_word_corpus": [0.3915910054, 0.7804183736],
                   "CTS_sent_word_corpus": [0.404177795, 0.7788979602]}
wsd_results_stc = {"STC_word": [0.002863888815, 0.002972097734],
                   "STC_sense": [0.001488362442, 0.001518730012]}
wsd_results_jointprob = {"jointprob_never_sense": [0.9039588975, 0.9042782481],
                         "jointprob_word_sense": [0.9596608046, 0.9635057475],
                         "jointprob_sent_sense": [0.9596602773, 0.963460062],
                         "jointprob_never_word": [0.9251907547, 0.9253438248],
                         "jointprob_word_word": [0.9729329017, 0.976066049],
                         "jointprob_sent_word": [0.9724270642, 0.9755365169]}

rat_results_cooc = {"cooc": [0.09154929577, 0.09154929577]}
rat_results_ub = 0.6478873239
rat_results_spread = {"spread_sffan": [0.2183098592, 0.3028169014],
                      "spread_swowen": [0.2605633803, 0.5492957746],
                      "spread_combined": [0.2887323944, 0.5845070423]}
rat_results_cts = {"CTS_swowen": [0.3098591549, 0.5563380282],
                   "CTS_sffan": [0.2112676056, 0.2816901408],
                   "CTS_combined": [0.338028169, 0.5985915493]}
rat_results_stc = {"STC_sffan": [0.02112676056, 0.02112676056],
                   "STC_swowen": [0.1126760563, 0.1126760563],
                   "STC_combined": [0.007042253521, 0.007042253521]}
rat_results_jointprob = {"jointprob_swowen": [0.09154929577, 0.09154929577],
                         "jointprob_sffan": [0.09154929577, 0.09154929577],
                         "jointprob_combined": [0.09154929577, 0.09154929577]}

plot_type = "cat_rectangle"

if plot_type == "simple":
    plt.xlabel("RAT Accuracies")
    plt.ylabel("WSD Accuracies")
    plt.axvline(x=rat_results_ub, color="blue")
    plt.axhline(y=wsd_results_ub, color="blue")
    plot_results(rat_results_cooc, wsd_results_cooc)
    plot_results(rat_results_spread, wsd_results_spread)
    plot_results(rat_results_cts, wsd_results_cts)
    plot_results(rat_results_stc, wsd_results_stc)
    plot_results(rat_results_jointprob, wsd_results_jointprob)
elif plot_type == "max":
    plt.xlabel("RAT Accuracies")
    plt.ylabel("WSD Accuracies")
    plt.axvline(x=rat_results_ub, color="blue")
    plt.axhline(y=wsd_results_ub, color="blue")
    plot_max_results(rat_results_cooc, wsd_results_cooc)
    plot_max_results(rat_results_spread, wsd_results_spread)
    plot_max_results(rat_results_cts, wsd_results_cts)
    plot_max_results(rat_results_stc, wsd_results_stc)
    plot_max_results(rat_results_jointprob, wsd_results_jointprob)
elif plot_type == "cat_rectangle":
    fig, ax = plt.subplots()
    ax.set_xlabel("RAT Accuracies")
    ax.set_ylabel("WSD Accuracies")
    ax.axvline(x=rat_results_ub, color="blue")
    ax.axhline(y=wsd_results_ub, color="blue")
    plot_category_rectangle(rat_results_cooc, wsd_results_cooc, ax)
    plot_category_rectangle(rat_results_spread, wsd_results_spread, ax)
    plot_category_rectangle(rat_results_cts, wsd_results_cts, ax)
    plot_category_rectangle(rat_results_stc, wsd_results_stc, ax)
    plot_category_rectangle(rat_results_jointprob, wsd_results_jointprob, ax)
plt.show()
