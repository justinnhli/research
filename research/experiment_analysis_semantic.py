from priming_experiment import *
import matplotlib.pyplot as plt
import matplotlib.axes as plt_axes
import matplotlib.patches as mpatches
import statistics
import seaborn as sns


def get_plot_data(num_trials, occ_control, occ_target, num_word_pairs, constant_offset, decay_parameter,
                  activation_base, spread_act_setting, coocc_setting, control):
    """
    Gets the plot data (target distances tested for each trial, target word activations for each trial, and means for
    each set of trials, and standard deviations for each set of trials) using input parameters for the experiment.

    Enables the same data to be reused for more than one visualization.
    """
    exp_distances = []
    exp_activations = []
    exp_means = []
    exp_stdev = []
    for target_distance in range(1, 11):
        exp_trial_activations = run_experiment(target_distance, num_trials, occ_control, occ_target, num_word_pairs,
                                               control, spread_act_setting, coocc_setting, constant_offset,
                                               decay_parameter, activation_base)
        exp_means.append(statistics.mean(exp_trial_activations))
        exp_stdev.append(statistics.pstdev(exp_trial_activations))

        for trial_num in range(len(exp_trial_activations)):
            exp_distances.append(target_distance)
            exp_activations.append(exp_trial_activations[trial_num])

    return exp_distances, exp_activations, exp_means, exp_stdev





def plot_error_target_act_vs_distance(occ_target, occ_control, num_trials, num_word_pairs, constant_offset,
                                      decay_parameter, activation_base):
    """
      Plots the target activation versus distance for control and experimental conditions of the experiment with only
     error bars, and no jitter.
    """
    exp_means = []
    exp_stdev = []

    control_means = []
    control_stdev = []

    exp_target_root_distances = list(range(1, 11))
    control_target_root_distances = []

    for target_distance in exp_target_root_distances:
        control_target_root_distances.append(target_distance + 0.1)


        exp_target_activations = run_experiment(target_distance, num_trials, occ_control, occ_target, num_word_pairs,
                                                False, constant_offset, decay_parameter, activation_base)
        exp_means.append(statistics.mean(exp_target_activations))
        exp_stdev.append(statistics.pstdev(exp_target_activations))

        control_target_activations = run_experiment(target_distance, num_trials, occ_control, occ_target,
                                                    num_word_pairs, True, constant_offset, decay_parameter,
                                                    activation_base)
        control_means.append(statistics.mean(control_target_activations))
        control_stdev.append(statistics.pstdev(exp_target_activations))

    plt.plot(exp_target_root_distances, exp_means, label='Experimental', color='b')
    plt.plot(control_target_root_distances, control_means, label='Control', color='r')
    plt.errorbar(x=exp_target_root_distances, y=exp_means, yerr=control_stdev, color='b', capsize=5)
    plt.errorbar(x=control_target_root_distances, y=control_means, yerr=control_stdev, color='r', capsize=5)
    plt.xlabel('Target/ Root Graph Distance')
    plt.ylabel('Target Word Activation')
    plt.legend()

    plt.show()





def plot_jitter_target_act_vs_distance(exp_graph_distances, exp_activations, control_graph_distances,
                                       control_activations):
    """
     Plots the target activation versus distance for control and experimental conditions of the experiment with only
     jitter, no lines indicating standard error.
    """
    exp_plot = sns.stripplot(x=exp_graph_distances, y=exp_activations, jitter=True, color='b', native_scale=True)
    sns.stripplot(x=control_graph_distances, y=control_activations, jitter=True, color='r', native_scale=True)
    control_patch = mpatches.Patch(color='r', label="Control")
    exp_patch = mpatches.Patch(color='b', label='Experimental')
    plt.legend(handles=[exp_patch, control_patch])
    plt_axes.Axes.set(exp_plot, xlabel="Target/Root Graph Distance", ylabel="Target Activation")
    plt.show()





def plot_2_sets_jitter_error_target_act_vs_distance(exp_graph_distances, exp_activations, exp_means, exp_stdev,
                                             control_graph_distances, control_activations, control_means, control_stdev):
    """
     Creates a jitter plot with error bars for experimental and control conditions of the experiment. Only use with
     one setting, i.e. spreading, cooccurrence, spreading & cooccurrence, for best visualization.
    """

    exp_plot = sns.stripplot(x=exp_graph_distances, y=exp_activations, jitter=True, color='b', native_scale=True,
                             alpha=0.25)
    sns.stripplot(x=[x + 0.2 for x in control_graph_distances], y=control_activations, jitter=True, color='r',
                  native_scale=True, alpha=0.25)
    control_patch = mpatches.Patch(color='r', label="Control")
    exp_patch = mpatches.Patch(color='b', label='Experimental')
    plt.legend(handles=[exp_patch, control_patch])
    plt_axes.Axes.set(exp_plot, xlabel="Target/Root Graph Distance", ylabel="Target Activation")
    plt.errorbar(x=list(range(1, 11)), y=exp_means, yerr=exp_stdev, color="navy", capsize=5)
    plt.errorbar(x=[x + 0.2 for x in list(range(1, 11))], y=control_means, yerr=control_stdev, color="darkred", capsize=5)

    plt.show()




def plot_3_sets_jitter_error_target_act_vs_distance(spread_graph_distances, spread_activations, spread_means, spread_stdev,
                                             coocc_graph_distances, coocc_activations, coocc_means, coocc_stdev,
                                             control_graph_distances, control_activations, control_means, control_stdev,
                                             ):
    """
    Creates a jitter plot with error bars for experimental and control conditions of the experiment. Only use with
     one setting, i.e. spreading, cooccurrence, spreading & cooccurrence, for best visualization.
    """
    exp_plot = sns.stripplot(x=spread_graph_distances, y=spread_activations, jitter=True, color='b', native_scale=True,
                             alpha=0.1)
    sns.stripplot(x=[x + 0.2 for x in control_graph_distances], y=control_activations, jitter=True, color='r',
                  native_scale=True, alpha=0.1)
    sns.stripplot(x=[x + 0.4 for x in coocc_graph_distances], y=coocc_activations, jitter=True, color='g',
                  native_scale=True, alpha=0.1)
    control_patch = mpatches.Patch(color='r', label='Control')
    spread_patch = mpatches.Patch(color='b', label='Spreading')
    coocc_patch = mpatches.Patch(color='g', label='Cooccurrence')
    plt.legend(handles=[spread_patch, control_patch, coocc_patch])
    plt_axes.Axes.set(exp_plot, xlabel="Target/Root Graph Distance", ylabel="Target Activation")
    plt.errorbar(x=list(range(1, 11)), y=spread_means, yerr=spread_stdev, color="navy", capsize=5)
    plt.errorbar(x=[x + 0.4 for x in list(range(1, 11))], y=coocc_means, yerr=coocc_stdev, color='darkgreen',
                 capsize=5)
    plt.errorbar(x=[x + 0.2 for x in list(range(1, 11))], y=control_means, yerr=control_stdev, color="darkred",
                 capsize=5)

    plt.show()


# Testing...
data_control_spread = get_plot_data(100, 0.3, 0.3, 100, 0, 0.05, 2, True, False, True)
data_exp_spread = get_plot_data(100, 0.3, 0.3, 100, 0, 0.05, 2, True, False, False)
data_exp_coocc = get_plot_data(100, 0.3, 0.3, 100, 0, 0.05, 2, False, True, False)
plot_3_sets_jitter_error_target_act_vs_distance(data_exp_spread[0], data_exp_spread[1], data_exp_spread[2],
                                                data_exp_spread[3], data_exp_coocc[0], data_exp_coocc[1],
                                                data_exp_coocc[2], data_exp_coocc[3], data_control_spread[0],
                                                data_control_spread[1], data_control_spread[2], data_control_spread[3])

#plot_2_sets_jitter_error_target_act_vs_distance(data_exp_spread[0], data_exp_spread[1], data_exp_spread[2],
                                               # data_exp_spread[3], data_control_spread[0], data_control_spread[1],
                                               # data_control_spread[2], data_control_spread[3])


