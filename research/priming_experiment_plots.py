from priming_experiment import *
import matplotlib.pyplot as plt
import matplotlib.axes as plt_axes
import matplotlib.patches as mpatches
import statistics
import seaborn as sns


def get_plot_data_distance(constant_offset, decay_parameter, activation_base, num_trials, cooccur_1_freq,
                           cooccur_2_freq, target_freq, num_word_pairs, semantic, cooccurrence, cooccur_num=1,
                           auto_storage=True):
    """
    Gets the plot data (target distances tested for each trial, target word activations for each trial, and means for
        each set of trials, and standard deviations for each set of trials) using input parameters for the experiment.
        Enables the same data to be reused for more than one visualization.
        Parameters:
            constant_offset (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            activation_base (float): A parameter in the activation equation.
            num_trials (int): How many trials the experiment should go through.
            cooccur_1_freq (float): The frequency from 0 to 1 of cooccur_1.
            cooccur_2_freq (float): The frequency from 0 to 1 of cooccur_2.
            target_freq (float): The frequency from 0 to 1 of the target word.
            num_word_pairs (int): The number of word pairs in the list.
            semantic (Boolean): If true, uses "prime" as the prime word. If false and cooccurrence == false, uses
                "control" as the prime word.
            cooccurrence (Boolean): If true, uses a cooccurrence element as the prime word. If false and
                semantic == false, uses "control" as the prime word.
            cooccur_num (int): Determines which cooccurrence element to use as the prime word when cooccurrence == true
                and semantic == false. If it equals 1, "cooccur_1" is used. If 2, "cooccur_2" is used.
            auto_storage (Boolean): Determines which activation_dynamics class is used. If true, BaseLevelActivation is
                used. If false, PairwiseCooccurrenceActivation.
    Returns:
        list: Target distances used for each trial.
        list: Activations of each trial.
        list: Means for all trials corresponding to each target distance.
        list: Standard deviations corresponding to each mean (all trials corresponding to each target distance).
    """
    distances = []
    activations = []
    means = []
    stdev = []
    for target_distance in range(1, 11):
        exp_trial_activations = run_experiment(constant_offset, decay_parameter, activation_base, target_distance,
                                               num_trials, cooccur_1_freq, cooccur_2_freq, target_freq, num_word_pairs,
                                               semantic, cooccurrence, cooccur_num, auto_storage)
        means.append(statistics.mean(exp_trial_activations))
        stdev.append(statistics.pstdev(exp_trial_activations))
        for trial_num in range(len(exp_trial_activations)):
            distances.append(target_distance)
            activations.append(exp_trial_activations[trial_num])
    return distances, activations, means, stdev


def get_plot_data_cooccurrence_freq(constant_offset, decay_parameter, activation_base, num_trials, target_distance,
                                    target_freq, num_word_pairs, semantic, cooccurrence, cooccur_num=1,
                                    auto_storage=True):
    """
    Gets the plot data (target distances tested for each trial, target word activations for each trial, and means for
        each set of trials, and standard deviations for each set of trials) using input parameters for the experiment.
        Enables the same data to be reused for more than one visualization.
        Parameters:
            constant_offset (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            activation_base (float): A parameter in the activation equation.
            target_distance (int): The number of edges between the target word and the prime word.
            num_trials (int): How many trials the experiment should go through.
            target_freq (float): The frequency from 0 to 1 of the target word.
            num_word_pairs (int): The number of word pairs in the list.
            semantic (Boolean): If true, uses "prime" as the prime word. If false and cooccurrence == false, uses
                "control" as the prime word.
            cooccurrence (Boolean): If true, uses a cooccurrence element as the prime word. If false and
                semantic == false, uses "control" as the prime word.
            cooccur_num (int): Determines which cooccurrence element to use as the prime word when cooccurrence == true
                and semantic == false. If it equals 1, "cooccur_1" is used. If 2, "cooccur_2" is used.
            auto_storage (Boolean): Determines which activation_dynamics class is used. If true, BaseLevelActivation is
                used. If false, PairwiseCooccurrenceActivation.
        Returns:
            list: Cooccurrence frequency of cooccur_1 used in each trial.
            list: Activations from each trial.
            list: Mean values of activations for each cooccurrence frequency.
            list: Standard deviations corresponding to each mean value above.

        """
    cooccurrence_freqs = []
    activations = []
    means = []
    stdev = []
    for cooccur_2_freq in [x * 0.1 for x in list(range(0, 10, 1))]:
        exp_trial_activations = run_experiment(constant_offset, decay_parameter, activation_base, target_distance,
                                               num_trials, 1 - cooccur_2_freq, cooccur_2_freq, target_freq,
                                               num_word_pairs, semantic, cooccurrence, cooccur_num, auto_storage)
        means.append(statistics.mean(exp_trial_activations))
        stdev.append(statistics.pstdev(exp_trial_activations))
        for trial_num in range(len(exp_trial_activations)):
            cooccurrence_freqs.append(cooccur_2_freq)
            activations.append(exp_trial_activations[trial_num])
    return cooccurrence_freqs, activations, means, stdev


def plot_2_sets_jitter_error_target_act_vs_distance(exp_graph_distances, exp_activations, exp_means, exp_stdev,
                                                    control_graph_distances, control_activations, control_means,
                                                    control_stdev):
    """
     Creates a jitter plot with error bars for experimental and control conditions of the experiment. Only use with
     one setting, i.e. priming, cooccurrence, priming & cooccurrence, for best visualization.
     Parameters:
         All parameters correspond to those output (in order) from the get_plot_data_distance function. This function
         enables you to compare conditions for two different data sets output from that function.
    Returns: None
    """

    exp_plot = sns.stripplot(x=exp_graph_distances, y=exp_activations, jitter=True, color='b', native_scale=True,
                             alpha=0.3)
    sns.stripplot(x=[x + 0.2 for x in control_graph_distances], y=control_activations, jitter=True, color='r',
                  native_scale=True, alpha=0.3)
    control_patch = mpatches.Patch(color='r', label="Control")
    exp_patch = mpatches.Patch(color='b', label='Experimental')
    plt.legend(handles=[exp_patch, control_patch])
    plt_axes.Axes.set(exp_plot, xlabel="Target/Shared Word Distance", ylabel="Target Activation")
    plt.errorbar(x=list(range(1, 11)), y=exp_means, yerr=exp_stdev, color="navy", capsize=5)
    plt.errorbar(x=[x + 0.2 for x in list(range(1, 11))], y=control_means, yerr=control_stdev, color="darkred",
                 capsize=5)
    plt.show()


def plot_3_sets_jitter_error_target_act_vs_distance(prime_graph_distances, prime_activations, prime_means, prime_stdev,
                                                    coocc_graph_distances, coocc_activations, coocc_means, coocc_stdev,
                                                    control_graph_distances, control_activations, control_means,
                                                    control_stdev,
                                                    ):
    """
    Creates a jitter plot with error bars for experimental and control conditions of the experiment. Only use with
        one setting, i.e. priming, cooccurrence, priming & cooccurrence, for best visualization.
        Parameters:
            All parameters correspond to those output (in order) from the get_plot_data_distance function. This function
            enables you to compare conditions for three different data sets output from that function.
         Returns: None
    """
    exp_plot = sns.stripplot(x=prime_graph_distances, y=prime_activations, jitter=True, color='b', native_scale=True,
                             alpha=0.3)
    sns.stripplot(x=[x + 0.2 for x in control_graph_distances], y=control_activations, jitter=True, color='r',
                  native_scale=True, alpha=0.3)
    sns.stripplot(x=[x + 0.4 for x in coocc_graph_distances], y=coocc_activations, jitter=True, color='g',
                  native_scale=True, alpha=0.3)
    control_patch = mpatches.Patch(color='r', label='Control')
    prime_patch = mpatches.Patch(color='b', label='Semantic')
    coocc_patch = mpatches.Patch(color='g', label='Cooccurrence')
    plt.legend(handles=[prime_patch, control_patch, coocc_patch])
    plt_axes.Axes.set(exp_plot, xlabel="Target/Shared Word Distance", ylabel="Target Activation")
    plt.errorbar(x=list(range(1, 11)), y=prime_means, yerr=prime_stdev, color="navy", capsize=5)
    plt.errorbar(x=[x + 0.4 for x in list(range(1, 11))], y=coocc_means, yerr=coocc_stdev, color='darkgreen',
                 capsize=5)
    plt.errorbar(x=[x + 0.2 for x in list(range(1, 11))], y=control_means, yerr=control_stdev, color="darkred",
                 capsize=5)
    plt.show()


def plot_3_sets_jitter_error_target_act_vs_coocc_freq(prime_coocc_freqs, prime_activations, prime_means, prime_stdev,
                                                      coocc_coocc_freqs, coocc_activations, coocc_means, coocc_stdev,
                                                      control_coocc_freqs, control_activations, control_means,
                                                      control_stdev,
                                                      ):
    """
    Creates a jitter plot with error bars for priming, (one) cooccurrence, and control conditions of the experiment.
        Parameters:
            All parameters correspond to those output (in order) from the get_plot_data_cooccurrence function. This
            function enables you to compare conditions for three different data sets output from that function.
        Returns: None
    """
    exp_plot = sns.stripplot(x=prime_coocc_freqs, y=prime_activations, jitter=True, color='b', native_scale=True,
                             alpha=0.3)
    sns.stripplot(x=[x + 0.04 for x in control_coocc_freqs], y=control_activations, jitter=True, color='r',
                  native_scale=True, alpha=0.3)
    sns.stripplot(x=[x + 0.02 for x in coocc_coocc_freqs], y=coocc_activations, jitter=True, color='g',
                  native_scale=True, alpha=0.3)
    control_patch = mpatches.Patch(color='r', label='Control')
    prime_patch = mpatches.Patch(color='b', label='Semantic')
    coocc_patch = mpatches.Patch(color='g', label='Cooccurrence')
    plt.legend(handles=[prime_patch, control_patch, coocc_patch])
    plt_axes.Axes.set(exp_plot, xlabel="Cooccurrence Frequency", ylabel="Target Activation")
    plt.errorbar(x=[0.1 * x for x in list(range(0, 10, 1))], y=prime_means, yerr=prime_stdev, color="navy", capsize=5)
    plt.errorbar(x=[0.1 * x + 0.02 for x in list(range(0, 10, 1))], y=coocc_means, yerr=coocc_stdev, color='darkgreen',
                 capsize=5)
    plt.errorbar(x=[0.1 * x + 0.04 for x in list(range(0, 10, 1))], y=control_means, yerr=control_stdev,
                 color="darkred", capsize=5)
    plt.show()


def plot_4_sets_jitter_error_target_act_vs_coocc_freq(prime_coocc_freqs, prime_activations, prime_means, prime_stdev,
                                                      control_coocc_freqs, control_activations, control_means,
                                                      control_stdev, coocc_1_freqs, coocc_1_activations, coocc_1_means,
                                                      coocc_1_stdev, coocc_2_freqs, coocc_2_activations, coocc_2_means,
                                                      coocc_2_stdev):
    """
    Creates a jitter plot with error bars for experimental and control conditions of the experiment. Only use with
        one setting, i.e. priming, cooccurrence, priming & cooccurrence, for best visualization.
        Parameters:
            All parameters correspond to those output (in order) from the get_plot_data_cooccurrence function. This
            function enables you to compare conditions for four different data sets output from that function.
        Returns: None
    """
    exp_plot = sns.stripplot(x=prime_coocc_freqs, y=prime_activations, jitter=True, color='b', native_scale=True,
                             alpha=0.1)
    sns.stripplot(x=[x + 0.02 for x in control_coocc_freqs], y=control_activations, jitter=True, color='r',
                  native_scale=True, alpha=0.1)
    sns.stripplot(x=[x + 0.04 for x in coocc_1_freqs], y=coocc_1_activations, jitter=True, color='g',
                  native_scale=True, alpha=0.1)
    sns.stripplot(x=[x + 0.06 for x in coocc_2_freqs], y=coocc_2_activations, jitter=True, color='orange',
                  native_scale=True, alpha=0.1)
    control_patch = mpatches.Patch(color='r', label='Control')
    prime_patch = mpatches.Patch(color='b', label='Semantic')
    coocc_1_patch = mpatches.Patch(color='g', label='Cooccurrence 1')
    coocc_2_patch = mpatches.Patch(color='orange', label='Cooccurrence 2')
    plt.legend(handles=[prime_patch, control_patch, coocc_1_patch, coocc_2_patch])
    plt_axes.Axes.set(exp_plot, xlabel="Cooccurrence Frequency", ylabel="Target Activation")
    plt.errorbar(x=[0.1 * x for x in list(range(0, 10, 1))], y=prime_means, yerr=prime_stdev, color="navy", capsize=5)
    plt.errorbar(x=[0.1 * x + 0.02 for x in list(range(0, 10, 1))], y=control_means, yerr=control_stdev,
                 color="darkred",
                 capsize=5)
    plt.errorbar(x=[0.1 * x + 0.04 for x in list(range(0, 10, 1))], y=coocc_1_means, yerr=coocc_1_stdev,
                 color='darkgreen',
                 capsize=5)
    plt.errorbar(x=[0.1 * x + 0.06 for x in list(range(0, 10, 1))], y=coocc_2_means, yerr=coocc_2_stdev, color="brown",
                 capsize=5)
    plt.show()

# Example
# data_control = get_plot_data_cooccurrence_freq(0, 0.05, 2, 1000, 2, 0.5, 100, False, False, auto_storage=False)
# data_exp_semantic = get_plot_data_cooccurrence_freq(0, 0.05, 2, 1000, 2, 0.5, 100, True, False, auto_storage=False)
# data_exp_coocc_1 = get_plot_data_cooccurrence_freq(0, 0.05, 2, 1000, 2, 0.5, 100, False, True, auto_storage=False)
# data_exp_coocc_2 = get_plot_data_cooccurrence_freq(0, 0.05, 2, 1000, 2, 0.5, 100, False, True, 2, auto_storage=False)

# plot_4_sets_jitter_error_target_act_vs_coocc_freq(data_exp_semantic[0], data_exp_semantic[1], data_exp_semantic[2],
# data_exp_semantic[3], data_control[0], data_control[1], data_control[2], data_control[3], data_exp_coocc_1[0],
# data_exp_coocc_1[1], data_exp_coocc_1[2], data_exp_coocc_1[3], data_exp_coocc_2[0], data_exp_coocc_2[1],
# data_exp_coocc_2[2], data_exp_coocc_2[3])
