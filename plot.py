import os
import numpy as np
import collections
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags

import seaborn as sns
import matplotlib.pyplot as plt

BASE_PATH = '/Users/ashishrao/Coding/research/vanroylab/epistemic-per-project/dopamine-plotting/runs'  # @param
METHODS = ['uniform_ensemble_dqn', 'loss_prioritized_ensemble_dqn', 'td_prioritized_ensemble_dqn', 'variance_reduction_prioritized_ensemble_dqn']

def load_data():
    experiment_data = {}
    for method in METHODS:
        log_dir = f"{BASE_PATH}/{method}"
        method_data = \
            colab_utils.read_experiment(log_dir, verbose=True, summary_keys=['train_episode_returns'])
        method_data['agent'] = method
        method_data['run_number'] = 1
        experiment_data[method] = method_data
    return experiment_data

def plot(experiment_data):
    fig, ax = plt.subplots(figsize=(16,8))
    for idx, method in enumerate(experiment_data.keys()):
        method_data = experiment_data[method]
        plt.plot(method_data['iteration'], method_data['train_episode_returns'], label=method)


        # plt.plot()
        # sns.lineplot(
        #     x='iteration', y='train_episode_returns', hue='agent',
        #     data=experiment_data[method], ax=ax)
    plt.title("CartPole")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot(load_data())



