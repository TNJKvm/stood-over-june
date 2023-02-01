import matplotlib.pyplot as plt
import numpy as np


def acc_step(accuracies: list, test: bool = False, xticks = [0, 500, 1000, 1500]):
    plt.title('Accuracy vs. Steps', fontsize=16)
    plt.xlabel('Steps', fontsize=16)
    plt.xticks(xticks)
    if test:
        plt.ylabel('Test Accuracy (%)', fontsize=16)
    else:
        plt.ylabel('Train Accuracy (%)', fontsize=16)
    plt.plot(accuracies)
    plt.show()
