# Author: Paul Reiners

import matplotlib.pyplot as plt
import sys

def create_loss_plot(file, metric1, metric1_label, metric2, metric2_label):
    file = open(file, 'r')
    lines = file.readlines()

    training_lines = [line for line in lines if metric1 in line]
    validation_lines = [line for line in lines if metric2 in line]

    def get_metric_from_line(line):
        n = line.rfind(" ")
        metric = float(line[n:])

        return metric

    training_losses = [get_metric_from_line(line) for line in training_lines]
    validation_losses = [get_metric_from_line(line) for line in validation_lines]

    epochs = range(100)
    plt.plot(epochs, training_losses, label=metric1_label)
    plt.plot(epochs, validation_losses, label=metric2_label)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    args = sys.argv[1:]
    create_loss_plot(args[0], args[1], args[2], args[3], args[4])