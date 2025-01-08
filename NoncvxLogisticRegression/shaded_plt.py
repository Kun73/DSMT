import numpy as np
import matplotlib.pyplot as plt

def shaded_plt(data, color, marker, mark_every, name):


    # Convert the list to a numpy array for easier manipulation
    data = np.array(data)

    # Calculate the mean, minimum, and maximum across each experiment run (axis=1 means along rows)
    means = np.mean(data, axis=0)
    # min_values = np.min(data, axis=0)
    # max_values = np.max(data, axis=0)
    std = np.std(data, axis=0)

    # X values representing the different runs (e.g., Run 1, Run 2, etc.)
    x = np.arange(0, data.shape[1])

    # Plot the mean line
    plt.plot(x, means, label=name, color=color, marker=marker, markevery=mark_every)

    # Fill the area between the min and max values for each experiment run
    plt.fill_between(x, means - std, means + std, color=color, alpha=0.2)

    # # Add titles and labels
    # plt.title("Mean with Shaded Area Representing Range of Experiment Runs")
    # plt.xlabel("Experiment Run")
    # plt.ylabel("Measurement Value")
    #
    # # Display legend
    # plt.legend()
    #
    # # Show the plot
    # plt.show()

if __name__ == '__main__':
    # Sample data: each sublist represents repeated measurements for a run
    data = [
        [1.1, 1.3, 1.2, 1.4, 1.2],  # Run 1
        [2.0, 2.1, 1.9, 2.2, 2.0],  # Run 2
        [3.1, 3.0, 3.2, 3.1, 3.3],  # Run 3
        [4.5, 4.6, 4.4, 4.5, 4.7],  # Run 4
        [5.0, 5.1, 4.9, 5.2, 5.0],  # Run 5
        [5.0, 5.1, 4.9, 5.2, 7.0],  # Run 6
    ]
    data2 = [
        [4.1, 1.3, 1.2, 1.4, 1.2],  # Run 1
        [5.0, 2.1, 1.9, 2.2, 2.0],  # Run 2
        [3.1, 5.0, 3.2, 3.1, 3.3],  # Run 3
        [4.5, 4.6, 2.4, 4.5, 4.7],  # Run 4
        [5.0, 5.1, 4.9, 5.2, 5.0],  # Run 5
        [5.0, 5.1, 4.9, 6.2, 1.0],  # Run 6
    ]
    plt.figure()
    shaded_plt(data, 'b', '<', 1, 'CSGDM')
    shaded_plt(data2, 'r', '<', 1, 'CSGD')
    plt.grid(True)
    plt.yscale('log')
    plt.tick_params(labelsize='large', width=3)
    plt.xlabel('# Iterations')
    plt.ylabel(r'$\frac{1}{n}\sum_{i = 1}^n\mathbb{E}||x_{i,k} - x^*||^2$', fontsize=12)
    plt.legend()
    plt.show()
