import matplotlib.pyplot as plt
import seaborn as sns


class PlotData:

    def __init__(self):
        print(" I am inside init function")

    @staticmethod
    def simple_histogram_plot(get_data, n_bins, y_label, x_label):
        """
        Plots a simple normalized histogram from the given data.

        The histogram shows the distribution of values using the specified number of bins,
        and displays probability density (not raw counts).

        Args:
                get_data (array-like): The data to be plotted.
                n_bins (int): Number of bins to divide the data into.
                y_label (str): Label for the y-axis.
                x_label (str): Label for the x-axis.
        Returns:
                None
        """
        plt.figure(figsize=(6, 6))

        plt.hist(get_data, density=True, bins=n_bins)  # density=False would make counts
        plt.ylabel(y_label)
        plt.xlabel(x_label)

    @staticmethod
    def simple_gaussian_histogram_plot(get_data, n_bins, y_label, x_label):
        """
        Displays a histogram with a Gaussian-like kernel density estimate (KDE) overlay.

        Args:
                get_data (array-like): The data to be plotted.
                n_bins (int): Number of bins to divide the data into for the histogram.
                y_label (str): Label for the y-axis.
                x_label (str): Label for the x-axis.
        Returns:
                None
        """
        plot_handle = sns.displot(get_data, bins=n_bins, kde=True, height=6, aspect=1)
        plot_handle.set(xlabel=x_label, ylabel=y_label)

    @staticmethod
    def simple_plot_four_curves(var_x, var_y, var_z, var_k, var_p, title_string, leg_1, leg_2, leg_3, leg_4,
                                full_plot_save_path, y_label, x_label, custom_y_lim=(0, 1), plot_loc="upper left"):
        """
        Plots four curves on the same figure and saves the plot as an image.

        This function draws four Y-series against a common X-axis, adds labels, title, legend,
        and grid, and saves the figure to the specified path.

        Args:
                var_x (array-like): The X-axis values shared by all curves.
                var_y (array-like): Y-values for the first curve.
                var_z (array-like): Y-values for the second curve.
                var_k (array-like): Y-values for the third curve.
                var_p (array-like): Y-values for the fourth curve.
                title_string (str): Title of the plot.
                leg_1 (str): Legend label for the first curve.
                leg_2 (str): Legend label for the second curve.
                leg_3 (str): Legend label for the third curve.
                leg_4 (str): Legend label for the fourth curve.
                full_plot_save_path (str): Path to save the resulting plot image.
                y_label (str): Label for the Y-axis.
                x_label (str): Label for the X-axis.
                custom_y_lim (tuple of float): Y-axis limits (min, max).
                plot_loc (str): Location of the legend on the plot (e.g., "upper left").
        Returns:
                None
        """
        
        plt.figure(figsize=(6, 6))
        plt.plot(var_x, var_y, '-b', label=leg_1)
        plt.plot(var_x, var_z, '-r', label=leg_2)
        plt.plot(var_x, var_k, '-m', label=leg_3)
        plt.plot(var_x, var_p, '-k', label=leg_4)

        plt.legend(loc=plot_loc)

        plt.title(title_string, fontsize=10)
        plt.xlabel(x_label, fontsize=15)
        plt.ylabel(y_label, fontsize=15)

        plt.xticks(var_x)
        plt.xticks(rotation=45)
        plt.grid(color='green', linestyle='--', linewidth=0.5)

        plt.ylim([custom_y_lim[0], custom_y_lim[1]])
        plt.savefig(full_plot_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        # plt.show()

    @staticmethod
    def simple_plot_three_curves(var_x, var_y, var_z, var_k, title_string, leg_1, leg_2, leg_3, full_plot_save_path,
                                 y_label, x_label, custom_y_lim=(0, 1), plot_loc="upper left"):
        """
        Plots three curves on the same figure and saves the plot as an image.

        This function draws three Y-series against a common X-axis, adds labels, title, legend,
        and grid, and saves the figure to the specified file path.

        Args:
                var_x (array-like): The X-axis values shared by all curves.
                var_y (array-like): Y-values for the first curve.
                var_z (array-like): Y-values for the second curve.
                var_k (array-like): Y-values for the third curve.
                title_string (str): Title of the plot.
                leg_1 (str): Legend label for the first curve.
                leg_2 (str): Legend label for the second curve.
                leg_3 (str): Legend label for the third curve.
                full_plot_save_path (str): Path to save the resulting plot image.
                y_label (str): Label for the Y-axis.
                x_label (str): Label for the X-axis.
                custom_y_lim (tuple of float): Y-axis limits (min, max).
                plot_loc (str): Location of the legend (e.g., "upper left").
        Returns:
                None
        """
        plt.figure(figsize=(6, 6))
        plt.plot(var_x, var_y, '-b', label=leg_1)
        plt.plot(var_x, var_z, '-r', label=leg_2)
        plt.plot(var_x, var_k, '-m', label=leg_3)

        plt.legend(loc=plot_loc)

        plt.title(title_string, fontsize=10)
        plt.xlabel(x_label, fontsize=15)
        plt.ylabel(y_label, fontsize=15)

        plt.xticks(var_x)
        plt.xticks(rotation=45)
        plt.grid(color='green', linestyle='--', linewidth=0.5)

        plt.ylim([custom_y_lim[0], custom_y_lim[1]])
        plt.savefig(full_plot_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        # plt.show()

    @staticmethod
    def simple_plot_one_curves(var_x, var_y, title_string, leg_1,
                               full_plot_save_path, y_label, x_label, custom_y_lim=(0, 1), plot_loc="upper left"):
        """
        Plots a single curve and saves the plot as an image.

        This function draws a Y-series against an X-axis, adds a title, labels,
        legend, and grid, and saves the figure to the specified file path.

        Args:
                var_x (array-like): The X-axis values.
                var_y (array-like): The Y-axis values to plot.
                title_string (str): Title of the plot.
                leg_1 (str): Legend label for the curve.
                full_plot_save_path (str): Path to save the resulting plot image.
                y_label (str): Label for the Y-axis.
                x_label (str): Label for the X-axis.
                custom_y_lim (tuple of float): Y-axis limits (min, max).
                plot_loc (str): Location of the legend (e.g., "upper left").
        Returns:
                None
        """
        plt.figure(figsize=(8, 8))
        plt.plot(var_x, var_y, '-b', label=leg_1)

        plt.legend(loc=plot_loc)

        plt.title(title_string, fontsize=10)
        plt.xlabel(x_label, fontsize=15)
        plt.ylabel(y_label, fontsize=15)

        plt.xticks(var_x)
        plt.xticks(rotation=45)
        plt.grid(color='green', linestyle='', linewidth=2)

        plt.ylim([custom_y_lim[0], custom_y_lim[1]])
        plt.savefig(full_plot_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        # plt.show()
