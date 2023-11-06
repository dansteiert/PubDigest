
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def set_figure_estetics(y_axis_elements: int = None, x_axis_elements: int = None, figure_ratio: float = 4 / 3,
                        figure_scale: float = 1, fontsize: int = 10, titlesize: str = "xx-large"):
    mpl.rcParams.update(mpl.rcParamsDefault)
    if not y_axis_elements is None:
        plt.rcParams["figure.figsize"] = np.multiply(
            [(y_axis_elements / 7.3) / figure_ratio, (y_axis_elements / 7.3)],
            figure_scale)
        plt.rcParams["font.size"] = fontsize
        plt.rcParams["axes.titlesize"] = titlesize
    elif y_axis_elements is None and x_axis_elements is None:
        plt.rcParams["figure.figsize"] = np.multiply(
            [4.8 / figure_ratio, 4.8], figure_scale)
        plt.rcParams["font.size"] = fontsize
        plt.rcParams["axes.titlesize"] = titlesize
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["font.serif"] = ["Times New Roman"]

        # plt.rcParams["figure.figsize"] = [(df_filtered.shape[0] / 10) * 2,
        #                                   (df_filtered.shape[0] / 10) * 2 * figure_ratio]
        # plt.rcParams["font.size"] = plt.rcParams["figure.figsize"][0] / 6.4 * 10
        # plt.rcParams["ytick.labelsize"] = 10
        # if plt.rcParams["font.size"] < 10:
        #     mpl.rcParams.update(mpl.rcParamsDefault)
        #     plt.rcParams["figure.figsize"] = [plt.rcParams["figure.figsize"][0],
        #                                       plt.rcParams["figure.figsize"][1] * figure_ratio]
        # plt.rcParams["axes.titlesize"] = "xx-large"