import logging, os

import matplotlib.pyplot as plt

def save_plt_figure(path: str, logger: logging.Logger, dpi: int = 400, transparent=True, save_as_svg=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=dpi, transparent=transparent)
    if save_as_svg:
        plt.savefig(path.replace(".png", ".svg"), transparent=transparent)
    plt.clf()
    plt.close()
    logger.debug(f"Figure Saved to: {path}")