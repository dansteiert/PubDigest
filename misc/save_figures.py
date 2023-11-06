import logging, os

import matplotlib.pyplot as plt

def save_plt_figure(path: str, logger: logging.Logger, dpi: int = 400, transparent=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=dpi, transparent=transparent)
    plt.clf()
    plt.close()
    logger.debug(f"Figure Saved to: {path}")