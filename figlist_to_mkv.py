import gc
import os
import imageio
import numpy as np
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt


def figure_to_image(figure):
    canvas = FigureCanvas(figure)
    canvas.draw()  # draw the canvas, cache the renderer
    width, height = figure.get_size_inches() * figure.get_dpi()
    image = np.array(canvas.buffer_rgba(), dtype='uint8').reshape(int(height), int(width), 4)
    plt.close(figure)
    return image


def create_mkv(figlist, path, name, fps=1):
    images = []
    pbar = tqdm(figlist, position=0, leave=True)
    for fig in pbar:
        pbar.set_description("Creating GIF")
        images.append(figure_to_image(fig))
        plt.close(fig)
        del fig

    gc.collect()

    imageio.mimsave(os.path.join(path, f"{name}.mkv"), images, fps=fps)
