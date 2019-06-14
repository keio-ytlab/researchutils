import matplotlib.pyplot as plt
import matplotlib.animation as pltanim

import numpy as np

import math


def _create_window(image, is_gray=False):
    if is_gray:
        return plt.imshow(image, cmap='gray')
    else:
        return plt.imshow(image)


def show_image(image, title='', is_gray=False):
    """
    Display given image

    Parameters
    -------
    image : numpy.ndarray
        Image to display
    title : string
        Title of the image to display
    is_gray : bool
        Whether given image is grayscale or colored
        True if image is grayscale
    """
    show_images([image], title=title, is_gray=is_gray)


def show_images(images, title='', comparisons=None, comparison_title='', is_gray=False, vertical=False):
    """
    Display given image

    Parameters
    -------
    images : list of numpy.ndarray
        Images to display
    title : string
        Title of each image to display. If length of images is greater than 1,
        index of each image will be appended to the tail of title.
    comaprisons : list of numpy.ndarray
        Comparison images to display
    comaprison_title : list of numpy.ndarray
        Title of each comparison image to display. If length of comparisons is greater than 1,
        index of each image will be appended to the tail of title.
    is_gray : bool
        Whether given images are grayscale or colored
        True if images are grayscale
    vertical : bool
        Display comparisons vertically
    """
    num_images = len(images)
    num_comparisons = 0 if comparisons is None else len(comparisons)
    rows = 1 if comparisons is None else 2
    columns = max(num_images, num_comparisons)
    if vertical:
        rows, columns = columns, rows
    for i in range(num_images):
        index = (i * columns + 1) if vertical else (i + 1)
        plt.subplot(rows, columns, index)
        _create_window(images[i], is_gray=is_gray)
        if num_images == 1:
            plt.title(title)
        else:
            plt.title('{} {}'.format(title, i))
    for i in range(num_comparisons):
        index = (i * columns + 2) if vertical else (i +
                                                    1 + max(num_images, num_comparisons))
        plt.subplot(rows, columns, index)
        _create_window(comparisons[i], is_gray=is_gray)
        if num_comparisons == 1:
            plt.title(comparison_title)
        else:
            plt.title('{} {}'.format(comparison_title, i))
    plt.show()


def animate(images, comparisons=None, titles=[], is_gray=False, fps=15, repeat=False,
            save_gif=False, save_mp4=False, auto_close=False):
    """
    Animate list of images

    Parameters
    -------
    images : list of numpy.ndarray
        Images to display
    comparisons : list of numpy.ndarray
        Comparison images or multiple list of comparison images to display (Optional)
    titles : list of string
        Titles for each animation
    is_gray : bool
        Whether given images are grayscale or colored
        True if images are grayscale
    fps : int
        Number of frames to display in 1 second.
        Abbreviation of "frame per second"
    repeat: bool
        Repeat animation when reaches end of images
    save_gif : bool
        Save animation as gif image
    save_mp4 : bool
        Save animation as mp4
    auto_close : bool
        Close animation window after finish animating
    """
    images_list = [images]
    comparison_list = []
    if comparisons:
        images = np.array(images)
        images_dim = images.ndim
        comparisons = np.array(comparisons)
        comparison_dim = comparisons.ndim
        if images_dim == comparison_dim:
            comparison_list = [comparisons]
        elif 1 < comparison_dim - images_dim:
            raise ValueError(
                "Unsupported images and comparisons dimension was given")
        else:
            comparison_list = comparisons
    images_list.extend(comparison_list)
    animate_in_matrix_form(images=images_list, titles=titles, is_gray=is_gray, fps=fps,
                           images_per_row=len(images_list), repeat=repeat, save_gif=save_gif,
                           save_mp4=save_mp4, auto_close=auto_close)


def animate_in_matrix_form(images=None, titles=[], is_gray=False, fps=15, images_per_row=None,
                           repeat=False, save_gif=False, save_mp4=False, auto_close=False,
                           remove_axis=False):
    """
    Animate list of images in matrix form

    Parameters
    -------
    images : list of numpy.ndarray
        Images to display
    titles : list of string
        Titles for each animation
    is_gray : bool
        Whether given images are grayscale or colored
        True if images are grayscale
    fps : int
        Number of frames to display in 1 second.
        Abbreviation of "frame per second"
    images_per_row: int
        Number of images to display in each row
    repeat: bool
        Repeat animation when reaches end of images
    save_gif : bool
        Save animation as gif image
    save_mp4 : bool
        Save animation as mp4
    auto_close : bool
        Close animation window after finish animating
    remove_axis : bool
        Remove x and y axis
    """
    fig = plt.figure()
    im_plt = []
    im = []
    images_num = len(images)
    images = np.array(images)
    if images_per_row is None:
        images_per_row = images_num
    images_per_row = min(images_num, images_per_row)
    row_num = int(math.ceil(images_num / images_per_row))
    col_num = images_per_row
    frame_num = len(images[0])
    for row in range(row_num):
        for column in range(col_num):
            index = row * col_num + column
            plot = plt.subplot(row_num, col_num, index + 1)
            if remove_axis:
                plot.axis('off')
            im_plt.append(plot)
            im.append(_create_window(images[index][0], is_gray=is_gray))

    def init():
        update(0)

    def update(index):
        for i in range(images_num):
            im[i].set_data(images[i][index])

        for i, title in enumerate(titles):
            if __is_list(title):
                title = title[index]
            else:
                title = '{} frame: {}'.format(title, index)
            im_plt[i].set_title(title)
        return im

    anim = pltanim.FuncAnimation(
        fig, update, init_func=init, frames=frame_num, interval=(1000//fps), repeat=repeat)
    if save_gif:
        anim.save('anim.gif', writer='imagemagick')
    if save_mp4:
        Writer = pltanim.writers['ffmpeg']
        writer = Writer(fps=fps, bitrate=1800)
        anim.save('anim.mp4', writer=writer)
    block = not auto_close
    plt.show(block=block)


def __is_list(object):
    return isinstance(object, list)
