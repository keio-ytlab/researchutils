import matplotlib.pyplot as plt
import matplotlib.animation as pltanim


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
        index = (i * columns + 2) if vertical else (i + 1 + max(num_images, num_comparisons)) 
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
        Comparison images to display (Optional)
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
    fig = plt.figure()
    cm = None
    cm_plt = None
    im_plt = None
    if comparisons:
        im_plt = plt.subplot(1, 2, 1)
        im = _create_window(images[0], is_gray=is_gray)
        cm_plt = plt.subplot(1, 2, 2)
        cm = _create_window(comparisons[0], is_gray=is_gray)
    else:
        im = _create_window(images[0], is_gray=is_gray)

    def init():
        update(0)

    def update(index):
        if cm:
            im.set_data(images[index])
            cm.set_data(comparisons[index])
            for i, title in enumerate(titles):
                if i == 0:
                    im_plt.set_title('{} frame: {}'.format(title, index))
                else:
                    cm_plt.set_title('{} frame: {}'.format(title, index))
        else:
            im.set_data(images[index])
            if 0 < len(titles):
                plt.title('{} frame: {}'.format(titles[0], index))
        return im

    anim = pltanim.FuncAnimation(
        fig, update, init_func=init, frames=len(images), interval=(1000//fps), repeat=repeat)
    if save_gif:
        anim.save('anim.gif', writer='imagemagick')
    if save_mp4:
        Writer = pltanim.writers['ffmpeg']
        writer = Writer(fps=fps, bitrate=1800)
        anim.save('anim.mp4', writer=writer)
    block = not auto_close
    plt.show(block=block)
