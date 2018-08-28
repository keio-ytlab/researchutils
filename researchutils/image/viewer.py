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
        Weather given image is grayscale or colored
        True if image is grayscale
    """
    show_images([image], titles=[title], is_gray=is_gray)


def show_images(images, titles=[], is_gray=False):
    """
    Display given image

    Parameters
    -------
    images : list of numpy.ndarray
        Images to display
    titles : list of string
        Titles for each image to display
    is_gray : bool
        Weather given images are grayscale or colored
        True if images are grayscale
    """
    num_images = len(images)
    num_titles = len(titles)
    if not num_images == num_titles:
        raise ValueError('number of images and titles does not match! {} != {}'.format(
            num_images, num_titles))
    for i in range(len(images)):
        plt.subplot(1, num_images, i + 1)
        _create_window(images[i], is_gray=is_gray)
        plt.title(titles[i])
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
        Weather given images are grayscale or colored
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
