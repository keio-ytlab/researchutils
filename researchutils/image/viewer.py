import matplotlib.pyplot as plt
import matplotlib.animation as pltanim


def create_window(image, is_gray=False):
    if is_gray:
        return plt.imshow(image, cmap='gray')
    else:
        return plt.imshow(image)


def show_image(image, title='', is_gray=False):
    create_window(image, is_gray=is_gray)
    plt.title(title)
    plt.show()


def show_images(images, titles=[], is_gray=False):
    num_images = len(images)
    num_titles = len(titles)
    if not num_images == num_titles:
        raise ValueError('number of images and titles does not match! {} != {}'.format(
            num_images, num_titles))
    for i in range(len(images)):
        plt.subplot(1, num_images, i + 1)
        create_window(images[i], is_gray=is_gray)
        plt.title(titles[i])
    plt.show()


def animate(images, comparisons=None, is_gray=False, repeat=False, save_gif=False, auto_close=False):
    fig = plt.figure()
    cm = None
    cm_plt = None
    im_plt = None
    if comparisons:
        im_plt = plt.subplot(1, 2, 1)
        im = create_window(images[0], is_gray=is_gray)
        cm_plt = plt.subplot(1, 2, 2)
        cm = create_window(comparisons[0], is_gray=is_gray)
    else:
        im = create_window(images[0], is_gray=is_gray)

    def init():
        if cm:
            im.set_data(images[0])
            cm.set_data(comparisons[0])
            im_plt.set_title('image frame: {}'.format(0))
            cm_plt.set_title('comparison frame: {}'.format(0))
        else:
            im.set_data(images[0])
            plt.title('frame: {}'.format(0))

    def update(index):
        if cm:
            im.set_data(images[index])
            cm.set_data(comparisons[index])
            im_plt.set_title('image frame: {}'.format(index))
            cm_plt.set_title('comparison frame: {}'.format(index))
        else:
            im.set_data(images[index])
            plt.title('frame: {}'.format(index))
        return im

    anim = pltanim.FuncAnimation(
        fig, update, init_func=init, frames=len(images), interval=10, repeat=repeat)
    if save_gif:
        anim.save('anim.gif', writer='imagemagick')
    block = not auto_close
    plt.show(block=block)
