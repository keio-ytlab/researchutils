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


def animate(images, is_gray=False, save_gif=False):
    fig = plt.figure()
    im = create_window(images[0], is_gray=is_gray)

    def init():
        im.set_data(images[0])

    def update(index):
        im.set_data(images[index])
        return im

    anim = pltanim.FuncAnimation(
        fig, update, init_func=init, frames=len(images), interval=10)
    if save_gif:
        anim.save('anim.gif', writer='imagemagick')
    plt.show()
