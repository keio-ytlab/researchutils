def hwc2chw(image):
    """
    Changes the order of image pixels from Height-Width-Color to Color-Height-Width

    Parameters
    -------
    image : numpy.ndarray
        Image with pixels in Height-Width-Color order

    Returns
    -------
    image : numpy.ndarray
        Image with pixels in Color-Height-Width order
    """
    return image.transpose((2, 0, 1))


def chw2hwc(image):
    """
    Changes the order of image pixels from Color-Height-Width to Height-Width-Color

    Parameters
    -------
    image : numpy.ndarray
        Image with pixels in Color-Height-Width order

    Returns
    -------
    image : numpy.ndarray
        Image with pixels in Height-Width-Color order
    """
    return image.transpose((1, 2, 0))
