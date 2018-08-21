def hwc2chw(image):
    return image.transpose((2, 0, 1))


def chw2hwc(image):
    return image.transpose((1, 2, 0))