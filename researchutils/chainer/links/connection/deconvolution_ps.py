import chainer
import chainer.links as L
import chainer.functions as F

from chainer.utils import conv


class DeconvolutionPS(chainer.Chain):
    """
    Two dimensional deconvolution layer with convolutional layer followed by pixel shuffler
    See: https://arxiv.org/abs/1609.05158

    This layer will automatically pad the input with zeros if height and width of given input
    does not have enough size to upscale

    Parameters
    -------
    in_channels : int
        number of input channels
    out_channels : int 
        number of output channels
    ksize: int or tuple of int
        height and width of the kernel (aka filter)
    stride: int or tuple of int
        step size of kernel to x and y direction
    pad: int of tuple of int
        padding size to add to input image
    nobias: bool
        ``True`` to add bias term while convolution
    outsize: tuple of int
        size of upscaled image
    initialW: chainer.initializer
        initializer to initialize weights used by convolution
    initial_bias: chainer.initializer
        initializer to initialize bias used by convolution
        If ``None``, the bias will be initialized to zero

    Raises
    -------
    ValueError
        ksize is not divisible by given stride
    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, nobias=False, outsize=None, initialW=None, initial_bias=None, **kwargs):
        super(DeconvolutionPS, self).__init__()
        r = stride
        self.r = r
        self.out_h, self.out_w = (None, None) if outsize is None else outsize
        kh, kw = self._pair(ksize)
        if kh % r is not 0 or kw % r is not 0:
            raise ValueError(
                'ksize must be divisible by stride. {} % {} != 0'.format(ksize, r))
        self.kh, self.kw = kh // r, kw // r
        self.sy, self.sx = (1, 1)
        self.ph, self.pw = (None, None)
        self.ph_mid, self.pw_mid = (None, None)

        # Keep original values for later computation
        self.orig_kh, self.orig_kw = self._pair(ksize)
        self.orig_sy, self.orig_sx = self._pair(stride)
        self.orig_ph, self.orig_pw = self._pair(pad)

        with self.init_scope():
            out_channels = out_channels * r * r
            self.conv = L.Convolution2D(in_channels=in_channels, out_channels=out_channels,
                                        ksize=(self.kh, self.kw), stride=1, nobias=nobias,
                                        initialW=initialW, initial_bias=initial_bias, **kwargs)

    def __call__(self, x):
        _, _, in_h, in_w = x.shape
        self._compute_outsize(in_h, in_w)
        self._compute_padsize(in_h, in_w, self.out_h, self.out_w)
        x = F.pad(x, ((0, 0), (0, 0), (self.ph_mid, self.ph - self.ph_mid),
                      (self.pw_mid, self.pw - self.pw_mid)), mode='constant')
        h = self.conv(x)
        return F.depth2space(h, self.r)

    def _compute_outsize(self, in_h, in_w):
        if self.out_h is None:
            self.out_h = conv.get_deconv_outsize(
                in_h, self.orig_kh, self.orig_sy, self.orig_ph, d=1) // self.r

        if self.out_w is None:
            self.out_w = conv.get_deconv_outsize(
                in_w, self.orig_kw, self.orig_sx, self.orig_pw, d=1) // self.r

    def _compute_padsize(self, in_h, in_w, out_h, out_w):
        if self.ph is None:
            ph = (out_h - 1) * self.sy + self.kh - in_h
            if ph <= 0:
                raise RuntimeError('Padding height must be positive.')
            self.ph = ph
            self.ph_mid = self.ph // 2

        if self.pw is None:
            pw = (out_w - 1) * self.sx + self.kw - in_w
            if pw <= 0:
                raise RuntimeError('Padding width must be positive.')
            self.pw = pw
            self.pw_mid = self.pw // 2

    def _pair(self, x):
        if hasattr(x, '__getitem__'):
            return x
        return x, x
