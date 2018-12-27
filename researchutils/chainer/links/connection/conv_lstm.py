import chainer
import chainer.links as L
import chainer.functions as F


class ConvLSTM(chainer.Chain):
    """
    LSTM layer which processes image input directly instead of conventional LSTM layer
    See: https://arxiv.org/abs/1506.04214

    If stride is set, this layer outputs image of size input image size divided by stride

    Parameters
    -------
    in_channels : int
        number of input channels
    out_channels : int 
        number of output channels
    ksize: int or tuple of int
        height and width of the kernel (aka filter). The size must be odd number.
    stride: int or tuple of int
        step size of kernel to x and y direction. Input image size must be divisible by given stride.
    initialW: chainer.initializer
        initializer to initialize weights used by convolution layer
    initial_bias: chainer.initializer
        initializer to initialize bias used by convolution layer
        If ``None``, the bias will be initialized to zero

    Raises
    -------
    ValueError
        ksize is even number
    """


    def __init__(self, in_channels, out_channels, ksize, stride=1, initialW=None, initial_bias=None):
        super(ConvLSTM, self).__init__()
        self.cell_output = None
        self.hidden_state = None
        self.out_channels = out_channels
        self.stride = self._pair(stride)
        self.ksize = self._pair(ksize)
        if self.ksize[0] % 2 == 0 or self.ksize[1] %2 == 0:
            raise ValueError('kernel size must be odd number. Given ksize: {}'.format(ksize))
        with self.init_scope():
            pad = (self.ksize[0] // 2, self.ksize[1] // 2)
            self.Wxi = L.Convolution2D(in_channels=in_channels, out_channels=out_channels,
                                       ksize=self.ksize, stride=self.stride, pad=pad, nobias=False,
                                       initialW=initialW, initial_bias=initial_bias)
            self.Wxf = L.Convolution2D(in_channels=in_channels, out_channels=out_channels,
                                       ksize=self.ksize, stride=self.stride, pad=pad, nobias=False,
                                       initialW=initialW, initial_bias=initial_bias)
            self.Wxo = L.Convolution2D(in_channels=in_channels, out_channels=out_channels,
                                       ksize=self.ksize, stride=self.stride, pad=pad, nobias=False,
                                       initialW=initialW, initial_bias=initial_bias)
            self.Wxc = L.Convolution2D(in_channels=in_channels, out_channels=out_channels,
                                       ksize=self.ksize, stride=self.stride, pad=pad, nobias=False,
                                       initialW=initialW, initial_bias=initial_bias)

            self.Whi = L.Convolution2D(in_channels=out_channels, out_channels=out_channels,
                                       ksize=self.ksize, stride=1, pad=pad, nobias=True)
            self.Whf = L.Convolution2D(in_channels=out_channels, out_channels=out_channels,
                                       ksize=self.ksize, stride=1, pad=pad, nobias=True)
            self.Who = L.Convolution2D(in_channels=out_channels, out_channels=out_channels,
                                       ksize=self.ksize, stride=1, pad=pad, nobias=True)
            self.Whc = L.Convolution2D(in_channels=out_channels, out_channels=out_channels,
                                       ksize=self.ksize, stride=1, pad=pad, nobias=True)

            initializer = chainer.initializers.Zero()
            self.Wci = chainer.Parameter(initializer)
            self.Wcf = chainer.Parameter(initializer)
            self.Wco = chainer.Parameter(initializer)

    def __call__(self, x):
        if not self._internal_state_initialized():
            shape = x.shape
            target_shape = (shape[0], self.out_channels, int(shape[2] / self.stride[0]), int(shape[3] / self.stride[1]))
            self._initialize_internal_states(target_shape)

        if not self._non_convolutional_weights_initialized():
            shape = x.shape
            target_shape = (1, self.out_channels, int(shape[2] / self.stride[0]), int(shape[3] / self.stride[1]))
            self._initialize_non_convolutional_weights(target_shape)

        it = F.sigmoid(self.Wxi(x) + self.Whi(self.hidden_state) + self.Wci * self.cell_output)
        ft = F.sigmoid(self.Wxf(x) + self.Whf(self.hidden_state) + self.Wcf * self.cell_output)
        ct = ft * self.cell_output + it * F.tanh(self.Wxc(x) + self.Whc(self.hidden_state))
        ot = F.sigmoid(self.Wxo(x) + self.Who(self.hidden_state) + self.Wco * ct)
        ht = ot * F.tanh(ct)

        self.cell_output = ct
        self.hidden_state = ht

        return ht

    def reset_state(self, c=None, h=None):
        self.cell_output = c
        self.hidden_state = h

    def _initialize_internal_states(self, shape):
        # shape is a tuple which contains (batch_size, channels, height, width)
        if self.cell_output is None:
            self.cell_output = chainer.Variable(
                self.xp.zeros(shape, dtype=self.xp.float32))
        if self.hidden_state is None:
            self.hidden_state = chainer.Variable(
                self.xp.zeros(shape, dtype=self.xp.float32))

    def _internal_state_initialized(self):
        return self.cell_output is not None and self.hidden_state is not None

    def _initialize_non_convolutional_weights(self, shape):
        if self.Wci.array is None:
            self.Wci.initialize(shape)
        if self.Wcf.array is None:
            self.Wcf.initialize(shape)
        if self.Wco.array is None:
            self.Wco.initialize(shape)

    def _non_convolutional_weights_initialized(self):
        return self.Wci.array is not None and self.Wcf.array is not None and self.Wco.array is not None

    def _pair(self, x):
        if hasattr(x, '__getitem__'):
            return x
        return x, x
