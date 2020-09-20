import nnabla as nn
import nnabla.parametric_functions as PF


def action_conditioned_lstm_cell(x, a, h, c, state_size, out_size,
                                 wh_init=None, bh_init=None,
                                 wa_init=None, ba_init=None,
                                 w_lstm_init=None, b_lstm_init=None,
                                 fix_parameters=False, name=None,
                                 rng=None):
    """
    LSTM layer which combines input and internal state with given action
    See: https://arxiv.org/pdf/1704.02254.pdf
    """
    scope_name = '' if name is None else name + '/'
    with nn.parameter_scope(scope_name + 'Wh'):
        Wht = PF.affine(h, n_outmaps=state_size, with_bias=False,
                        w_init=wh_init, b_init=bh_init,
                        fix_parameters=fix_parameters,
                        rng=rng)
    with nn.parameter_scope(scope_name + 'Wa'):
        Wat = PF.affine(a, n_outmaps=state_size, with_bias=False,
                        w_init=wa_init, b_init=ba_init,
                        fix_parameters=fix_parameters,
                        rng=rng)
    vt = Wht * Wat
    with nn.parameter_scope(scope_name + 'lstm_cell'):
        return PF.lstm_cell(x, vt, c,
                            state_size=out_size,
                            w_init=w_lstm_init,
                            b_init=b_lstm_init,
                            fix_parameters=fix_parameters)


class ActionConditionedLSTMCell(object):
    def __init__(self, batch_size, state_size, out_size, h=None, c=None, name=None):
        """
        Initializes an LSTM cell.

        Args:
            batch_size (int): Internal batch size is set to `batch_size`.
            state_size (int): Internal state size (size of vt) is set to `state_size`.
            out_size (int): Output variable size is set to `out_size`.
            h (~nnabla.Variable): Input N-D array with shape (batch_size, out_size). If not specified, it is initialized to zero by default.
            c (~nnabla.Variable): Input N-D array with shape (batch_size, out_size). If not specified, it is initialized to zero by default.
            name (str): Name for this LSTM Cell.
        """
        self.batch_size = batch_size
        self.state_size = state_size
        self.out_size = out_size
        self.name = name
        if h:  # when user defines h
            self.h = h
        else:
            self.h = nn.Variable((self.batch_size, self.out_size))
            self.h.data.zero()
        if c:  # when user defines c
            self.c = c
        else:
            self.c = nn.Variable((self.batch_size, self.out_size))
            self.c.data.zero()

    def reset_state(self):
        """
        Resets states h and c to zero.
        """

        self.h.data.zero()
        self.c.data.zero()

    def __call__(self, x, a,
                 wh_init=None, bh_init=None,
                 wa_init=None, ba_init=None,
                 w_lstm_init=None, b_lstm_init=None,
                 fix_parameters=False):
        """
        Updates h and c by calling lstm function.

        Args:
            x (~nnabla.Variable): Input N-D array with shape (batch_size, input_size).
            w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`, optional): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.  
            b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`, optional): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
            fix_parameters (bool): When set to `True`, the weights and biases will not be updated.

        """
        self.h, self.c = action_conditioned_lstm_cell(
            x, a, self.h, self.c, self.state_size, self.out_size,
            wh_init=wh_init, bh_init=bh_init,
            wa_init=wa_init, ba_init=wa_init,
            w_lstm_init=w_lstm_init, b_lstm_init=w_lstm_init,
            fix_parameters=fix_parameters, name=self.name)
        return self.h
