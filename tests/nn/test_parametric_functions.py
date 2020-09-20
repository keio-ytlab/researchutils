import pytest
import numpy as np

import nnabla as nn
import nnabla.parametric_functions as PF
import researchutils.nn.parametric_functions as RPF


class TestParametricFunctions(object):
    def setup_method(self, method):
        nn.clear_parameters()

    @pytest.mark.parametrize("in_size, state_size, out_size, action_size",
                             [(i * 10, i * 15, i * 20, i) for i in range(1, 10)])
    def test_action_conditioned_lstm(self, in_size, state_size, out_size, action_size):
        batch_size = 5
        x_shape = (batch_size, in_size)
        a_shape = (batch_size, action_size)
        h_expect = nn.Variable.from_numpy_array(
            np.zeros((batch_size, out_size)))
        c_expect = nn.Variable.from_numpy_array(
            np.zeros((batch_size, out_size)))
        h_actual = nn.Variable.from_numpy_array(
            np.zeros((batch_size, out_size)))
        c_actual = nn.Variable.from_numpy_array(
            np.zeros((batch_size, out_size)))

        for _ in range(1, 10):
            x = np.random.uniform(-1, 1, x_shape).astype(np.float32)
            x = nn.Variable.from_numpy_array(x)

            a = np.random.uniform(-1, 1, a_shape).astype(np.float32)
            a = nn.Variable.from_numpy_array(a)

            rng = np.random.RandomState(seed=0)
            h_expect, c_expect = self._compute_aclstm_cell(
                x, a, h_expect, c_expect, state_size, out_size, rng=rng)

            rng = np.random.RandomState(seed=0)
            h_actual, c_actual = RPF.action_conditioned_lstm_cell(
                x, a, h_actual, c_actual, state_size, out_size, rng=rng)

            nn.forward_all([c_expect, h_expect, c_actual, h_actual])
            assert h_actual.d == pytest.approx(h_expect.d)
            assert c_actual.d == pytest.approx(c_expect.d)

    def _compute_aclstm_cell(self, x, a, h, c, state_size, out_size, rng):
        batch_size = x.shape[0]
        with nn.parameter_scope('test/Wh'):
            Wht = PF.affine(h, n_outmaps=state_size, with_bias=False,
                            rng=rng)
        assert Wht.shape == (batch_size, state_size)
        with nn.parameter_scope('test/Wa'):
            Wat = PF.affine(a, n_outmaps=state_size, with_bias=False,
                            rng=rng)
        assert Wat.shape == (batch_size, state_size)

        vt = Wht * Wat
        assert vt.shape == (batch_size, state_size)
        # use same lstm cell for testing by specifing parameter scope
        with nn.parameter_scope('lstm_cell'):
            return PF.lstm_cell(x, vt, c, state_size=out_size)


if __name__ == '__main__':
    pytest.main()
