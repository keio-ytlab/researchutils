============================
Utilities related to chainer
============================

Activation functions
=======================

.. automodule:: researchutils.chainer.functions.activation.grad_clip_lstm
   :members:

Loss functions
=======================

.. automodule:: researchutils.chainer.functions.loss.average_k_step_squared_error
   :members:


Iterators
=======================

.. autoclass:: researchutils.chainer.iterators.decorable_multithread_iterator.DecorableMultithreadIterator

.. autoclass:: researchutils.chainer.iterators.decorable_serial_iterator.DecorableSerialIterator

.. autoclass:: researchutils.chainer.iterators.decorable_multiprocess_iterator.DecorableMultiprocessIterator

.. autoclass:: researchutils.chainer.iterators.unmodifiable_decorable_list.UnmodifiableDecorableList


Connection links
=======================

.. autoclass:: researchutils.chainer.links.connection.grad_clip_lstm.GradClipLSTM

Serializers
=======================

.. automodule:: researchutils.chainer.serializers.npz
   :members:

Training extensions
=======================

.. autoclass:: researchutils.chainer.training.extensions.slack_report.SlackReport