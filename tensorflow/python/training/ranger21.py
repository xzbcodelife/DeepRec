# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ranger21 for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework.embedding import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import kv_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import slot_creator
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import gen_hash_training_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.training import training_util
from tensorflow.python.util.tf_export import tf_export
import numpy as np


@tf_export(v1=["train.Ranger21Optimizer"])
class Ranger21Optimizer(optimizer.Optimizer):
    """Optimizer that implements the Ranger21 algorithm.

    See [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    """

    def __init__(self,
                 global_step=0,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 epsilon_clipping=1e-3,
                 weight_decay=1e-4,
                 beta0=0.9,
                 beta_lookahead=0.5,
                 tau_clipping=1e-2,
                 kappa_lookahead=5,
                 use_adaptive_grad_clipping=False,
                 use_grad_centralization=False,
                 use_positive_negative_momentum=False,
                 use_norm_loss=False,
                 use_stable_weight_decay=False,
                 use_lookahead=False,
                 use_locking=False,
                 name="Ranger21"):
        r"""Construct a new Ranger21 optimizer.

        Initialization:

        $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
        $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
        $$t := 0 \text{(Initialize timestep)}$$

        The update rule for `variable` with gradient `g` uses an optimization
        described at the end of section 2 of the paper:

        $$t := t + 1$$
        $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$

        $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
        $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
        $$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$

        The default value of 1e-8 for epsilon might not be a good default in
        general. For example, when training an Inception network on ImageNet a
        current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
        formulation just before Section 2.1 of the Kingma and Ba paper rather than
        the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
        hat" in the paper.

        The sparse implementation of this algorithm (used when the gradient is an
        IndexedSlices object, typically because of `tf.gather` or an embedding
        lookup in the forward pass) does apply momentum to variable slices even if
        they were not used in the forward pass (meaning they have a gradient equal
        to zero). Momentum decay (beta1) is also applied to the entire momentum
        accumulator. This means that the sparse behavior is equivalent to the dense
        behavior (in contrast to some momentum implementations which ignore momentum
        unless a variable slice was actually used).

        Args:
          learning_rate: A Tensor or a floating point value.  The learning rate.
          beta1: A float value or a constant float tensor. The exponential decay
            rate for the 1st moment estimates.
          beta2: A float value or a constant float tensor. The exponential decay
            rate for the 2nd moment estimates.
          epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
          use_locking: If True use locks for update operations.
          name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".  @compatibility(eager) When eager execution is
            enabled, `learning_rate`, `beta1`, `beta2`, and `epsilon` can each be a
            callable that takes no arguments and returns the actual value to use.
            This can be useful for changing these values across different
            invocations of optimizer functions. @end_compatibility
        """
        super(Ranger21Optimizer, self).__init__(use_locking, name)
        self._global_step = global_step
        self._global_step_on_worker = None
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._epsilon_clipping = epsilon_clipping
        self._weight_decay = weight_decay
        self._beta0 = beta0
        self._beta_lookahead = beta_lookahead
        self._tau_clipping = tau_clipping
        self._kappa_lookahead = kappa_lookahead

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None
        self._epsilon_clipping_t = None
        self._weight_decay_t = None
        self._beta0_t = None
        self._beta_lookahead_t = None
        self._tau_clipping_t = None
        self._kappa_lookahead_t = None

        self._use_adaptive_grad_clipping = use_adaptive_grad_clipping
        self._use_grad_centralization = use_grad_centralization
        self._use_positive_negative_momentum = use_positive_negative_momentum
        self._use_norm_loss = use_norm_loss
        self._use_stable_weight_decay = use_stable_weight_decay
        self._use_lookahead = use_lookahead
        self._use_locking = use_locking

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).

        for v in var_list:
            with ops.colocate_with(v):
                self._zeros_slot(v, "m", self._name, slot_config=slot_creator.SlotConfig(
                    slot_index=1, slot_num=2))
                self._zeros_slot(v, "v", self._name, slot_config=slot_creator.SlotConfig(
                    slot_index=2, slot_num=2))
                self._zeros_slot(v, "v_max", self._name, slot_config=slot_creator.SlotConfig(
                    slot_index=2, slot_num=2))
                self._zeros_slot(v, "lookahead_var", self._name, slot_config=slot_creator.SlotConfig(
                    slot_index=2, slot_num=2))

                if isinstance(v, kv_variable_ops.EmbeddingVariable):
                    self._get_or_make_slot(v,
                                           array_ops.expand_dims(
                                               ops.convert_to_tensor(self._beta1, dtype=v.dtype.base_dtype), -1),
                                           "beta1_power", self._name, slot_config=slot_creator.SlotConfig(slot_type=config_pb2.SlotType.VARIABLE))
                    self._get_or_make_slot(v,
                                           array_ops.expand_dims(
                                               ops.convert_to_tensor(self._beta2, dtype=v.dtype.base_dtype), -1),
                                           "beta2_power", self._name, slot_config=slot_creator.SlotConfig(slot_type=config_pb2.SlotType.VARIABLE))
                else:
                    self._get_or_make_slot(v,
                                           ops.convert_to_tensor(
                                               self._beta1, dtype=v.dtype.base_dtype),
                                           "beta1_power", self._name)
                    self._get_or_make_slot(v,
                                           ops.convert_to_tensor(
                                               self._beta2, dtype=v.dtype.base_dtype),
                                           "beta2_power", self._name)

    def _prepare(self):
        lr = self._call_if_callable(self._lr)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        epsilon = self._call_if_callable(self._epsilon)
        epsilon_clipping = self._call_if_callable(self._epsilon_clipping)
        weight_decay = self._call_if_callable(self._weight_decay)
        beta0 = self._call_if_callable(self._beta0)
        beta_lookahead = self._call_if_callable(self._beta_lookahead)
        tau_clipping = self._call_if_callable(self._tau_clipping)
        kappa_lookahead = self._call_if_callable(self._kappa_lookahead)

        self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")
        self._epsilon_clipping_t = ops.convert_to_tensor(
            epsilon_clipping, name="epsilon_clipping")
        self._weight_decay_t = ops.convert_to_tensor(
            weight_decay, name="weight_decay")
        self._beta0_t = ops.convert_to_tensor(beta0, name="beta0")
        self._beta_lookahead_t = ops.convert_to_tensor(
            beta_lookahead, name="beta_lookahead")
        self._tau_clipping_t = ops.convert_to_tensor(
            tau_clipping, name="tau_clipping")
        self._kappa_lookahead_t = ops.convert_to_tensor(
            kappa_lookahead, name="kappa_lookahead")
        with ops.colocate_with(self._lr_t):
            self._global_step_on_worker = array_ops.identity(
                self._global_step) + 1

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        v_max = self.get_slot(var, "v_max")
        lookahead_var = self.get_slot(var, "lookahead_var")

        beta1_power = self.get_slot(var, 'beta1_power')
        beta2_power = self.get_slot(var, 'beta2_power')
        with ops.device(var.device):
            global_step = array_ops.identity(self._global_step_on_worker)

        return training_ops.apply_ranger21(
            var,
            m,
            v,
            v_max,
            lookahead_var,
            beta1_power,
            beta2_power,
            global_step,
            math_ops.cast(self._lr_t, var.dtype.base_dtype),
            math_ops.cast(self._beta1_t, var.dtype.base_dtype),
            math_ops.cast(self._beta2_t, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_clipping_t, var.dtype.base_dtype),
            math_ops.cast(self._weight_decay_t, var.dtype.base_dtype),
            math_ops.cast(self._beta0_t, var.dtype.base_dtype),
            math_ops.cast(self._beta_lookahead_t, var.dtype.base_dtype),
            math_ops.cast(self._tau_clipping_t, var.dtype.base_dtype),
            self._kappa_lookahead_t,
            grad,
            self._use_adaptive_grad_clipping,
            self._use_grad_centralization,
            self._use_positive_negative_momentum,
            self._use_norm_loss,
            self._use_stable_weight_decay,
            self._use_lookahead,
            use_locking=self._use_locking).op

    def _resource_apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        v_max = self.get_slot(var, "v_max")
        lookahead_var = self.get_slot(var, "lookahead_var")

        beta1_power = self.get_slot(var, 'beta1_power')
        beta2_power = self.get_slot(var, 'beta2_power')
        with ops.device(var.device):
            global_step = array_ops.identity(self._global_step_on_worker)

        return training_ops.resource_apply_ranger21(
            var.handle,
            m.handle,
            v.handle,
            v_max.handle,
            lookahead_var.handle,
            beta1_power.handle,
            beta2_power.handle,
            global_step,
            math_ops.cast(self._lr_t, var.dtype.base_dtype),
            math_ops.cast(self._beta1_t, var.dtype.base_dtype),
            math_ops.cast(self._beta2_t, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_clipping_t, var.dtype.base_dtype),
            math_ops.cast(self._weight_decay_t, var.dtype.base_dtype),
            math_ops.cast(self._beta0_t, var.dtype.base_dtype),
            math_ops.cast(self._beta_lookahead_t, var.dtype.base_dtype),
            math_ops.cast(self._tau_clipping_t, var.dtype.base_dtype),
            self._kappa_lookahead_t,
            grad,
            self._use_adaptive_grad_clipping,
            self._use_grad_centralization,
            self._use_positive_negative_momentum,
            self._use_norm_loss,
            self._use_stable_weight_decay,
            self._use_lookahead,
            self._use_locking
            )

    def _apply_sparse(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        v_max = self.get_slot(var, "v_max")
        lookahead_var = self.get_slot(var, "lookahead_var")

        beta1_power = self.get_slot(var, 'beta1_power')
        beta2_power = self.get_slot(var, 'beta2_power')
        with ops.device(var.device):
            global_step = array_ops.identity(self._global_step_on_worker)

        return training_ops.sparse_apply_ranger21(
            var,
            m,
            v,
            v_max,
            lookahead_var,
            beta1_power,
            beta2_power,
            global_step,
            math_ops.cast(self._lr_t, var.dtype.base_dtype),
            math_ops.cast(self._beta1_t, var.dtype.base_dtype),
            math_ops.cast(self._beta2_t, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_clipping_t, var.dtype.base_dtype),
            math_ops.cast(self._weight_decay_t, var.dtype.base_dtype),
            math_ops.cast(self._beta0_t, var.dtype.base_dtype),
            math_ops.cast(self._beta_lookahead_t, var.dtype.base_dtype),
            math_ops.cast(self._tau_clipping_t, var.dtype.base_dtype),
            self._kappa_lookahead_t,
            self._use_adaptive_grad_clipping,
            self._use_grad_centralization,
            self._use_positive_negative_momentum,
            self._use_norm_loss,
            self._use_stable_weight_decay,
            self._use_lookahead,
            grad.values, grad.indices,
            self._use_locking,
        )

    def _resource_apply_sparse(self, grad, var, indices):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        v_max = self.get_slot(var, "v_max")
        lookahead_var = self.get_slot(var, "lookahead_var")

        beta1_power = self.get_slot(var, 'beta1_power')
        beta2_power = self.get_slot(var, 'beta2_power')
        with ops.device(var.device):
            global_step = array_ops.identity(self._global_step_on_worker)

        if isinstance(var, kv_variable_ops.EmbeddingVariable):
            return training_ops.kv_resource_sparse_apply_ranger21(
                var.handle,
                m.handle,
                v.handle,
                v_max.handle,
                lookahead_var.handle,
                beta1_power.handle,
                beta2_power.handle,
                global_step,
                math_ops.cast(self._lr_t, var.dtype.base_dtype),
                math_ops.cast(self._beta1_t, var.dtype.base_dtype),
                math_ops.cast(self._beta2_t, var.dtype.base_dtype),
                math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
                math_ops.cast(self._epsilon_clipping_t, var.dtype.base_dtype),
                math_ops.cast(self._weight_decay_t, var.dtype.base_dtype),
                math_ops.cast(self._beta0_t, var.dtype.base_dtype),
                math_ops.cast(self._beta_lookahead_t, var.dtype.base_dtype),
                math_ops.cast(self._tau_clipping_t, var.dtype.base_dtype),
                self._kappa_lookahead_t,
                self._use_adaptive_grad_clipping,
                self._use_grad_centralization,
                self._use_positive_negative_momentum,
                self._use_norm_loss,
                self._use_stable_weight_decay,
                self._use_lookahead,
                grad, indices, self._use_locking)
        else:
            return training_ops.resource_sparse_apply_ranger21(
                m.handle,
                v.handle,
                v_max.handle,
                lookahead_var.handle,
                beta1_power.handle,
                beta2_power.handle,
                global_step,
                math_ops.cast(self._lr_t, var.dtype.base_dtype),
                math_ops.cast(self._beta1_t, var.dtype.base_dtype),
                math_ops.cast(self._beta2_t, var.dtype.base_dtype),
                math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
                math_ops.cast(self._epsilon_clipping_t, var.dtype.base_dtype),
                math_ops.cast(self._weight_decay_t, var.dtype.base_dtype),
                math_ops.cast(self._beta0_t, var.dtype.base_dtype),
                math_ops.cast(self._beta_lookahead_t, var.dtype.base_dtype),
                math_ops.cast(self._tau_clipping_t, var.dtype.base_dtype),
                self._kappa_lookahead_t,
                self._use_adaptive_grad_clipping,
                self._use_grad_centralization,
                self._use_positive_negative_momentum,
                self._use_norm_loss,
                self._use_stable_weight_decay,
                self._use_lookahead,
                grad, indices, self._use_locking)
