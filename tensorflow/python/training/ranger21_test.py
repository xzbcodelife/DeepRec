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
"""Tests for Ranger21."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import hash_table
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import ranger21


def ranger21_update_numpy(t, 
                          param,
                          g_t,
                          m,
                          v,
                          v_max,
                          lr=0.001,
                          beta0=0.9,
                          beta1=0.9,
                          beta2=0.999,
                          epsilon=1e-8,
                          weight_decay=1e-4,
                          use_positive_negative_momentum=True):

    print("="*200)

    beta1_power = beta1**t
    # beta1_power = np.power(beta1, t)
    beta2_power = beta2**t

    print('param', param)
    print('g_t', g_t)
    print('m', m)
    print('v', v)
    print('v_max', v_max)
    print('beta1_power', beta1_power)
    print('beta2_power', beta2_power)

    if use_positive_negative_momentum:
        beta1_square = beta1*beta1
        m_t = beta1_square * m + (1 - beta1_square) * g_t
        print('m_t', m_t)

        m_bias_correction_t = (1+beta0)*m_t-beta0*m
        print('m_bias_correction_t', m_bias_correction_t)
        m_bias_correction_t = ((1+beta0)*m_t-beta0*m) / (1-beta1_power)
        print('m_bias_correction_t', m_bias_correction_t)

        v_t = beta2 * v + (1 - beta2) * g_t * g_t

        # print("v_t.shape", v_t.shape)
        # print("v_max.shape", v_max.shape)
        print('v_t', v_t)

        v_max_t = np.fmax(v_t, v_max)
        print('v_max_t', v_max_t)

        v_bias_correction_t = np.divide(v_max_t, 1-beta2_power)
        print('v_bias_correction_t', v_bias_correction_t)

        u = m_bias_correction_t / \
            (np.sqrt((1+beta0)**2+beta0**2) *
             (np.sqrt(v_bias_correction_t) + epsilon))
        print('u', u)

        param_t = param - lr*u
        print('param_t', param_t)

        param_t = param - lr*u - lr*weight_decay*param
        print('param_t', param_t)

        return param_t, m_t, v_t, v_max_t
    else:
        alpha_t = lr * np.sqrt(1 - beta2_power) / (1 - beta1_power)
        m_t = beta1 * m + (1 - beta1) * g_t
        v_t = beta2 * v + (1 - beta2) * g_t * g_t

        # print('m', m)
        # print('v', v)

        print('m_t', m_t)
        print('v_t', v_t)

        param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon) - lr*weight_decay*param

        return param_t, m_t, v_t


class Ranger21OptimizerTest(test.TestCase):
    def testDenseCpu(self):
        dtype = dtypes.float32
        with self.session(graph=ops.Graph(), use_gpu=True), ops.device("/device:GPU:0"):
        # with self.session(graph=ops.Graph(), use_gpu=True):
            # Initialize variables for numpy implementation.
            m0 = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
            v0 = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
            v_max_0 = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)

            m1 = np.array([0.0, 0.0, 0.0], dtype=dtype.as_numpy_dtype)
            v1 = np.array([0.0, 0.0, 0.0], dtype=dtype.as_numpy_dtype)
            v_max_1 = np.array([0.0, 0.0, 0.0], dtype=dtype.as_numpy_dtype)

            var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
            grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)

            var1_np = np.array([3.0, 4.0, 5.0], dtype=dtype.as_numpy_dtype)
            grads1_np = np.array([0.01, 0.01, 0.01], dtype=dtype.as_numpy_dtype)

            # var0 = variables.Variable(var0_np)
            # var1 = variables.Variable(var1_np)

            var0 = resource_variable_ops.ResourceVariable(var0_np, name="var0_0")
            var1 = resource_variable_ops.ResourceVariable(var1_np, name="var1_0")

            grads0 = constant_op.constant(grads0_np)
            grads1 = constant_op.constant(grads1_np)

            use_positive_negative_momentum = True
            grads_vars =  zip([grads0], [var0])
            # grads_vars = zip([grads0, grads1], [var0, var1])
            # update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
            with ops.device('/device:GPU:0'):
                opt = ranger21.Ranger21Optimizer(
                use_positive_negative_momentum=use_positive_negative_momentum)
                update = opt.apply_gradients(grads_vars)

            opt_variables = opt.variables()
            beta1_power = opt.get_slot(var0, 'beta1_power')
            beta2_power = opt.get_slot(var0, 'beta2_power')
            self.assertTrue(beta1_power is not None)
            self.assertTrue(beta2_power is not None)
            self.assertIn(beta1_power, opt_variables)
            self.assertIn(beta2_power, opt_variables)

            if not context.executing_eagerly():
                with ops.Graph().as_default():
                    # Shouldn't return non-slot variables from other graphs.
                    self.assertEqual(0, len(opt.variables()))

                self.evaluate(variables.global_variables_initializer())
            # self.assertAllClose([1.0, 2.0], self.evaluate(var0))
            # self.assertAllClose([3.0, 4.0], self.evaluate(var1))

            beta1_power = opt.get_slot(var0, 'beta1_power')
            beta2_power = opt.get_slot(var0, 'beta2_power')
            m_var = opt.get_slot(var0, 'm')
            v_var = opt.get_slot(var0, 'v')

            # Run 3 steps of Ranger21
            for t in range(1, 2):
                print("t=", t)
                if not context.executing_eagerly():
                    self.evaluate(update)
                elif t > 1:
                    # opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                    opt.apply_gradients(grads_vars)
                    # opt.apply_gradients(zip([grads0], [var0]))

                if use_positive_negative_momentum:
                    var0_np, m0, v0, v_max_0 = ranger21_update_numpy(
                        t, var0_np, grads0_np, m0, v0, v_max_0, use_positive_negative_momentum=use_positive_negative_momentum)

                    # var1_np, m1, v1, v_max_1 = ranger21_update_numpy(
                    #     t, var1_np, grads1_np, m1, v1, v_max_1, use_positive_negative_momentum=use_positive_negative_momentum)
                else:
                    var0_np, m0, v0 = ranger21_update_numpy(
                        t, var0_np, grads0_np, m0, v0, v_max_0, use_positive_negative_momentum=use_positive_negative_momentum)

                    # var1_np, m1, v1 = ranger21_update_numpy(
                    #     t, var1_np, grads1_np, m1, v1, v_max_1, use_positive_negative_momentum=use_positive_negative_momentum)

                # beta1_power_eval = self.evaluate(beta1_power)
                # print("beta1_power_eval", beta1_power_eval)

                # beta2_power_eval = self.evaluate(beta2_power)
                # print("beta2_power_eval", beta2_power_eval)

                # print("beta1_power_np", beta1_power_np)
                # print("beta2_power_np", beta2_power_np)

                # self.assertAllCloseAccordingToType(beta1_power_np, beta1_power_eval)
                # self.assertAllCloseAccordingToType(beta2_power_np, beta2_power_eval)


                # m_var_eval = self.evaluate(m_var)
                # v_var_eval = self.evaluate(v_var)
                var0_eval = self.evaluate(var0)

                # print('m0', m0, 'm_var_eval', m_var_eval)
                # print('v0', v0, 'v_var_eval', v_var_eval)
                print('var0_np',var0_np, 'var0_eval', var0_eval)
                print('update.device', update.device)

                # self.assertAllCloseAccordingToType(m0, m_var_eval)
                # self.assertAllCloseAccordingToType(v0, v_var_eval)
                self.assertAllCloseAccordingToType(var0_np, var0_eval, rtol=1e-3, atol=1e-3)
                

                # if use_positive_negative_momentum:
                #     var1_np, m1, v1, v_max_1 = ranger21_update_numpy(
                #         t, var1_np, grads1_np, m1, v1, v_max_1, use_positive_negative_momentum=use_positive_negative_momentum)
                # else:
                #     var1_np, m1, v1 = ranger21_update_numpy(
                #         t, var1_np, grads1_np, m1, v1, v_max_1, use_positive_negative_momentum=use_positive_negative_momentum)

                # # m_var_eval = self.evaluate(m_var)
                # # v_var_eval = self.evaluate(v_var)
                # var1_eval = self.evaluate(var1)

                # # print('m1', m1, 'm_var_eval', m_var_eval)
                # # print('v1', v1, 'v_var_eval', v_var_eval)
                # print('var1_np',var1_np, 'var1_eval', var1_eval)

                #   # # Validate updated params
                # # self.assertAllCloseAccordingToType(m1, m_var_eval)
                # # self.assertAllCloseAccordingToType(v1, v_var_eval)
                # self.assertAllCloseAccordingToType(var1_np, var1_eval, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test.main()
