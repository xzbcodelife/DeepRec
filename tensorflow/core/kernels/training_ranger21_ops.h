/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_KERNELS_TRAINING_RANGER21_OPS_H_
#define TENSORFLOW_CORE_KERNELS_TRAINING_RANGER21_OPS_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct ApplyRanger21 {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::Flat v_max,
                  typename TTypes<T>::Flat lookahead_var,
                  typename TTypes<T>::Scalar beta1_power,
                  typename TTypes<T>::Scalar beta2_power, 
                  int64 global_step,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstScalar epsilon_clipping,
                  typename TTypes<T>::ConstScalar weight_decay,
                  typename TTypes<T>::ConstScalar beta0,
                  typename TTypes<T>::ConstScalar beta_lookahead,
                  typename TTypes<T>::ConstScalar tau_clipping,
                  int64 kappa_lookahead,
                  bool use_adaptive_grad_clipping, bool use_grad_centralization,
                  bool use_positive_negative_momentum, bool use_norm_loss,
                  bool use_stable_weight_decay, bool use_lookahead,
                  typename TTypes<T>::ConstFlat grad);
};

template <typename Device, typename T, typename Tindex>
struct SparseApplyRanger21 {
  Status operator()(const Device& d, typename TTypes<T>::Matrix var,
                    typename TTypes<T>::Matrix m, typename TTypes<T>::Matrix v,
                    typename TTypes<T>::Matrix v_max,
                    typename TTypes<T>::Matrix lookahead_var,
                    typename TTypes<T>::Scalar beta1_power_scalar,
                    typename TTypes<T>::Scalar beta2_power_scalar,
                    int64 global_step,
                    typename TTypes<T>::ConstScalar lr_scalar,
                    typename TTypes<T>::ConstScalar beta1_scalar,
                    typename TTypes<T>::ConstScalar beta2_scalar,
                    typename TTypes<T>::ConstScalar epsilon_scalar,
                    typename TTypes<T>::ConstScalar epsilon_clipping_scalar,
                    typename TTypes<T>::ConstScalar weight_decay_scalar,
                    typename TTypes<T>::ConstScalar beta0_scalar,
                    typename TTypes<T>::ConstScalar beta_lookahead_scalar,
                    typename TTypes<T>::ConstScalar tau_clipping_scalar,
                    int64 kappa_lookahead_scalar,
                    bool use_adaptive_grad_clipping,
                    bool use_grad_centralization,
                    bool use_positive_negative_momentum, bool use_norm_loss,
                    bool use_stable_weight_decay, bool use_lookahead,
                    typename TTypes<T>::Matrix grad,
                    typename TTypes<Tindex>::ConstVec indices_vec,
                    int64 inner_dim);
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TRAINING_RANGER21_OPS_H_
