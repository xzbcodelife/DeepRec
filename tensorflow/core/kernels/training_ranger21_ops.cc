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

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA
#include "tensorflow/core/kernels/training_ranger21_ops.h"

#include <algorithm>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#include "tensorflow/core/kernels/training_ali_op_helpers.h"
// #include "tensorflow/core/kernels/training_ali_ops.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/util/work_sharder.h"

#ifdef TENSORFLOW_USE_SYCL
#include "tensorflow/core/common_runtime/sycl/sycl_util.h"
#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
// #if TF_ENABLE_GPU_EV
// #include "tensorflow/core/kernels/training_ali_ops_gpu.h"
#include "tensorflow/core/kernels/training_ranger21_ops_gpu.h"
// #endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using SYCLDevice = Eigen::SyclDevice;

namespace functor {
template <typename T>
struct ApplyRanger21<CPUDevice, T> {

  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::Flat v_max,
                  typename TTypes<T>::Flat lookahead_var,
                  typename TTypes<T>::Scalar beta1_power,
                  typename TTypes<T>::Scalar beta2_power, int64 global_step,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstScalar epsilon_clipping,
                  typename TTypes<T>::ConstScalar weight_decay,
                  typename TTypes<T>::ConstScalar beta0,
                  typename TTypes<T>::ConstScalar beta_lookahead,
                  typename TTypes<T>::ConstScalar tau_clipping,
                  int64 kappa_lookahead, bool use_adaptive_grad_clipping,
                  bool use_grad_centralization,
                  bool use_positive_negative_momentum, bool use_norm_loss,
                  bool use_stable_weight_decay, bool use_lookahead,
                  typename TTypes<T>::ConstFlat grad) {

    // auto var_size = var.size();
    // auto m_size = m.size();
    // auto v_size = v.size();

    // std::cout << "var_size=" << var_size << std::endl;
    // std::cout << "m_size=" << m_size << std::endl;
    // std::cout << "v_size=" << v_size << std::endl;


    std::cout << "m=" << m.eval() << std::endl;
    std::cout << "v=" << v.eval() << std::endl;
    std::cout << "v_max=" << v_max.eval() << std::endl;
    std::cout << "var=" << var.eval() << std::endl;
    std::cout << "grad=" << grad.eval() << std::endl;
    std::cout << "beta1_power=" << beta1_power.eval() << std::endl;
    std::cout << "beta2_power=" << beta2_power.eval() << std::endl;


    // Gradient clipping
    // if (use_adaptive_grad_clipping) {
    //   auto clipping = grad.square().sum().sqrt() /
    //                   max(var.square().sum().sqrt(), epsilon_clipping);

    //   if (clipping > tau_clipping) {
    //     grad = (tau_clipping / clipping) * grad;
    //   }
    // }

    // Gradient centralization
    // if (use_grad_centralization) {
    //   auto grad_mean = grad.mean();
    //   grad = grad - grad_mean;
    // }



    // auto grad_dim = grad.dimensions();
    // std::cout << "grad_dim.size=" <<  grad_dim.size<< ", grad.size=" << grad.size() << std::endl;

    // typename TTypes<T>::Flat u(var.data(), var.size());
    if (use_positive_negative_momentum) {
      auto beta1_square = beta1() * beta1();

      // std::cout << "var=" << var(0) << "," << var(1) << std::endl;
      // std::cout << "beta1_square=" << beta1_square << std::endl;
      // std::cout << "m=" << m(0) << "," << m(1) << std::endl;

      // auto mm= m();
      // std::cout << "mm=" << mm[0] << std::endl;

      // auto m_old = m;
      auto m_new = m * beta1_square + grad * (T(1) - beta1_square);

      // std::cout << "m_new=" << m_new.eval() << std::endl;
      // std::cout << "m_new=" << m_new()(0) << std::endl;
      // std::cout << "m_new=" << m_new.flat<T>()(0) << "," << m_new(1) << std::endl;
      // std::cout << "m_old=" << m_old(0) << "," << m_old(1) << std::endl;

      auto m_bias_correction =
          ((T(1) + beta0()) * m_new - beta0() * m) / (T(1) - beta1_power());

      // std::cout << "m_bias_correction=" << m_bias_correction.eval() << std::endl;

      v.device(d) = v * beta2() + grad.square() * (T(1) - beta2());
      // std::cout << "v=" << v(0) << "," << v(1) << std::endl;

      v_max.device(d) = v_max.cwiseMax(v);
      // std::cout << "v_max=" << v_max(0) << "," << v_max(1) << std::endl;

      auto v_bias_correction = v_max / (T(1) - beta2_power());
      // std::cout << "v_bias_correction=" << v_bias_correction(0) << "," <<
      // v_bias_correction(1) << std::endl;
      // std::cout << "v_bias_correction=" << v_bias_correction.eval() << std::endl;


      auto u = m_bias_correction /
          (Eigen::numext::sqrt(Eigen::numext::pow(T(1) + beta0(), T(2)) +
                               Eigen::numext::pow(beta0(), T(2.0))) *
           (v_bias_correction.sqrt() + epsilon()));

      // std::cout << "u=" << u(0) << "," << u(1) << std::endl;
      std::cout << "u=" << u.eval() << std::endl;

      m.device(d) = m_new;
      // std::cout << "m=" << m(0) << "," << m(1) << std::endl;

      var.device(d) -= lr() * u;

    } else {
      m.device(d) = m * beta1() + grad * (T(1) - beta1());
      auto m_bias_correction = m / (T(1) - beta1_power());
      v.device(d) = v * beta2() + grad.square() * (T(1) - beta2());
      auto v_bias_correction = v / (T(1) - beta2_power());
      auto u = m_bias_correction / (v_bias_correction.sqrt() + epsilon());

      std::cout << "u=" << u.eval() << std::endl;

      var.device(d) -= lr() * u;
    }
    // std::cout << "var=" << var(0) << "," << var(1) << std::endl;


    // if (use_norm_loss) {
    //   auto weight_decay_update =
    //       lr() * weight_decay() * (T(1) - 1 / var.square().sum().sqrt()) *
    //       var;
    // } else if (use_stable_weight_decay) {
    //   auto weight_decay_update = lr() / v_bias_correction.sqrt() *
    //                              weight_decay() *
    //                              (T(1) - 1 / var.square().sum().sqrt()) *
    //                              var;
    // } else {
    auto weight_decay_update = lr() * weight_decay() * var;
    // }

    // std::cout << "var=" << var.eval() << std::endl;
    // var.device(d) -= (lr() * u + weight_decay_update);
    var.device(d) -= weight_decay_update;
    // std::cout << "var=" << var.eval() << std::endl;


    if (use_lookahead) {
      if (global_step % kappa_lookahead == 0) {
        lookahead_var.device(d) =
            beta_lookahead() * lookahead_var + (T(1) - beta_lookahead()) * var;
        var.device(d) = lookahead_var;
      }
    }

    // update beta1_power && beta2_power
    beta1_power.device(d) = beta1_power * beta1();
    beta2_power.device(d) = beta2_power * beta2();
  }
};
}  // namespace functor

template <typename Device, typename T>
class ApplyRanger21Op : public OpKernel {
 public:
  explicit ApplyRanger21Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_adaptive_grad_clipping",
                                     &use_adaptive_grad_clipping_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_grad_centralization",
                                     &use_grad_centralization_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_positive_negative_momentum",
                                     &use_positive_negative_momentum_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_norm_loss", &use_norm_loss_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_stable_weight_decay",
                                     &use_stable_weight_decay_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_lookahead", &use_lookahead_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2, 3, 4, 5, 6});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, false, &var));

    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, false, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, false, &v));

    Tensor v_max;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 3, use_exclusive_lock_, false, &v_max));

    Tensor lookahead_var;
    OP_REQUIRES_OK(
        ctx, GetInputTensorFromVariable<Device, T>(ctx, 4, use_exclusive_lock_,
                                                   false, &lookahead_var));

    Tensor beta1_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 5, use_exclusive_lock_, false, &beta1_power));
    Tensor beta2_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 6, use_exclusive_lock_, false, &beta2_power));

    const Tensor& global_step = ctx->input(7);

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));

    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));

    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));

    OP_REQUIRES(
        ctx, v_max.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(3)));

    OP_REQUIRES(
        ctx, lookahead_var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(4)));

    OP_REQUIRES(
        ctx, beta1_power.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(5)));

    OP_REQUIRES(
        ctx, beta2_power.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(6)));

    const Tensor& lr = ctx->input(8);
    const Tensor& beta1 = ctx->input(9);
    const Tensor& beta2 = ctx->input(10);
    const Tensor& epsilon = ctx->input(11);
    const Tensor& epsilon_clipping = ctx->input(12);
    const Tensor& weight_decay = ctx->input(13);
    const Tensor& beta0 = ctx->input(14);
    const Tensor& beta_lookahead = ctx->input(15);
    const Tensor& tau_clipping = ctx->input(16);
    const Tensor& kappa_lookahead = ctx->input(17);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    const Tensor& grad = ctx->input(18);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();

    functor::ApplyRanger21<Device, T>()(
        device, var.flat<T>(), m.flat<T>(), v.flat<T>(), v_max.flat<T>(),
        lookahead_var.flat<T>(), beta1_power.scalar<T>(),
        beta2_power.scalar<T>(), global_step.scalar<int64>()(), lr.scalar<T>(),
        beta1.scalar<T>(), beta2.scalar<T>(), epsilon.scalar<T>(),
        epsilon_clipping.scalar<T>(), weight_decay.scalar<T>(),
        beta0.scalar<T>(), beta_lookahead.scalar<T>(), tau_clipping.scalar<T>(),
        kappa_lookahead.scalar<int64>()(), use_adaptive_grad_clipping_,
        use_grad_centralization_, use_positive_negative_momentum_,
        use_norm_loss_, use_stable_weight_decay_, use_lookahead_,
        grad.flat<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_adaptive_grad_clipping_;
  bool use_grad_centralization_;
  bool use_positive_negative_momentum_;
  bool use_norm_loss_;
  bool use_stable_weight_decay_;
  bool use_lookahead_;
};

#define REGISTER_KERNELS(D, T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ApplyRanger21").Device(DEVICE_##D).TypeConstraint<T>("T"),         \
      ApplyRanger21Op<D##Device, T>);                                          \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ResourceApplyRanger21").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyRanger21Op<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS

// #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if GOOGLE_CUDA
// #if TF_ENABLE_GPU_EV
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                   \
  template <>                                                                 \
  void ApplyRanger21<GPUDevice, T>::operator()(                               \
      const GPUDevice& d, typename TTypes<T>::Flat var,                       \
      typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,                 \
      typename TTypes<T>::Flat v_max, typename TTypes<T>::Flat lookahead_var, \
      typename TTypes<T>::Scalar beta1_power,                                 \
      typename TTypes<T>::Scalar beta2_power, int64 global_step,              \
      typename TTypes<T>::ConstScalar lr,                                     \
      typename TTypes<T>::ConstScalar beta1,                                  \
      typename TTypes<T>::ConstScalar beta2,                                  \
      typename TTypes<T>::ConstScalar epsilon,                                \
      typename TTypes<T>::ConstScalar epsilon_clipping,                       \
      typename TTypes<T>::ConstScalar weight_decay,                           \
      typename TTypes<T>::ConstScalar beta0,                                  \
      typename TTypes<T>::ConstScalar beta_lookahead,                         \
      typename TTypes<T>::ConstScalar tau_clipping, int64 kappa_lookahead,    \
      bool use_adaptive_grad_clipping, bool use_grad_centralization,          \
      bool use_positive_negative_momentum, bool use_norm_loss,                \
      bool use_stable_weight_decay, bool use_lookahead,                       \
      typename TTypes<T>::ConstFlat grad);                                         \
  extern template struct ApplyRanger21<GPUDevice, T>;

DECLARE_GPU_SPEC(Eigen::half)
DECLARE_GPU_SPEC(float)
DECLARE_GPU_SPEC(double)
#undef DECLARE_GPU_SPEC
}  // end of namespace functor

#define REGISTER_GPU_KERNELS(T) REGISTER_KERNELS(GPU, T);

TF_CALL_half(REGISTER_GPU_KERNELS);
TF_CALL_float(REGISTER_GPU_KERNELS);
TF_CALL_double(REGISTER_GPU_KERNELS);

#undef REGISTER_GPU_KERNELS
// #endif  // TF_ENABLE_GPU_EV
#endif  // end of GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef REGISTER_KERNELS

namespace functor {
template <typename T, typename Tindex>
struct SparseApplyRanger21<CPUDevice, T, Tindex> {
  Status operator()(const CPUDevice& d, typename TTypes<T>::Matrix var,
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
                    int64 kappa_lookahead, bool use_adaptive_grad_clipping,
                    bool use_grad_centralization,
                    bool use_positive_negative_momentum, bool use_norm_loss,
                    bool use_stable_weight_decay, bool use_lookahead,
                    typename TTypes<T>::Matrix grad,
                    typename TTypes<Tindex>::ConstVec indices_vec,
                    int64 inner_dim) {
    const int64 N = indices_vec.dimension(0);
    if (N <= 0) return Status::OK();

    const T beta1_power = beta1_power_scalar();
    const T beta2_power = beta2_power_scalar();
    const T lr = lr_scalar();
    const T beta1 = beta1_scalar();
    const T beta2 = beta2_scalar();
    const T epsilon = epsilon_scalar();
    const T epsilon_clipping = epsilon_clipping_scalar();
    const T weight_decay = weight_decay_scalar();
    const T beta0 = beta0_scalar();
    const T beta_lookahead = beta_lookahead_scalar();
    const T tau_clipping = tau_clipping_scalar();

    const int64 first_dim_size = static_cast<int64>(var.dimension(0));

    // Validate all the indices are in range
    for (int64 i = 0; i < N; i++) {
      const Tindex index = indices_vec(i);
      if (index < 0 || index >= first_dim_size) {
        return errors::InvalidArgument(strings::StrCat(
            "Index ", index, " at offset ", i, " in indices is out of range"));
      }
    }

    auto DoWork = [this, &var, &m, &v, &grad, &v_max, &lookahead_var,
                   &indices_vec, &beta1_power, &beta2_power, &global_step, &lr,
                   &beta1, &beta2, &epsilon, &epsilon_clipping, &weight_decay,
                   &beta0, &beta_lookahead, &tau_clipping, &kappa_lookahead,
                   use_adaptive_grad_clipping, use_grad_centralization,
                   use_positive_negative_momentum, use_norm_loss,
                   use_stable_weight_decay, use_lookahead,
                   inner_dim](int64 start_i, int64 limit_i) {
      if (inner_dim > 1) {
        for (Tindex i = static_cast<Tindex>(start_i);
             i < static_cast<Tindex>(limit_i); i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));

          auto var_a = var.template chip<0>(index);
          auto m_a = m.template chip<0>(index);
          auto v_a = v.template chip<0>(index);
          auto v_max_a = v_max.template chip<0>(index);
          auto lookahead_var_a = lookahead_var.template chip<0>(index);
          auto g_i = grad.template chip<0>(i);

          // Gradient clipping
          // if (use_adaptive_grad_clipping) {
          //   auto clipping = g_i.square().sum().sqrt() /
          //                   max(var_a.square().sum().sqrt(),
          //                   epsilon_clipping);

          //   if (clipping > tau_clipping) {
          //     g_i = (tau_clipping / clipping) * g_i;
          //   }
          // }

          // Gradient centralization
          // if (use_grad_centralization) {
          //   auto grad_mean = g_i.mean();
          //   g_i = g_i - grad_mean;
          // }

          if (use_positive_negative_momentum) {
            auto beta1_square = beta1 * beta1;

            auto m_old = m_a;
            auto m_new =
                m_a * beta1_square + g_i * (static_cast<T>(1) - beta1_square);
            m_a = m_new;
            auto m_bias_correction =
                ((static_cast<T>(1) + beta0) * m_new - beta0 * m_old) /
                (static_cast<T>(1) - beta1_power);

            v_a = v_a * beta2 + g_i.square() * (static_cast<T>(1) - beta2);
            v_max_a = v_max_a.cwiseMax(v_a);
            auto v_bias_correction =
                v_max_a / (static_cast<T>(1) - beta2_power);
            auto u = m_bias_correction /
                     (Eigen::numext::sqrt(
                          Eigen::numext::pow(static_cast<T>(1) + beta0,
                                             static_cast<T>(2)) +
                          Eigen::numext::pow(beta0, static_cast<T>(2))) *
                      (v_bias_correction.sqrt() + epsilon));
            var_a -= lr * u;
          } else {
            m_a = m_a * beta1 + g_i * (static_cast<T>(1) - beta1);
            auto m_bias_correction = m_a / (static_cast<T>(1) - beta1_power);
            v_a = v_a * beta2 + g_i.square() * (static_cast<T>(1) - beta2);
            auto v_bias_correction = v_a / (static_cast<T>(1) - beta2_power);
            auto u = m_bias_correction / (v_bias_correction.sqrt() + epsilon);
            var_a -= lr * u;
          }

          // if (use_norm_loss) {
          // auto  weight_decay_update = lr( * weight_decay *
          //                         (static_cast<T>(1) - 1 /
          //                         var_i.square().sum().sqrt()) * var_a;

          // } else if (use_stable_weight_decay) {
          //   auto weight_decay_update =
          //       lr / v_bias_correction.sqrt() * weight_decay *
          //       (static_cast<T>(1) - 1 / var.square().sum().sqrt()) *
          //       var_a;
          // } else {
          auto weight_decay_update = lr * weight_decay * var_a;
          // }

          var_a -= weight_decay_update;

          if (use_lookahead) {
            if (global_step % kappa_lookahead == 0) {
              lookahead_var_a = beta_lookahead * lookahead_var_a +
                                (static_cast<T>(1) - beta_lookahead) * var_a;
              var_a = lookahead_var;
            }
          }
        }

      } else {
        for (Tindex i = static_cast<Tindex>(start_i);
             i < static_cast<Tindex>(limit_i); i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));

          T& var_a = var(index);
          T& m_a = m(index);
          T& v_a = v(index);
          T& v_max_a = v_max(index);
          T& lookahead_var_a = lookahead_var(index);
          T g_i = grad(i);

          // Gradient clipping
          // if (use_adaptive_grad_clipping) {
          //   auto clipping = g_i.square().sum().sqrt() /
          //                   max(var_a.square().sum().sqrt(),
          //                   epsilon_clipping);

          //   if (clipping > tau_clipping) {
          //     g_i = (tau_clipping / clipping) * g_i;
          //   }
          // }

          // // Gradient centralization
          // if (use_grad_centralization) {
          //   auto grad_mean = g_i.mean();
          //   g_i = g_i - grad_mean;
          // }

          if (use_positive_negative_momentum) {
            auto beta1_square = beta1 * beta1;

            auto m_old = m_a;
            auto m_new =
                m_a * beta1_square + g_i * (static_cast<T>(1) - beta1_square);
            m_a = m_new;
            auto m_bias_correction =
                ((static_cast<T>(1) + beta0) * m_new - beta0 * m_old) /
                (static_cast<T>(1) - beta1_power);

            v_a = v_a * beta2 + g_i * g_i * (static_cast<T>(1) - beta2);
            v_max_a = Eigen::numext::maxi(v_max_a, v_a);
            auto v_bias_correction =
                v_max_a / (static_cast<T>(1) - beta2_power);
            auto u = m_bias_correction /
                     (Eigen::numext::sqrt(
                          Eigen::numext::pow(static_cast<T>(1) + beta0,
                                             static_cast<T>(2)) +
                          Eigen::numext::pow(beta0, static_cast<T>(2))) *
                      (Eigen::numext::sqrt(v_bias_correction) + epsilon));
            var_a -= lr * u;

          } else {
            m_a = m_a * beta1 + g_i * (static_cast<T>(1) - beta1);
            auto m_bias_correction = m_a / (static_cast<T>(1) - beta1_power);
            v_a = v_a * beta2 + g_i * g_i * (static_cast<T>(1) - beta2);
            auto v_bias_correction = v_a / (static_cast<T>(1) - beta2_power);
            auto u = m_bias_correction /
                     (Eigen::numext::sqrt(v_bias_correction) + epsilon);
            var_a -= lr * u;
          }

          // if (use_norm_loss) {
          //  auto  weight_decay_update = lr( * weight_decay *
          //                         (static_cast<T>(1) - 1 /
          //                         var_i.square().sum().sqrt()) * var_a;

          // } else if (use_stable_weight_decay) {
          //   auto weight_decay_update =
          //       lr / v_bias_correction.sqrt() * weight_decay *
          //       (static_cast<T>(1) - 1 / var.square().sum().sqrt()) *
          //       var_a;
          // } else {
          auto weight_decay_update = lr * weight_decay * var_a;
          // }

          var_a -= weight_decay_update;

          if (use_lookahead) {
            if (global_step % kappa_lookahead == 0) {
              lookahead_var_a = beta_lookahead * lookahead_var_a +
                                (static_cast<T>(1) - beta_lookahead) * var_a;
              var_a = lookahead_var_a;
            }
          }
        }
      }
    };

    const int in_bytes = inner_dim * sizeof(T) * 4;
    const int out_bytes = inner_dim * sizeof(T) * 3;
    const int cycles = inner_dim * (Eigen::TensorOpCost::AddCost<T>() * 6 +
                                    Eigen::TensorOpCost::MulCost<T>() * 6 +
                                    Eigen::TensorOpCost::DivCost<T>());
    const Eigen::TensorOpCost cost(in_bytes, out_bytes, cycles);

    d.parallelFor(N, cost, DoWork);

    beta1_power_scalar() *= beta1;
    beta2_power_scalar() *= beta2;

    return Status::OK();
  }
};
}  // namespace functor

template <typename Device, typename T, typename Tindex>
class SparseApplyRanger21Op : public OpKernel {
 public:
  explicit SparseApplyRanger21Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_adaptive_grad_clipping",
                                     &use_adaptive_grad_clipping_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_grad_centralization",
                                     &use_grad_centralization_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_positive_negative_momentum",
                                     &use_positive_negative_momentum_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_norm_loss", &use_norm_loss_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_stable_weight_decay",
                                     &use_stable_weight_decay_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_lookahead", &use_lookahead_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2, 3, 4, 5, 6});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, false, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, false, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, false, &v));

    Tensor v_max;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 3, use_exclusive_lock_, false, &v_max));

    Tensor lookahead_var;
    OP_REQUIRES_OK(
        ctx, GetInputTensorFromVariable<Device, T>(ctx, 4, use_exclusive_lock_,
                                                   false, &lookahead_var));

    Tensor beta1_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 5, use_exclusive_lock_, false, &beta1_power));
    Tensor beta2_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 6, use_exclusive_lock_, false, &beta2_power));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));

    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));

    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));

    OP_REQUIRES(
        ctx, v_max.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(3)));

    OP_REQUIRES(
        ctx, lookahead_var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(4)));

    OP_REQUIRES(
        ctx, beta1_power.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(5)));

    OP_REQUIRES(
        ctx, beta2_power.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(6)));

    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& global_step = ctx->input(7);

    const Tensor& lr = ctx->input(8);
    const Tensor& beta1 = ctx->input(9);
    const Tensor& beta2 = ctx->input(10);
    const Tensor& epsilon = ctx->input(11);
    const Tensor& epsilon_clipping = ctx->input(12);
    const Tensor& weight_decay = ctx->input(13);
    const Tensor& beta0 = ctx->input(14);
    const Tensor& beta_lookahead = ctx->input(15);
    const Tensor& tau_clipping = ctx->input(16);
    const Tensor& kappa_lookahead = ctx->input(17);

    Tensor grad = ctx->input(18);
    const Tensor& indices = ctx->input(19);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const int64 N = indices.dim_size(0);
    OP_REQUIRES(ctx, grad.dim_size(0) == N,
                errors::InvalidArgument("grad must be the same size as indices "
                                        "in the first dimension."));

    const Device& device = ctx->template eigen_device<Device>();
    OP_REQUIRES_OK(
        ctx, functor::SparseApplyRanger21<Device, T, Tindex>()(
                 device, var.flat_outer_dims<T>(), m.flat_outer_dims<T>(),
                 v.flat_outer_dims<T>(), v_max.flat_outer_dims<T>(),
                 lookahead_var.flat_outer_dims<T>(), beta1_power.scalar<T>(),
                 beta2_power.scalar<T>(), global_step.scalar<int64>()(),
                 lr.scalar<T>(), beta1.scalar<T>(), beta2.scalar<T>(),
                 epsilon.scalar<T>(), epsilon_clipping.scalar<T>(),
                 weight_decay.scalar<T>(), beta0.scalar<T>(),
                 beta_lookahead.scalar<T>(), tau_clipping.scalar<T>(),
                 kappa_lookahead.scalar<int64>()(), use_adaptive_grad_clipping_,
                 use_grad_centralization_, use_positive_negative_momentum_,
                 use_norm_loss_, use_stable_weight_decay_, use_lookahead_,
                 grad.flat_outer_dims<T>(), indices.vec<Tindex>(), inner_dim));

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_adaptive_grad_clipping_;
  bool use_grad_centralization_;
  bool use_positive_negative_momentum_;
  bool use_norm_loss_;
  bool use_stable_weight_decay_;
  bool use_lookahead_;
};

#define REGISTER_KERNELS(D, T, Tindices)                                  \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyRanger21")                     \
                              .Device(DEVICE_##D)                         \
                              .TypeConstraint<T>("T")                     \
                              .TypeConstraint<Tindices>("Tindices"),      \
                          SparseApplyRanger21Op<D##Device, T, Tindices>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyRanger21")             \
                              .Device(DEVICE_##D)                         \
                              .TypeConstraint<T>("T")                     \
                              .TypeConstraint<Tindices>("Tindices"),      \
                          SparseApplyRanger21Op<D##Device, T, Tindices>);
#define REGISTER_CPU_KERNELS(T)    \
  REGISTER_KERNELS(CPU, T, int32); \
  REGISTER_KERNELS(CPU, T, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS

// #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if GOOGLE_CUDA
// #if TF_ENABLE_GPU_EV
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                                        \
  template <>                                                              \
  Status SparseApplyRanger21<GPUDevice, T, Tindex>::operator()(            \
      const GPUDevice& d, typename TTypes<T>::Matrix var,                  \
      typename TTypes<T>::Matrix m, typename TTypes<T>::Matrix v,          \
      typename TTypes<T>::Matrix v_max,                                    \
      typename TTypes<T>::Matrix lookahead_var,                            \
      typename TTypes<T>::Scalar beta1_power,                              \
      typename TTypes<T>::Scalar beta2_power, int64 global_step,           \
      typename TTypes<T>::ConstScalar lr,                                  \
      typename TTypes<T>::ConstScalar beta1,                               \
      typename TTypes<T>::ConstScalar beta2,                               \
      typename TTypes<T>::ConstScalar epsilon,                             \
      typename TTypes<T>::ConstScalar epsilon_clipping,                    \
      typename TTypes<T>::ConstScalar weight_decay,                        \
      typename TTypes<T>::ConstScalar beta0,                               \
      typename TTypes<T>::ConstScalar beta_lookahead,                      \
      typename TTypes<T>::ConstScalar tau_clipping, int64 kappa_lookahead, \
      bool use_adaptive_grad_clipping, bool use_grad_centralization,       \
      bool use_positive_negative_momentum, bool use_norm_loss,             \
      bool use_stable_weight_decay, bool use_lookahead,                    \
      typename TTypes<T>::Matrix grad,                                     \
      typename TTypes<Tindex>::ConstVec indices_vec, int64 inner_dim);     \
  extern template struct SparseApplyRanger21<GPUDevice, T, Tindex>;

DECLARE_GPU_SPEC(Eigen::half, int32);
DECLARE_GPU_SPEC(Eigen::half, int64);
DECLARE_GPU_SPEC(float, int32);
DECLARE_GPU_SPEC(float, int64);
DECLARE_GPU_SPEC(double, int32);
DECLARE_GPU_SPEC(double, int64);
#undef DECLARE_GPU_SPEC
}  // end of namespace functor

#define REGISTER_GPU_KERNEL(T)     \
  REGISTER_KERNELS(GPU, T, int32); \
  REGISTER_KERNELS(GPU, T, int64);

TF_CALL_half(REGISTER_GPU_KERNEL);
TF_CALL_float(REGISTER_GPU_KERNEL);
TF_CALL_double(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL
// #endif  // TF_ENABLE_GPU_EV
#endif  // End of #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex, typename Tstep>
class KvSparseApplyRanger21Op : public OpKernel {
 public:
  explicit KvSparseApplyRanger21Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_adaptive_grad_clipping",
                                     &use_adaptive_grad_clipping_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_grad_centralization",
                                     &use_grad_centralization_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_positive_negative_momentum",
                                     &use_positive_negative_momentum_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_norm_loss", &use_norm_loss_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_stable_weight_decay",
                                     &use_stable_weight_decay_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_lookahead", &use_lookahead_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<Tindex, T>(
        ctx, use_exclusive_lock_, {0, 1, 2, 3, 4, 5, 6});

    EmbeddingVar<Tindex, T>* var = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var));
    core::ScopedUnref unref_var(var);

    EmbeddingVar<Tindex, T>* m = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 1, &m));
    core::ScopedUnref unref_m(m);

    EmbeddingVar<Tindex, T>* v = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 2, &v));
    core::ScopedUnref unref_v(v);

    EmbeddingVar<Tindex, T>* v_max = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 3, &v_max));
    core::ScopedUnref unref_v_max(v_max);

    EmbeddingVar<Tindex, T>* lookahead_var = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 4, &lookahead_var));
    core::ScopedUnref unref_lookahead_var(lookahead_var);

    Tensor beta1_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 5, use_exclusive_lock_, true, &beta1_power));

    Tensor beta2_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 6, use_exclusive_lock_, true, &beta2_power));

    OP_REQUIRES(
        ctx, beta1_power.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(5)));
    OP_REQUIRES(
        ctx, beta2_power.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(6)));

    const Tensor& global_step = ctx->input(7);

    const Tensor& lr = ctx->input(8);
    const Tensor& beta1 = ctx->input(9);
    const Tensor& beta2 = ctx->input(10);
    const Tensor& epsilon = ctx->input(11);
    const Tensor& epsilon_clipping = ctx->input(12);
    const Tensor& weight_decay = ctx->input(13);
    const Tensor& beta0 = ctx->input(14);
    const Tensor& beta_lookahead = ctx->input(15);
    const Tensor& tau_clipping = ctx->input(16);
    const Tensor& kappa_lookahead = ctx->input(17);

    const Tensor& grad = ctx->input(18);
    const Tensor& indices = ctx->input(19);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    TensorShape var_shape({var->ValueLen()});
    for (int d = 0; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d + 1),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d + 1)));
      inner_dim *= grad.dim_size(d + 1);
    }
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    OP_REQUIRES(ctx, IsLegacyScalar(global_step.shape()),
                errors::InvalidArgument("global_step is not a scalar: ",
                                        global_step.shape().DebugString()));

    const int64 N = indices.dim_size(0);
    OP_REQUIRES(ctx, grad.dim_size(0) == N,
                errors::InvalidArgument("grad must be the same size as indices "
                                        "in the first dimension."));

    if (N > 0) {
      T lr_scalar = lr.scalar<T>()();
      T beta1_scalar = beta1.scalar<T>()();
      T beta2_scalar = beta2.scalar<T>()();
      T epsilon_scalar = epsilon.scalar<T>()();

      auto beta1_power_scalar = beta1_power.scalar<T>()();
      auto beta2_power_scalar = beta2_power.scalar<T>()();

      T epsilon_clipping_scalar = epsilon_clipping.scalar<T>()();
      T weight_decay_scalar = weight_decay.scalar<T>()();
      T beta0_scalar = beta0.scalar<T>()();
      T beta_lookahead_scalar = beta_lookahead.scalar<T>()();
      T tau_clipping_scalar = tau_clipping.scalar<T>()();
      int64 kappa_lookahead_scalar = kappa_lookahead.scalar<int64>()();

      /*

      use_adaptive_grad_clipping_,
                            use_grad_centralization_,
      use_positive_negative_momentum_, use_norm_loss_,
      use_stable_weight_decay_, use_lookahead_
      */

      bool use_adaptive_grad_clipping = use_adaptive_grad_clipping_;
      bool use_grad_centralization = use_grad_centralization_;
      bool use_positive_negative_momentum = use_positive_negative_momentum_;
      bool use_norm_loss = use_norm_loss_;
      bool use_stable_weight_decay = use_stable_weight_decay_;
      bool use_lookahead = use_lookahead_;

      auto do_work = [this, ctx, inner_dim, &indices, &var, &m, &v, &grad,
                      &v_max, &lookahead_var, &beta1_power_scalar,
                      &beta2_power_scalar, &global_step, &lr_scalar,
                      &beta1_scalar, &beta2_scalar, &epsilon_scalar,
                      &epsilon_clipping_scalar, &weight_decay_scalar,
                      &beta0_scalar, &beta_lookahead_scalar,
                      &tau_clipping_scalar, &kappa_lookahead_scalar,
                      use_adaptive_grad_clipping, use_grad_centralization,
                      use_positive_negative_momentum, use_norm_loss,
                      use_stable_weight_decay,
                      use_lookahead](int64 start_i, int64 limit_i) {
        if (inner_dim > 0) {
          int64 gs = global_step.scalar<int64>()();
          Tstep gs_step = global_step.scalar<Tstep>()();
          auto grad_flat = grad.flat_outer_dims<T>();
          auto indices_vec = indices.vec<Tindex>();

          for (int64 i = start_i; i < limit_i; i++) {
            const Tindex index = indices_vec(i);
            ValuePtr<T>* value_ptr = nullptr;
            bool is_filter = false;
            OP_REQUIRES_OK(
                ctx, var->LookupOrCreateKey(index, &value_ptr, &is_filter));
            var->UpdateVersion(value_ptr, gs_step);

            if (is_filter) {
              auto m_a = m->flat(value_ptr);
              auto v_a = v->flat(value_ptr);
              auto v_max_a = v_max->flat(value_ptr);
              auto lookahead_var_a = lookahead_var->flat(value_ptr);
              auto var_a = var->flat(value_ptr);
              auto g_i = grad_flat.template chip<0>(i);

              //       // Gradient clipping
              //       // if (use_adaptive_grad_clipping_) {
              //       //   auto clipping =
              //       //       g_i.square().sum().sqrt() /
              //       //       max(var_a.square().sum().sqrt(),
              //       //       epsilon_clipping_scalar);

              //       //   if (clipping > tau_clipping_scalar) {
              //       //     g_i = (tau_clipping_scalar / clipping) * g_i;
              //       //   }
              //       // }

              //       // // Gradient centralization
              //       // if (use_grad_centralization_) {
              //       //   auto grad_mean = g_i.mean();
              //       //   g_i = g_i - grad_mean;
              //       // }

              if (use_positive_negative_momentum_) {
                auto beta1_square = beta1_scalar * beta1_scalar;

                auto m_old = m_a;
                auto m_new = m_a * beta1_square +
                             g_i * (static_cast<T>(1) - beta1_square);

                m_a = m_new;
                auto m_bias_correction =
                    ((static_cast<T>(1) + beta0_scalar) * m_new -
                     beta0_scalar * m_old) /
                    (static_cast<T>(1) - beta1_power_scalar);

                v_a = v_a * beta2_scalar +
                      g_i.square() * (static_cast<T>(1) - beta2_scalar);

                v_max_a = v_max_a.cwiseMax(v_a);

                auto v_bias_correction =
                    v_max_a / (static_cast<T>(1) - beta2_power_scalar);

                auto u =
                    m_bias_correction /
                    (Eigen::numext::sqrt(
                         Eigen::numext::pow(static_cast<T>(1) + beta0_scalar,
                                            static_cast<T>(2)) +
                         Eigen::numext::pow(beta0_scalar, static_cast<T>(2))) *
                     (v_bias_correction.sqrt() + epsilon_scalar));
                var_a -= lr_scalar * u;

              } else {
                m_a = m_a * beta1_scalar +
                      g_i * (static_cast<T>(1) - beta1_scalar);
                auto m_bias_correction =
                    m_a / (static_cast<T>(1) - beta1_power_scalar);

                v_a = v_a * beta2_scalar +
                      g_i.square() * (static_cast<T>(1) - beta2_scalar);

                auto v_bias_correction =
                    v_a / (static_cast<T>(1) - beta2_power_scalar);

                auto u = m_bias_correction /
                         (v_bias_correction.sqrt() + epsilon_scalar);

                var_a -= lr_scalar * u;
              }

              // if (use_norm_loss_) {
              //       //   auto weight_decay_update =
              //       //       lr * weight_decay_scalar *
              //       //       (static_cast<T>(1) - 1 /
              //       //       var_i.square().sum().sqrt()) * var_i;

              //       // } else if (use_stable_weight_decay_) {
              //       //   auto weight_decay_update =
              //       //       lr_scalar / v_bias_correction.sqrt() *
              //       //       weight_decay_scalar * (static_cast<T>(1) - 1
              //       /
              //       //       var.square().sum().sqrt()) * var_a;
              //       // } else {
              auto weight_decay_update =
                  lr_scalar * weight_decay_scalar * var_a;
              //       // }

              var_a -= weight_decay_update;

              if (use_lookahead_) {
                if (gs % kappa_lookahead_scalar == 0) {
                  auto new_value =
                      beta_lookahead_scalar * lookahead_var_a +
                      (static_cast<T>(1) - beta_lookahead_scalar) * var_a;

                  lookahead_var_a = new_value;
                  var_a = new_value;
                }
              }

              var->Commit(index, value_ptr);
            }
          }
        }
      };

      const int64 cost = 1000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            do_work);

      beta1_power_scalar *= beta1_scalar;
      beta2_power_scalar *= beta2_scalar;
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_adaptive_grad_clipping_;
  bool use_grad_centralization_;
  bool use_positive_negative_momentum_;
  bool use_norm_loss_;
  bool use_stable_weight_decay_;
  bool use_lookahead_;
};

#define REGISTER_KERNELS(D, T, Tindices, Tstep) \
  REGISTER_KERNEL_BUILDER(                      \
      Name("KvResourceSparseApplyRanger21")     \
          .Device(DEVICE_##D)                   \
          .TypeConstraint<T>("T")               \
          .TypeConstraint<Tindices>("Tindices") \
          .TypeConstraint<Tstep>("Tstep"),      \
      KvSparseApplyRanger21Op<D##Device, T, Tindices, Tstep>);
#define REGISTER_CPU_KERNELS(T)           \
  REGISTER_KERNELS(CPU, T, int32, int32); \
  REGISTER_KERNELS(CPU, T, int64, int32); \
  REGISTER_KERNELS(CPU, T, int32, int64); \
  REGISTER_KERNELS(CPU, T, int64, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#if TF_ENABLE_GPU_EV
template <typename Device, typename T, typename Tindex, typename Tstep>
class KvSparseApplyRanger21OpGPU : public OpKernel {
 public:
  explicit KvSparseApplyRanger21OpGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_adaptive_grad_clipping",
                                     &use_adaptive_grad_clipping_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_grad_centralization",
                                     &use_grad_centralization_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_positive_negative_momentum",
                                     &use_positive_negative_momentum_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_norm_loss", &use_norm_loss_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_stable_weight_decay",
                                     &use_stable_weight_decay_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_lookahead", &use_lookahead_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<Tindex, T>(
        ctx, use_exclusive_lock_, {0, 1, 2, 3, 4, 5, 6});
    EmbeddingVar<Tindex, T>* var = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var));
    core::ScopedUnref unref_var(var);

    EmbeddingVar<Tindex, T>* m = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 1, &m));
    core::ScopedUnref unref_m(m);

    EmbeddingVar<Tindex, T>* v = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 2, &v));
    core::ScopedUnref unref_v(v);

    EmbeddingVar<Tindex, T>* v_max = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 3, &v_max));
    core::ScopedUnref unref_v_max(v_max);

    EmbeddingVar<Tindex, T>* lookahead_var = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 4, &lookahead_var));
    core::ScopedUnref unref_lookahead_var(lookahead_var);

    Tensor beta1_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 5, use_exclusive_lock_, true, &beta1_power));

    Tensor beta2_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 6, use_exclusive_lock_, true, &beta2_power));
    OP_REQUIRES(
        ctx, beta1_power.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(5)));
    OP_REQUIRES(
        ctx, beta2_power.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(6)));

    const Tensor& global_step = ctx->input(7);

    const Tensor& lr = ctx->input(8);
    const Tensor& beta1 = ctx->input(9);
    const Tensor& beta2 = ctx->input(10);
    const Tensor& epsilon = ctx->input(11);
    const Tensor& epsilon_clipping = ctx->input(12);
    const Tensor& weight_decay = ctx->input(13);
    const Tensor& beta0 = ctx->input(14);
    const Tensor& beta_lookahead = ctx->input(15);
    const Tensor& tau_clipping = ctx->input(16);
    const Tensor& kappa_lookahead = ctx->input(17);

    Tensor grad = ctx->input(18);
    const Tensor& indices = ctx->input(19);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    TensorShape var_shape({var->ValueLen()});
    for (int d = 0; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d + 1),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d + 1)));
      inner_dim *= grad.dim_size(d + 1);
    }
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    OP_REQUIRES(ctx, IsLegacyScalar(global_step.shape()),
                errors::InvalidArgument("global_step is not a scalar: ",
                                        global_step.shape().DebugString()));

    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(ctx, grad.dim_size(0) == N,
                errors::InvalidArgument("grad must be the same size as indices "
                                        "in the first dimension."));

    const Device& device = ctx->eigen_device<Device>();
    OP_REQUIRES_OK(
        ctx, functor::KvSparseApplyRanger21<Device, T, Tindex, Tstep>()(
                 device, var, m, v, v_max, lookahead_var,
                 global_step.scalar<int64>()(), beta1_power.scalar<T>(),
                 beta2_power.scalar<T>(), lr.scalar<T>(), beta1.scalar<T>(),
                 beta2.scalar<T>(), epsilon.scalar<T>(),
                 epsilon_clipping.scalar<T>(), weight_decay.scalar<T>(),
                 beta0.scalar<T>(), beta_lookahead.scalar<T>(),
                 tau_clipping.scalar<T>(), kappa_lookahead.scalar<int64>()(),
                 use_adaptive_grad_clipping_, use_grad_centralization_,
                 use_positive_negative_momentum_, use_norm_loss_,
                 use_stable_weight_decay_, use_lookahead_,
                 indices.vec<Tindex>(), grad.flat_outer_dims<T>(), inner_dim,
                 ctx->get_allocator(AllocatorAttributes())));

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_adaptive_grad_clipping_;
  bool use_grad_centralization_;
  bool use_positive_negative_momentum_;
  bool use_norm_loss_;
  bool use_stable_weight_decay_;
  bool use_lookahead_;
};

#define REGISTER_KERNELS(D, T, Tindices, Tstep) \
  REGISTER_KERNEL_BUILDER(                      \
      Name("KvResourceSparseApplyRanger21")     \
          .Device(DEVICE_##D)                   \
          .HostMemory("global_step")            \
          .TypeConstraint<T>("T")               \
          .TypeConstraint<Tindices>("Tindices") \
          .TypeConstraint<Tstep>("Tstep"),      \
      KvSparseApplyRanger21OpGPU<D##Device, T, Tindices, Tstep>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex, Tstep)                                  \
  template <>                                                               \
  Status KvSparseApplyRanger21<GPUDevice, T, Tindex, Tstep>::operator()(    \
      const GPUDevice& d, EmbeddingVarGPU<Tindex, T>* var,                  \
      EmbeddingVarGPU<Tindex, T>* m, EmbeddingVarGPU<Tindex, T>* v,         \
      EmbeddingVarGPU<Tindex, T>* v_max,                                    \
      EmbeddingVarGPU<Tindex, T>* lookahead_var,                            \
      typename TTypes<T>::Scalar beta1_power,                               \
      typename TTypes<T>::Scalar beta2_power, int64 global_step,            \
      typename TTypes<T>::ConstScalar lr,                                   \
      typename TTypes<T>::ConstScalar beta1,                                \
      typename TTypes<T>::ConstScalar beta2,                                \
      typename TTypes<T>::ConstScalar epsilon,                              \
      typename TTypes<T>::ConstScalar epsilon_clipping,                     \
      typename TTypes<T>::ConstScalar weight_decay,                         \
      typename TTypes<T>::ConstScalar beta0,                                \
      typename TTypes<T>::ConstScalar beta_lookahead,                       \
      typename TTypes<T>::ConstScalar tau_clipping, int64 kappa_lookahead,  \
      bool use_adaptive_grad_clipping, bool use_grad_centralization,        \
      bool use_positive_negative_momentum, bool use_norm_loss,              \
      bool use_stable_weight_decay, bool use_lookahead,                     \
      typename TTypes<T>::ConstMatrix grad,                                 \
      typename TTypes<Tindex>::ConstVec indices_vec, const int64 inner_dim, \
      Allocator* alloc);                                                    \
  extern template struct KvSparseApplyRanger21<GPUDevice, T, Tindex, Tstep>;

#define DECLARE_GPU_SPEC_TYPE(T)     \
  DECLARE_GPU_SPEC(T, int32, int32); \
  DECLARE_GPU_SPEC(T, int32, int64); \
  DECLARE_GPU_SPEC(T, int64, int32); \
  DECLARE_GPU_SPEC(T, int64, int64);

DECLARE_GPU_SPEC_TYPE(float);
DECLARE_GPU_SPEC_TYPE(double);

#undef DECLARE_GPU_SPEC_TYPE
#undef DECLARE_GPU_SPEC
}  // end of namespace functor

#define REGISTER_GPU_KERNEL(T)            \
  REGISTER_KERNELS(GPU, T, int32, int32); \
  REGISTER_KERNELS(GPU, T, int32, int64); \
  REGISTER_KERNELS(GPU, T, int64, int32); \
  REGISTER_KERNELS(GPU, T, int64, int64);

TF_CALL_float(REGISTER_GPU_KERNEL);
TF_CALL_double(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL
#endif  // End of GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef REGISTER_KERNELS

#endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
