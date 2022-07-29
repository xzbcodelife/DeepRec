/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

static ShapeHandle ShapeOrHandleShape(InferenceContext* c, int input) {
  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

// Handle the gradient and, if <sparse>, indices inputs.
// <s> is an input+output parameter, containing the current known input shape to
// the gradient.
static Status HandleGradAndIndicesInputs(InferenceContext* c, bool sparse,
                                         int grad_idx, ShapeHandle* s) {
  ShapeHandle grad = ShapeOrHandleShape(c, grad_idx);
  if (!sparse) {
    TF_RETURN_IF_ERROR(c->Merge(*s, grad, s));
    return Status::OK();
  }
  // Indices is a vector where indices.dim[0].rank == grad[0].rank.
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(grad_idx + 1), 1, &indices));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused));

  // Trailing part of grad matches trailing part of *s.
  ShapeHandle grad_unknown_first;
  TF_RETURN_IF_ERROR(
      c->ReplaceDim(grad, 0, c->UnknownDim(), &grad_unknown_first));
  TF_RETURN_IF_ERROR(c->Merge(*s, grad_unknown_first, s));

  return Status::OK();
}

static Status HandleKvGradAndIndicesInputs(InferenceContext* c, bool sparse,
                                           int grad_idx, ShapeHandle* s) {
  ShapeHandle grad = ShapeOrHandleShape(c, grad_idx);
  if (!sparse) {
    TF_RETURN_IF_ERROR(c->Merge(*s, grad, s));
    return Status::OK();
  }
  // Indices is a vector where indices.dim[0].rank == grad[0].rank.
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(grad_idx + 1), 1, &indices));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused));

  // Trailing part of grad matches trailing part of *s.
  ShapeHandle grad_unknown_first;
  TF_RETURN_IF_ERROR(c->Subshape(grad, 1, &grad_unknown_first));
  TF_RETURN_IF_ERROR(c->Merge(*s, grad_unknown_first, s));

  return Status::OK();
}

static Status ApplyRanger21ShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 3), &s));  // v_max
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape(c, 4), &s));             // lookahead_var
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));   // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));   // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));   // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 0, &unused));   // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(10), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(11), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(
      c->WithRank(c->input(12), 0, &unused));  // epsilon_clipping
  TF_RETURN_IF_ERROR(c->WithRank(c->input(13), 0, &unused));  // weight_decay
  TF_RETURN_IF_ERROR(c->WithRank(c->input(14), 0, &unused));  // beta0
  TF_RETURN_IF_ERROR(c->WithRank(c->input(15), 0, &unused));  // beta_lookahead
  TF_RETURN_IF_ERROR(c->WithRank(c->input(16), 0, &unused));  // tau_clipping
  TF_RETURN_IF_ERROR(c->WithRank(c->input(17), 0, &unused));  // kappa_lookahead
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 18 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyRanger21")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("v_max: Ref(T)")
    .Input("lookahead_var: Ref(T)")
    .Input("beta1_power: Ref(T)")
    .Input("beta2_power: Ref(T)")
    .Input("global_step: Tstep")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("epsilon_clipping: T")
    .Input("weight_decay: T")
    .Input("beta0: T")
    .Input("beta_lookahead: T")
    .Input("tau_clipping: T")
    .Input("kappa_lookahead: Tstep")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_adaptive_grad_clipping: bool = false")
    .Attr("use_grad_centralization: bool = false")
    .Attr("use_positive_negative_momentum: bool = false")
    .Attr("use_norm_loss: bool = false")
    .Attr("use_stable_weight_decay: bool = false")
    .Attr("use_lookahead: bool = false")
    .Attr("use_locking: bool = false")
    .Attr("Tstep: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyRanger21ShapeFn(c, false /* sparse */);
    });

REGISTER_OP("SparseApplyRanger21")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("v_max: Ref(T)")
    .Input("lookahead_var: Ref(T)")
    .Input("beta1_power: Ref(T)")
    .Input("beta2_power: Ref(T)")
    .Input("global_step: Tstep")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("epsilon_clipping: T")
    .Input("weight_decay: T")
    .Input("beta0: T")
    .Input("beta_lookahead: T")
    .Input("tau_clipping: T")
    .Input("kappa_lookahead: Tstep")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_adaptive_grad_clipping: bool = false")
    .Attr("use_grad_centralization: bool = false")
    .Attr("use_positive_negative_momentum: bool = false")
    .Attr("use_norm_loss: bool = false")
    .Attr("use_stable_weight_decay: bool = false")
    .Attr("use_lookahead: bool = false")
    .Attr("use_locking: bool = false")
    .Attr("Tindices: {int32, int64}")
    .Attr("Tstep: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyRanger21ShapeFn(c, true /* sparse */);
    });

REGISTER_OP("ResourceApplyRanger21")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("v_max: resource")
    .Input("lookahead_var: resource")
    .Input("beta1_power: resource")
    .Input("beta2_power: resource")
    .Input("global_step: Tstep")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("epsilon_clipping: T")
    .Input("weight_decay: T")
    .Input("beta0: T")
    .Input("beta_lookahead: T")
    .Input("tau_clipping: T")
    .Input("kappa_lookahead: Tstep")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_adaptive_grad_clipping: bool = false")
    .Attr("use_grad_centralization: bool = false")
    .Attr("use_positive_negative_momentum: bool = false")
    .Attr("use_norm_loss: bool = false")
    .Attr("use_stable_weight_decay: bool = false")
    .Attr("use_lookahead: bool = false")
    .Attr("use_locking: bool = false")
    .Attr("Tstep: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyRanger21ShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceSparseApplyRanger21")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("v_max: resource")
    .Input("lookahead_var: resource")
    .Input("beta1_power: resource")
    .Input("beta2_power: resource")
    .Input("global_step: Tstep")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("epsilon_clipping: T")
    .Input("weight_decay: T")
    .Input("beta0: T")
    .Input("beta_lookahead: T")
    .Input("tau_clipping: T")
    .Input("kappa_lookahead: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("use_adaptive_grad_clipping: bool = false")
    .Attr("use_grad_centralization: bool = false")
    .Attr("use_positive_negative_momentum: bool = false")
    .Attr("use_norm_loss: bool = false")
    .Attr("use_stable_weight_decay: bool = false")
    .Attr("use_lookahead: bool = false")
    .Attr("use_locking: bool = false")
    .Attr("Tindices: {int32, int64}")
    .Attr("Tstep: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyRanger21ShapeFn(c, true /* sparse */);
    });

static Status KvApplyRanger21ShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 3), &s));  // v_max
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape(c, 4), &s));             // lookahead_var
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));   // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));   // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));   // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 0, &unused));   // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(10), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(11), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(
      c->WithRank(c->input(12), 0, &unused));  // epsilon_clipping
  TF_RETURN_IF_ERROR(c->WithRank(c->input(13), 0, &unused));  // weight_decay
  TF_RETURN_IF_ERROR(c->WithRank(c->input(14), 0, &unused));  // beta0
  TF_RETURN_IF_ERROR(c->WithRank(c->input(15), 0, &unused));  // beta_lookahead
  TF_RETURN_IF_ERROR(c->WithRank(c->input(16), 0, &unused));  // tau_clipping
  TF_RETURN_IF_ERROR(c->WithRank(c->input(17), 0, &unused));  // kappa_lookahead
  TF_RETURN_IF_ERROR(
      HandleKvGradAndIndicesInputs(c, sparse, 18 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("KvResourceSparseApplyRanger21")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("v_max: resource")
    .Input("lookahead_var: resource")
    .Input("beta1_power: resource")
    .Input("beta2_power: resource")
    .Input("global_step: Tstep")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("epsilon_clipping: T")
    .Input("weight_decay: T")
    .Input("beta0: T")
    .Input("beta_lookahead: T")
    .Input("tau_clipping: T")
    .Input("kappa_lookahead: Tstep")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_adaptive_grad_clipping: bool = false")
    .Attr("use_grad_centralization: bool = false")
    .Attr("use_positive_negative_momentum: bool = false")
    .Attr("use_norm_loss: bool = false")
    .Attr("use_stable_weight_decay: bool = false")
    .Attr("use_lookahead: bool = false")
    .Attr("use_locking: bool = false")
    .Attr("Tstep: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      return KvApplyRanger21ShapeFn(c, true /* sparse */);
    });

}  // namespace tensorflow
