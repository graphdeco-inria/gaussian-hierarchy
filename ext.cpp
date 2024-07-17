/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "torch/torch_interface.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("load_hierarchy", &LoadHierarchy);
  m.def("write_hierarchy", &WriteHierarchy);
  m.def("expand_to_target", &ExpandToTarget);
  m.def("expand_to_size", &ExpandToSize);
  m.def("get_interpolation_weights", &GetTsIndexed);
}