/*
 * Copyright (C) 2024, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <vector>
#include <Eigen/Dense>
#include "common.h"
#include <iostream>
#include <fstream>

class HierarchyWriter
{
public:
	void write(const char* filename,
		int allP, int allN,
		Eigen::Vector3f* positions,
		SHs* shs,
		float* opacities,
		Eigen::Vector3f* log_scales,
		Eigen::Vector4f* rotations,
		Node* nodes,
		Box* boxes,
		bool compressed = true);
};