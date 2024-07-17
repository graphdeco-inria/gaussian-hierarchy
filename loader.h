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

#include "common.h"
#include <vector>

class Loader
{
public:
	static void loadPlyDir(const char* filename, std::vector<Gaussian>& gaussian);

	static void loadPly(const char* filename, std::vector<Gaussian>& gaussian, int skyboxpoints = 0);

	static void loadBin(const char* filename, std::vector<Gaussian>& gaussian, int skyboxpoints = 0);
};