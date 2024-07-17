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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <float.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <fstream>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <nvtx3/nvToolsExt.h>
#include <thrust/host_vector.h>
#include <tuple>
#include "types.h"
#include "runtime_switching.h"

__global__ void markTargetNodes(Node* nodes, int N, int target, int* node_counts)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int count = 0;
	Node node = nodes[idx];
	if (node.depth > target)
		count = node.count_leafs;
	else if (node.parent != -1)
	{
		Node parentnode = nodes[node.parent];
		if (parentnode.depth > target)
		{
			count = node.count_leafs;
			if (node.depth != 0)
				count += node.count_merged;
		}
	}
	node_counts[idx] = count;
}

__global__ void putRenderIndices(Node* nodes, int N, int* node_counts, int* node_offsets, int* render_indices, int* parent_indices = nullptr, int* nodes_for_render_indices = nullptr)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	Node node = nodes[idx];
	int count = node_counts[idx];
	int offset = idx == 0 ? 0 : node_offsets[idx - 1];
	int start = node.start;
	
	int parentgaussian = -1;
	if (node.parent != -1)
	{
		parentgaussian = nodes[node.parent].start;
	}

	for (int i = 0; i < count; i++)
	{
		render_indices[offset + i] = node.start + i;
		if (parent_indices)
			parent_indices[offset + i] = parentgaussian; 
		if (nodes_for_render_indices)
			nodes_for_render_indices[offset + i] = idx;
	}
}

int Switching::expandToTarget(
	int N,
	int target,
	int* nodes,
	int* render_indices
)
{
	thrust::device_vector<int> render_counts(N);
	thrust::device_vector<int> render_offsets(N);

	int num_blocks = (N + 255) / 256;
	markTargetNodes << <num_blocks, 256 >> > ((Node*)nodes, N, target, render_counts.data().get());

	size_t temp_storage_bytes;
	thrust::device_vector<char> temp_storage;
	cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, render_counts.data().get(), render_offsets.data().get(), N);
	temp_storage.resize(temp_storage_bytes);
	cub::DeviceScan::InclusiveSum(temp_storage.data().get(), temp_storage_bytes, render_counts.data().get(), render_offsets.data().get(), N);

	putRenderIndices << <num_blocks, 256 >> > ((Node*)nodes, N, render_counts.data().get(), render_offsets.data().get(), render_indices);

	int count = 0;
	cudaMemcpy(&count, render_offsets.data().get() + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
	return count;
}

__device__ bool inboxCUDA(Box& box, Point viewpoint)
{
	bool inside = true;
	for (int i = 0; i < 3; i++)
	{
		inside &= viewpoint.xyz[i] >= box.minn.xyz[i] && viewpoint.xyz[i] <= box.maxx.xyz[i];
	}
	return inside;
}

__device__ float pointboxdistCUDA(Box& box, Point viewpoint)
{
	Point closest = {
		max(box.minn.xyz[0], min(box.maxx.xyz[0], viewpoint.xyz[0])),
		max(box.minn.xyz[1], min(box.maxx.xyz[1], viewpoint.xyz[1])),
		max(box.minn.xyz[2], min(box.maxx.xyz[2], viewpoint.xyz[2]))
	};

	Point diff = {
		viewpoint.xyz[0] - closest.xyz[0],
		viewpoint.xyz[1] - closest.xyz[1],
		viewpoint.xyz[2] - closest.xyz[2]
	};

	return sqrt(diff.xyz[0] * diff.xyz[0] + diff.xyz[1] * diff.xyz[1] + diff.xyz[2] * diff.xyz[2]);
}

__device__ float computeSizeGPU(Box& box, Point viewpoint, Point zdir)
{
	if (inboxCUDA(box, viewpoint))
		return FLT_MAX;

	float min_dist = pointboxdistCUDA(box, viewpoint);

	return box.minn.xyz[3] / min_dist;
}

__global__ void changeNodesOnce(
	Node* nodes,
	int N,
	int* indices,
	Box* boxes,
	Point* viewpoint,
	Point zdir,
	float target_size,
	int* split,
	int* node_counts,
	int* node_ids,
	char* needs_children
)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_id = indices[idx];
	Node node = nodes[node_id];
	float size = computeSizeGPU(boxes[node_id], *viewpoint, zdir);

	int count = 1; // repeat yourself
	char need_child = 0;
	if (size >= target_size)
	{
		if (node.depth > 0 && split[node_id] == 0) // split
		{
			if (node.start_children == -1)
			{
				node_ids[idx] = node_id;
				need_child = 1;
			}
			else
			{
				count += node.count_children;
				split[node_id] = 1;
			}
		}
	}
	else
	{
		int parent_node_id = node.parent;
		if (parent_node_id != -1)
		{
			Node parent_node = nodes[parent_node_id];
			float parent_size = computeSizeGPU(boxes[parent_node_id], *viewpoint, zdir);
			if (parent_size < target_size) // collapse
			{
				split[parent_node_id] = 0;
				count = 0; // forget yourself
			}
		}
	}
	needs_children[idx] = need_child;
	node_counts[idx] = count;
}

__global__ void putNodes(
	Node* nodes,
	int N,
	int* indices,
	int* node_counts,
	int* node_offsets,
	int* next_nodes)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int count = node_counts[idx];
	if (count == 0)
		return;

	int node_id = indices[idx];
	Node node = nodes[node_id];
	int offset = idx == 0 ? 0 : node_offsets[idx - 1];

	next_nodes[offset] = node_id;
	for (int i = 1; i < count; i++)
		next_nodes[offset + i] = node.start_children + i - 1;
}

__global__ void countRenderIndicesIndexed(Node* nodes, int* split, int N, int* node_indices, int* render_counts)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_idx = node_indices[idx];

	Node node = nodes[node_idx];
	int count = node.count_leafs;
	if (node.depth > 0 && split[node_idx] == 0)
		count += node.count_merged;

	render_counts[idx] = count;
}

__global__ void putRenderIndicesIndexed(Node* nodes, int N, int* node_indices, int* render_counts, int* render_offsets, int* render_indices, int* parent_indices, int* nodes_of_render_indices, Box* boxes, float3* debug)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_idx = node_indices[idx];

	Node node = nodes[node_idx];
	int count = render_counts[idx];
	int offset = idx == 0 ? 0 : render_offsets[idx - 1];
	int start = node.start;

	int parentgaussian = -1;
	if (node.parent != -1)
	{
		parentgaussian = nodes[node.parent].start;
	}

	for (int i = 0; i < count; i++)
	{
		render_indices[offset + i] = node.start + i;
		parent_indices[offset + i] = parentgaussian;
		nodes_of_render_indices[offset + i] = node_idx;
	}

	if (debug != nullptr)
	{
		Box box = boxes[node_idx];
		for (int i = 0; i < count; i++)
		{
			float red = min(1.0f, node.depth / 10.0f);
			debug[offset + i] = { red, 1.0f - red, 0 };
			if (node.depth == 0)
				debug[offset + i] = { 0, 0, 1.0f };
		}
	}
}

void Switching::changeToSizeStep(
	float target_size,
	int N,
	int* node_indices,
	int* new_node_indices,
	int* nodes,
	float* boxes,
	float* viewpoint,
	float x, float y, float z,
	int* split,
	int* render_indices,
	int* parent_indices,
	int* nodes_of_render_indices,
	int* nodes_to_expand,
	float* debug,
	char*& scratchspace,
	size_t& scratchspacesize,
	int* NsrcI,
	int* NdstI,
	char* NdstC,
	int* numI,
	int maxN,
	int& add_success,
	int* new_N,
	int* new_R,
	int* need_expansion,
	void* maintenanceStream)
{
	cudaStream_t stream = (cudaStream_t)maintenanceStream;

	int num_node_blocks = (N + 255) / 256;

	Point zdir = { x, y, z };

	int* num_to_expand = numI;
	int* node_counts = NsrcI, * node_offsets = NdstI, * node_ids = NdstI;
	char* need_children = NdstC;
	if (scratchspacesize == 0)
	{
		size_t testsize;

		cub::DeviceScan::InclusiveSum(nullptr, testsize, node_counts, node_offsets, maxN, stream);
		scratchspacesize = testsize;
		cub::DeviceSelect::Flagged(nullptr, testsize, node_ids, need_children, nodes_to_expand, num_to_expand, maxN, stream);
		scratchspacesize = std::max(testsize, scratchspacesize);

		if (scratchspace)
			cudaFree(scratchspace);
		scratchspacesize = testsize;
		cudaMalloc(&scratchspace, scratchspacesize);
	}

	changeNodesOnce << <num_node_blocks, 256, 0, stream >> > (
		(Node*)nodes, 
		N, 
		node_indices, 
		(Box*)boxes, 
		(Point*)viewpoint, 
		zdir, 
		target_size, 
		split, 
		node_counts, 
		node_ids, 
		need_children
		);

	cub::DeviceSelect::Flagged(scratchspace, scratchspacesize, node_ids, need_children, nodes_to_expand, num_to_expand, N, stream);
	cub::DeviceScan::InclusiveSum(scratchspace, scratchspacesize, node_counts, node_offsets, N, stream);

	cudaMemcpyAsync(need_expansion, num_to_expand, sizeof(int), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(new_N, node_offsets + N - 1, sizeof(int), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	if (*new_N > maxN)
	{
		add_success = 0;
		return;
	}

	putNodes << <num_node_blocks, 256, 0, stream>> > (
		(Node*)nodes,
		N, 
		node_indices, 
		node_counts, 
		node_offsets, 
		new_node_indices
		);

	int num_render_blocks = (*new_N + 255) / 256;
	int* render_counts = NsrcI, * render_offsets = NdstI;

	countRenderIndicesIndexed << <num_render_blocks, 256, 0, stream >> > (
		(Node*)nodes, 
		split, 
		*new_N, 
		new_node_indices, 
		render_counts
		);

	cub::DeviceScan::InclusiveSum(scratchspace, scratchspacesize, render_counts, render_offsets, *new_N, stream);

	putRenderIndicesIndexed << <num_render_blocks, 256, 0, stream >> > (
		(Node*)nodes, 
		*new_N, 
		new_node_indices, 
		render_counts, 
		render_offsets, 
		render_indices, 
		parent_indices, 
		nodes_of_render_indices, 
		(Box*)boxes,
		(float3*)debug
		);

	cudaMemcpyAsync(new_R, render_offsets + *new_N - 1, sizeof(int), cudaMemcpyDeviceToHost, stream);

	add_success = 1;
}

__global__ void markNodesForSize(Node* nodes, Box* boxes, int N, Point* viewpoint, Point zdir, float target_size, int* render_counts, int* node_markers)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_id = idx;
	Node node = nodes[node_id];
	float size = computeSizeGPU(boxes[node_id], *viewpoint, zdir);

	int count = 0;
	if (size >= target_size)
		count = node.count_leafs;
	else if (node.parent != -1)
	{
		float parent_size = computeSizeGPU(boxes[node.parent], *viewpoint, zdir);
		if (parent_size >= target_size)
		{
			count = node.count_leafs;
			if (node.depth != 0)
				count += node.count_merged;
		}
	}

	if (count != 0 && node_markers != nullptr)
		node_markers[node_id] = 1;

	if (render_counts != nullptr)
		render_counts[node_id] = count;
}

__global__ void computeTsIndexed(
	Node* nodes,
	Box* boxes,
	int N,
	int* indices,
	Point viewpoint,
	Point zdir,
	float target_size,
	float* ts,
	int* kids
)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_id = indices[idx];
	Node node = nodes[node_id];

	float t;
	if (node.parent == -1)
		t = 1.0f;
	else
	{
		float parentsize = computeSizeGPU(boxes[node.parent], viewpoint, zdir);

		if (parentsize > 2.0f * target_size)
			t = 1.0f;
		else
		{
			float size = computeSizeGPU(boxes[node_id], viewpoint, zdir);
			float start = max(0.5f * parentsize, size);
			float diff = parentsize - start;

			if (diff <= 0)
				t = 1.0f;
			else
			{
				float tdiff = max(0.0f, target_size - start);
				t = max(1.0f - (tdiff / diff), 0.0f);
			}
		}
	}

	ts[idx] = t;
	kids[idx] = (node.parent == -1) ? 1 : nodes[node.parent].count_children;
}

void Switching::getTsIndexed(
	int N,
	int* indices,
	float target_size,
	int* nodes,
	float* boxes,
	float vx, float vy, float vz,
	float x, float y, float z,
	float* ts,
	int* kids,
	void* stream
)
{
	Point zdir = { x, y, z };
	Point cam = { vx, vy, vz };
	int num_blocks = (N + 255) / 256;
	computeTsIndexed<<<num_blocks, 256, 0, (cudaStream_t)stream >>>(
		(Node*)nodes, 
		(Box*)boxes, 
		N, 
		indices, 
		cam,
		zdir, 
		target_size, 
		ts, 
		kids);
}

int Switching::expandToSize(
	int N,
	float target_size,
	int* nodes,
	float* boxes,
	float* viewpoint,
	float x, float y, float z,
	int* render_indices,
	int* node_markers,
	int* parent_indices,
	int* nodes_for_render_indices)
{
	size_t temp_storage_bytes;
	thrust::device_vector<char> temp_storage;
	thrust::device_vector<int> render_counts(N);
	thrust::device_vector<int> render_offsets(N);

	Point zdir = { x, y, z };

	int num_blocks = (N + 255) / 256;
	markNodesForSize << <num_blocks, 256 >> > ((Node*)nodes, (Box*)boxes, N, (Point*)viewpoint, zdir, target_size, render_counts.data().get(), node_markers);

	cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, render_counts.data().get(), render_offsets.data().get(), N);
	temp_storage.resize(temp_storage_bytes);
	cub::DeviceScan::InclusiveSum(temp_storage.data().get(), temp_storage_bytes, render_counts.data().get(), render_offsets.data().get(), N);

	putRenderIndices << <num_blocks, 256 >> > ((Node*)nodes, N, render_counts.data().get(), render_offsets.data().get(), render_indices, parent_indices, nodes_for_render_indices);

	int count = 0;
	cudaMemcpy(&count, render_offsets.data().get() + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
	return count;
}

void Switching::markVisibleForAllViewpoints(
	float target_size,
	int* nodes,
	int num_nodes,
	float* boxes,
	float* viewpoints,
	int num_viewpoints,
	int* seen,
	float zx,
	float zy,
	float zz
)
{
	thrust::device_vector<int> seen_cuda(num_nodes);
	thrust::device_vector<Point> viewpoint_cuda(1);
	thrust::device_vector<Node> nodes_cuda(num_nodes);
	thrust::device_vector<Box> boxes_cuda(num_nodes);

	cudaMemcpy(nodes_cuda.data().get(), nodes, sizeof(Node) * num_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(boxes_cuda.data().get(), boxes, sizeof(Box) * num_nodes, cudaMemcpyHostToDevice);

	Point zdir = { zx, zy, zz };

	Point* points = (Point*)viewpoints;
	int num_blocks = (num_nodes + 255) / 256;
	for (int i = 0; i < num_viewpoints; i++)
	{
		Point viewpoint = points[i];
		cudaMemcpy(viewpoint_cuda.data().get(), &viewpoint, sizeof(Point), cudaMemcpyHostToDevice);

		markNodesForSize << <num_blocks, 256 >> > (
			nodes_cuda.data().get(),
			boxes_cuda.data().get(),
			num_nodes,
			viewpoint_cuda.data().get(),
			zdir,
			target_size,
			nullptr,
			seen_cuda.data().get());
	}
	cudaMemcpy(seen, seen_cuda.data().get(), sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);

	if (cudaDeviceSynchronize())
		std::cout << "Errors: " << cudaDeviceSynchronize() << std::endl;
}