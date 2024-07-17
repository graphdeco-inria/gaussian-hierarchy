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


#include "writer.h"
#include <iostream>
#include <fstream>
#include "hierarchy_writer.h"
#include <map>

void populateRec(
	const ExplicitTreeNode* treenode,
	int id,
	const std::vector<Gaussian>& gaussians, 
	std::vector<Eigen::Vector3f>& positions,
	std::vector<Eigen::Vector4f>& rotations,
	std::vector<Eigen::Vector3f>& log_scales,
	std::vector<float>& opacities,
	std::vector<SHs>& shs,
	std::vector<Node>& basenodes,
	std::vector<Box>& boxes,
	std::map<int, const ExplicitTreeNode*>* base2tree = nullptr)
{
	if(base2tree)
		base2tree->insert(std::make_pair(id, treenode));

	boxes[id] = treenode->bounds;
	basenodes[id].start = positions.size();
	for (auto& i : treenode->leaf_indices)
	{
		const Gaussian& g = gaussians[i];
		positions.push_back(g.position);
		rotations.push_back(g.rotation);
		log_scales.push_back(g.scale.array().log());
		opacities.push_back(g.opacity);
		shs.push_back(g.shs);
	}
	basenodes[id].count_leafs = treenode->leaf_indices.size();

	for (auto& g : treenode->merged)
	{
		positions.push_back(g.position);
		rotations.push_back(g.rotation);
		log_scales.push_back(g.scale.array().log());
		opacities.push_back(g.opacity);
		shs.push_back(g.shs);
	}
	basenodes[id].count_merged = treenode->merged.size();

	basenodes[id].start_children = basenodes.size();
	for (int n = 0; n < treenode->children.size(); n++)
	{
		basenodes.push_back(Node());
		basenodes.back().parent = id;
		boxes.push_back(Box());
	}
	basenodes[id].count_children = treenode->children.size();

	basenodes[id].depth = treenode->depth;

	for (int n = 0; n < treenode->children.size(); n++)
	{
		populateRec(
			treenode->children[n],
			basenodes[id].start_children + n,
			gaussians, 
			positions, 
			rotations, 
			log_scales, 
			opacities,
			shs, 
			basenodes,
			boxes,
			base2tree);
	}
}

void recTraverse(int id, std::vector<Node>& nodes, int& count)
{
	if (nodes[id].depth == 0)
		count++;
	if (nodes[id].count_children != 0 && nodes[id].depth == 0)
		throw std::runtime_error("An error occurred in traversal");
	for (int i = 0; i < nodes[id].count_children; i++)
	{
		recTraverse(nodes[id].start_children + i, nodes, count);
	}
}

void Writer::makeHierarchy(
	const std::vector<Gaussian>& gaussians,
	const ExplicitTreeNode* root,
	std::vector<Eigen::Vector3f>& positions,
	std::vector<Eigen::Vector4f>& rotations,
	std::vector<Eigen::Vector3f>& log_scales,
	std::vector<float>& opacities,
	std::vector<SHs>& shs,
	std::vector<Node>& basenodes,
	std::vector<Box>& boxes,
	std::map<int, const ExplicitTreeNode*>* base2tree)
{
	basenodes.resize(1);
	boxes.resize(1);

	populateRec(
		root,
		0,
		gaussians,
		positions, rotations, log_scales, opacities, shs, basenodes, boxes,
		base2tree);
}

void Writer::writeHierarchy(const char* filename, const std::vector<Gaussian>& gaussians, const ExplicitTreeNode* root, bool compressed)
{
	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Vector4f> rotations;
	std::vector<Eigen::Vector3f> log_scales;
	std::vector<float> opacities;
	std::vector<SHs> shs;
	std::vector<Node> basenodes;
	std::vector<Box> boxes;

	makeHierarchy(gaussians, root, positions, rotations, log_scales, opacities, shs, basenodes, boxes);

	HierarchyWriter writer;
	writer.write(
		filename,
		positions.size(),
		basenodes.size(),
		positions.data(),
		shs.data(),
		opacities.data(),
		log_scales.data(),
		rotations.data(),
		basenodes.data(),
		boxes.data(),
		compressed
	);
}