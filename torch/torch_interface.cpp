#include "torch_interface.h"
#include "../hierarchy_loader.h"
#include "../hierarchy_writer.h"
#include "../traversal.h"
#include "../runtime_switching.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LoadHierarchy(std::string filename)
{
	HierarchyLoader loader;
	
	std::vector<Eigen::Vector3f> pos;
	std::vector<SHs> shs;
	std::vector<float> alphas;
	std::vector<Eigen::Vector3f> scales;
	std::vector<Eigen::Vector4f> rot;
	std::vector<Node> nodes;
	std::vector<Box> boxes;
	
	loader.load(filename.c_str(), pos, shs, alphas, scales, rot, nodes, boxes);
	
	int P = pos.size();
	
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	torch::Tensor pos_tensor = torch::from_blob(pos.data(), {P, 3}, options).clone();
	torch::Tensor shs_tensor = torch::from_blob(shs.data(), {P, 16, 3}, options).clone();
	torch::Tensor alpha_tensor = torch::from_blob(alphas.data(), {P, 1}, options).clone();
	torch::Tensor scale_tensor = torch::from_blob(scales.data(), {P, 3}, options).clone();
	torch::Tensor rot_tensor = torch::from_blob(rot.data(), {P, 4}, options).clone();
	
	int N = nodes.size();
	torch::TensorOptions intoptions = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
	
	torch::Tensor nodes_tensor = torch::from_blob(nodes.data(), {N, 7}, intoptions).clone();
	torch::Tensor box_tensor = torch::from_blob(boxes.data(), {N, 2, 4}, options).clone();
	
	return std::make_tuple(pos_tensor, shs_tensor, alpha_tensor, scale_tensor, rot_tensor, nodes_tensor, box_tensor);
}

void WriteHierarchy(
					std::string filename,
					torch::Tensor& pos,
					torch::Tensor& shs,
					torch::Tensor& opacities,
					torch::Tensor& log_scales,
					torch::Tensor& rotations,
					torch::Tensor& nodes,
					torch::Tensor& boxes)
{
	HierarchyWriter writer;
	
	int allP = pos.size(0);
	int allN = nodes.size(0);
	
	writer.write(
		filename.c_str(),
		allP,
		allN,
		(Eigen::Vector3f*)pos.cpu().contiguous().data_ptr<float>(),
		(SHs*)shs.cpu().contiguous().data_ptr<float>(),
		opacities.cpu().contiguous().data_ptr<float>(),
		(Eigen::Vector3f*)log_scales.cpu().contiguous().data_ptr<float>(),
		(Eigen::Vector4f*)rotations.cpu().contiguous().data_ptr<float>(),
		(Node*)nodes.cpu().contiguous().data_ptr<int>(),
		(Box*)boxes.cpu().contiguous().data_ptr<float>()
	);
}

torch::Tensor
ExpandToTarget(torch::Tensor& nodes, int target)
{
	std::vector<int> indices = Traversal::expandToTarget((Node*)nodes.cpu().contiguous().data_ptr<int>(), target);
	torch::TensorOptions intoptions = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
	return torch::from_blob(indices.data(), {(int)indices.size()}, intoptions).clone();
}

int ExpandToSize(
torch::Tensor& nodes, 
torch::Tensor& boxes, 
float size, 
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& render_indices,
torch::Tensor& parent_indices,
torch::Tensor& nodes_for_render_indices)
{
	return Switching::expandToSize(
	nodes.size(0), 
	size,
	nodes.contiguous().data_ptr<int>(), 
	boxes.contiguous().data_ptr<float>(),
	viewpoint.contiguous().data_ptr<float>(),
	viewdir.data_ptr<float>()[0], viewdir.data_ptr<float>()[1], viewdir.data_ptr<float>()[2],
	render_indices.contiguous().data_ptr<int>(),
	nullptr,
	parent_indices.contiguous().data_ptr<int>(),
	nodes_for_render_indices.contiguous().data_ptr<int>());
}

void GetTsIndexed(
torch::Tensor& indices,
float size,
torch::Tensor& nodes,
torch::Tensor& boxes,
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& ts,
torch::Tensor& num_kids)
{
	Switching::getTsIndexed(
	indices.size(0),
	indices.contiguous().data_ptr<int>(),
	size,
	nodes.contiguous().data_ptr<int>(), 
	boxes.contiguous().data_ptr<float>(),
	viewpoint.data_ptr<float>()[0], viewpoint.data_ptr<float>()[1], viewpoint.data_ptr<float>()[2],
	viewdir.data_ptr<float>()[0], viewdir.data_ptr<float>()[1], viewdir.data_ptr<float>()[2],
	ts.contiguous().data_ptr<float>(),
	num_kids.contiguous().data_ptr<int>(), 0);
}