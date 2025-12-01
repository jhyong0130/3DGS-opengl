#ifndef __KNN_CU_H
#define __KNN_CU_H
#pragma once

#include <thrust/device_vector.h>
#include "knn_cu/aabb.cuh"
#include "knn_cu/morton_code.cuh"

namespace lbvh {
    struct __align__(16) BVHNode // size: 48 bytes
    {
        AABB bounds; // 24 bytes
        unsigned int parent; // 4
        unsigned int child_left; // 4 bytes
        unsigned int child_right; // 4 bytes
        int atomic; // 4
        unsigned int range_left; // 4
        unsigned int range_right; // 4
    };

    __device__
        inline HashType morton_code(const float3& point, const lbvh::AABB& extent, float resolution = 1024.0) noexcept;

    __device__
        inline HashType morton_code(const lbvh::AABB& box, const lbvh::AABB& extent, float resolution = 1024.0) noexcept;


    __device__ inline bool is_leaf(const BVHNode* node);

    // Sets the bounding box and traverses to root
    __device__ inline void process_parent(unsigned int node_idx,
        BVHNode* nodes,
        const unsigned int* morton_codes,
        unsigned int* root_index,
        unsigned int N);

    /**
     * Merge an internal node into a leaf node using the leftmost leaf node of the subtree
     * @tparam T
     * @param node
     * @param leaf
     */
    __forceinline__ __device__ void make_leaf(unsigned int node_idx,
        unsigned int leaf_idx,
        BVHNode* nodes, unsigned int N);

};

using namespace lbvh;

class lbvh_tree {
public:

	int _leaf_size;
	bool _compact;
	bool _shrink_to_fit;
	float _radius;
	bool _sort_queries;
	int _num_nodes = 0;
	int _knn = 0;

	thrust::device_vector<uint32_t> d_morton_codes;
	thrust::device_vector<AABB> d_aabbs;
	thrust::device_vector<AABB> extent;

	BVHNode* d_nodes = NULL;
	thrust::device_vector<float3> d_pts;
	thrust::device_vector<uint32_t> d_indices;
	uint32_t _root_node;

	lbvh_tree(int leaf_size, bool compact, bool shrink_to_fit, float radius, bool sort_queries, int knn) : _leaf_size(leaf_size), _compact(compact), 
						_shrink_to_fit(shrink_to_fit), _radius(radius), _sort_queries(sort_queries), _knn(knn) {

	};


	void Build(float3* pts, int num_pts);

	void Prepare_queries(float3* queries, uint32_t* morton_code, uint32_t* sorted_indices, int nb_queries);

	void Query_KNN(float3* queries, uint32_t* morton_codes, uint32_t* sorted_indices, uint32_t* indices_out, float* distances_out, uint32_t* n_neighbors_out, int nb_queries);

    void Remap2uint4(uint4* adjacencies, uint4* adjacencies_delaunay, unsigned int* indices_out_sorted, int CurrNbPts, int KVal, int KVal_d);
};



#endif