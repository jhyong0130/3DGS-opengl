#define HASH_64 1 // use 64 bit morton codes

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/system/cuda/execution_policy.h> 

#include <iostream>
#include "KNN_cuda.cuh"
#include "knn_cu/aabb.cuh"
#include "knn_cu/morton_code.cuh"

using namespace lbvh;

__device__ inline lbvh::HashType lbvh::morton_code(const float3& point, const lbvh::AABB& extent, float resolution) noexcept {
    float3 p = point;

    // scale to [0, 1]
    p.x -= extent.min.x;
    p.y -= extent.min.y;
    p.z -= extent.min.z;

    p.x /= (extent.max.x - extent.min.x);
    p.y /= (extent.max.y - extent.min.y);
    p.z /= (extent.max.z - extent.min.z);
    return morton_code(p, resolution);
}

__device__ inline lbvh::HashType lbvh::morton_code(const lbvh::AABB& box, const lbvh::AABB& extent, float resolution) noexcept {
    auto p = centroid(box);
    return morton_code(p, extent, resolution);
}


__device__ inline bool lbvh::is_leaf(const BVHNode* node) {
    return node->child_right == UINT_MAX && node->child_right == UINT_MAX;
}

// Sets the bounding box and traverses to root
__device__ inline void lbvh::process_parent(unsigned int node_idx,
    BVHNode* nodes,
    const unsigned int* morton_codes,
    unsigned int* root_index,
    unsigned int N)
{
    unsigned int current_idx = node_idx;
    BVHNode* current_node = &nodes[current_idx];

    while (true) {
        // Allow only one thread to process a node
        if (atomicAdd(&(current_node->atomic), 1) != 1)
            //printf("Terminating at node %u\n", node_idx);
            return; // terminate the first thread encountering this

        //printf("Processing node %u\n", current_idx);
        //printf("Node %u, is leaf: %u\n", current_idx, is_leaf(current_node));
        //printf("Node %u children: %u, %u\n", current_idx, current_node->child_left, current_node->child_right);

        unsigned int left = current_node->range_left;
        unsigned int right = current_node->range_right;
        //printf("Range of node %u: %u, %u\n", current_idx, left, right);

        // Set bounding box if the node is no leaf
        if (!is_leaf(current_node)) {
            // Fuse bounding box from children AABBs
            current_node->bounds = merge(nodes[current_node->child_left].bounds,
                nodes[current_node->child_right].bounds);
        }


        if (left == 0 && right == N - 1) {
            root_index[0] = current_idx; // return the root
            return; // at the root, abort
        }


        unsigned int parent_idx;
        BVHNode* parent;

        if (left == 0 || (right != N - 1 && highest_bit(morton_codes[right], morton_codes[right + 1]) <
            highest_bit(morton_codes[left - 1], morton_codes[left]))) {
            // parent = right, set parent left child and range to node
            parent_idx = N + right;

            parent = &nodes[parent_idx];
            parent->child_left = current_idx;
            parent->range_left = left;
        }
        else {
            // parent = left -1, set parent right child and range to node
            parent_idx = N + left - 1;

            parent = &nodes[parent_idx];
            parent->child_right = current_idx;
            parent->range_right = right;
        }

        current_node->parent = parent_idx; // store the parent in the current node

        // up to the parent next
        current_node = parent;
        current_idx = parent_idx;
    }
}




__global__ void compute_morton_kernel(AABB* __restrict__ const aabbs,
    AABB* __restrict__ const extent,
    unsigned int* morton_codes,
    unsigned int N);

__global__ void initialize_tree_kernel(BVHNode* nodes,
    const AABB* sorted_aabbs,
    unsigned int N);

__global__ void construct_tree_kernel(BVHNode* nodes,
    unsigned int* root_index,
    const unsigned int* sorted_morton_codes,
    unsigned int N);

__global__ void optimize_tree_kernel(BVHNode* nodes,
    unsigned int* root_index,
    unsigned int* valid,
    unsigned int max_node_size,
    unsigned int N);

__global__ void compute_free_indices_kernel(const unsigned int* valid_sums,
    const unsigned int* isums,
    unsigned int* free_indices,
    unsigned int N);

__global__ void compact_tree_kernel(BVHNode* nodes,
    unsigned int* root_index,
    const unsigned int* valid_sums,
    const unsigned int* free_positions,
    unsigned int first_moved,
    unsigned int node_cnt_new,
    unsigned int N);

__global__ void init_aabbs_extent_kernel(AABB* aabbs, float3* pts, unsigned int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    aabbs[idx].min.x = pts[idx].x;
    aabbs[idx].min.y = pts[idx].y;
    aabbs[idx].min.z = pts[idx].z;

    aabbs[idx].max.x = pts[idx].x;
    aabbs[idx].max.y = pts[idx].y;
    aabbs[idx].max.z = pts[idx].z;
}

__global__ void compute_morton_points_kernel(float3* __restrict__ const points,
            AABB* __restrict__ const extent,
            unsigned int* morton_codes,
            unsigned int N);

__global__ void query_knn_kernel(const BVHNode* nodes,
    const float3* __restrict__ points,
    const unsigned int* __restrict__ sorted_indices,
    const unsigned int root_index,
    const float max_radius,
    const float3* __restrict__ query_points,
    const unsigned int* __restrict__ sorted_queries,
    const unsigned int N,
    // custom parameters
    unsigned int* indices_out,
    float* distances_out,
    unsigned int* n_neighbors_out
);

struct float3_min {
    __host__ __device__
        float3 operator()(const float3& a, const float3& b) const {
        return {
            fminf(a.x, b.x),
            fminf(a.y, b.y),
            fminf(a.z, b.z)
        };
    }
};

struct float3_max {
    __host__ __device__
        float3 operator()(const float3& a, const float3& b) const {
        return {
            fmaxf(a.x, b.x),
            fmaxf(a.y, b.y),
            fmaxf(a.z, b.z)
        };
    }
};


// aabbs should be initialized with pts
void lbvh_tree::Build(float3* pts, int num_pts) {
    //std::cout << "Building the LBVH tree " << std::endl;
	int num_nodes = num_pts * 2 - 1;
    _num_nodes = num_nodes;
    d_pts.clear();
    extent.clear();
    d_aabbs.clear();
    d_morton_codes.clear();
    d_indices.clear();

    int threadsPerBlock = 256;  // Safer default
    int blocksPerGrid = (num_pts + threadsPerBlock - 1) / threadsPerBlock;

    // Wrap raw device pointer with thrust::device_ptr
    thrust::device_ptr<float3> dev_pts_ptr(pts);
    // Create device_vector by copying from device_ptr range
    d_pts = thrust::device_vector<float3>(dev_pts_ptr, dev_pts_ptr + num_pts);
    
    cudaError_t errr;
    AABB min_max_pts;
    float3 res;
    try {
        res = thrust::reduce(d_pts.begin(),
            d_pts.end(), float3{ FLT_MAX, FLT_MAX, FLT_MAX }, float3_min());
        cudaDeviceSynchronize();
        errr = cudaGetLastError();  // Check for launch error
        if (errr != cudaSuccess) {
            printf("CUDA reduce launch error: %s\n", cudaGetErrorString(errr));
        }
    }
    catch (thrust::system_error& e) {
        std::cerr << "Thrust error: " << e.what() << std::endl;
    }
    catch (std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
    }
    min_max_pts.min.x = res.x;
    min_max_pts.min.y = res.y;
    min_max_pts.min.z = res.z;
    //std::cout << min_max_pts.min.x << ", " << min_max_pts.min.y << ", " << min_max_pts.min.z << std::endl;

    try {
        res = thrust::reduce(d_pts.begin(),
            d_pts.end(), float3{ -FLT_MAX, -FLT_MAX, -FLT_MAX }, float3_max());
        cudaDeviceSynchronize();
        if (errr != cudaSuccess) {
            printf("CUDA init_aabbs_extent_kernel launch error: %s\n", cudaGetErrorString(errr));
        }
        //float3{ -FLT_MAX, -FLT_MAX, -FLT_MAX },
        //float3_max());
    }
    catch (thrust::system_error& e) {
        std::cerr << "Thrust error: " << e.what() << std::endl;
    }
    catch (std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
    }
    min_max_pts.max.x = res.x;
    min_max_pts.max.y = res.y;
    min_max_pts.max.z = res.z;
    //std::cout << min_max_pts.max.x << ", " << min_max_pts.max.y << ", " << min_max_pts.max.z << std::endl;

    extent = thrust::device_vector<AABB>(1);
    thrust::copy_n(&min_max_pts, 1, extent.begin());

    AABB ex = extent[0];
    if ((ex.max.x - ex.min.x) <= 0.0f ||
        (ex.max.y - ex.min.y) <= 0.0f ||
        (ex.max.z - ex.min.z) <= 0.0f) {
        std::cerr << "Invalid extent! Cannot normalize coordinates.\n";
        exit(1);
    }

    d_aabbs = thrust::device_vector<AABB>(num_pts);
    init_aabbs_extent_kernel << < blocksPerGrid, threadsPerBlock >> > (thrust::raw_pointer_cast(d_aabbs.data()), pts, num_pts);
    errr = cudaGetLastError();  // Check for launch error
    if (errr != cudaSuccess) {
        printf("CUDA init_aabbs_extent_kernel launch error: %s\n", cudaGetErrorString(errr));
    }

    d_morton_codes = thrust::device_vector<uint32_t>(num_pts);
	compute_morton_kernel << < blocksPerGrid, threadsPerBlock >> > (thrust::raw_pointer_cast(d_aabbs.data()),
                                                                thrust::raw_pointer_cast(extent.data()),
                                                                thrust::raw_pointer_cast(d_morton_codes.data()), num_pts);

    cudaDeviceSynchronize();
    errr = cudaGetLastError();  // Check for launch error
    if (errr != cudaSuccess) {
        printf("CUDA compute_morton_kernel launch error: %s\n", cudaGetErrorString(errr));
    }

    // Create device vector for indices: 0,1,2,...N-1
    d_indices = thrust::device_vector<uint32_t>(num_pts);
    thrust::sequence(d_indices.begin(), d_indices.end());


    assert(d_morton_codes.size() == d_indices.size());
    try {
        // Sort Morton codes, reorder indices accordingly
        auto exec_policy = thrust::cuda::par.on(0);
        thrust::sort_by_key(exec_policy, d_morton_codes.begin(), d_morton_codes.end(), d_indices.begin());
        //thrust::stable_sort_by_key(exec_policy, d_morton_codes.begin(), d_morton_codes.end(), d_indices.begin());
    }
    catch (thrust::system_error& e) {
        std::cerr << "Thrust error: " << e.what() << std::endl;
    }
    catch (std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
    }

    //std::cout << "Reorder AABBs" << std::endl;
    // Reorder AABBs based on sorted indices using gather
    thrust::device_vector<AABB> d_aabbs_sorted(num_pts);
    thrust::gather(d_indices.begin(), d_indices.end(), d_aabbs.begin(), d_aabbs_sorted.begin());

    if (d_nodes == NULL) {
        cudaMalloc((void**)&d_nodes, num_nodes * sizeof(BVHNode));
    }
    else {
        cudaFree(d_nodes);
        cudaMalloc((void**)&d_nodes, num_nodes * sizeof(BVHNode));
    }

    thrust::device_vector<uint32_t> d_root_node(1);
    thrust::fill(d_root_node.begin(), d_root_node.end(), UINT_MAX);

    initialize_tree_kernel << < blocksPerGrid, threadsPerBlock >> > (d_nodes, thrust::raw_pointer_cast(d_aabbs_sorted.data()), num_pts);

    construct_tree_kernel << < blocksPerGrid, threadsPerBlock >> > (d_nodes, thrust::raw_pointer_cast(d_root_node.data()), thrust::raw_pointer_cast(d_morton_codes.data()), num_pts);

    cudaDeviceSynchronize();
    errr = cudaGetLastError();  // Check for launch error
    if (errr != cudaSuccess) {
        printf("CUDA construct_tree_kernel launch error: %s\n", cudaGetErrorString(errr));
    }

    if (_leaf_size > 1) {
        //std::cout << "_leaf_size > 1 " << std::endl;
        thrust::device_vector<uint32_t> d_valid(num_nodes);
        thrust::fill(d_valid.begin(), d_valid.end(), 1);
        //std::cout << "d_valid done" << std::endl;

        optimize_tree_kernel << < blocksPerGrid, threadsPerBlock >> > (d_nodes,
            thrust::raw_pointer_cast(d_root_node.data()),
            thrust::raw_pointer_cast(d_valid.data()),
            _leaf_size,
            num_pts);
        cudaDeviceSynchronize();
        //std::cout << "optimize_tree_kernel done" << std::endl;
        cudaError_t errr = cudaGetLastError();  // Check for launch error
        if (errr != cudaSuccess) {
            printf("CUDA update_kernel launch error: %s\n", cudaGetErrorString(errr));
        }

        if (_compact) {
            //std::cout << "_compact" << std::endl;
            // compute the prefix sum of the valid array to determine the indices of the free space
           // Step 1: Resize valid_sums to be one element longer than valid
            thrust::device_vector<uint32_t> d_valid_sums(d_valid.size() + 1);

            // Step 2: Set the first element to 0
            d_valid_sums[0] = 0;

            // Step 3: Compute exclusive scan of 'valid' into valid_sums starting at position 1
            thrust::exclusive_scan(d_valid_sums.begin(), d_valid_sums.end(), d_valid_sums.begin() + 1);

            // get the number of actually used nodes after optimization
            uint32_t new_node_count = d_valid_sums[num_nodes];
            // leave out the last element again to align with the valid array

            thrust::device_vector<uint32_t> d_valid_sums_aligned(d_valid_sums.begin(), d_valid_sums.end() - 1);

            // compute the isum parameter to get the indices of the free elements
            thrust::device_vector<uint32_t> isum(num_nodes);

            thrust::transform(
                thrust::make_counting_iterator<uint32_t>(0),
                thrust::make_counting_iterator<uint32_t>(num_nodes),
                d_valid_sums_aligned.begin(),
                isum.begin(),
                thrust::minus<uint32_t>()  // i - valid_sums_aligned[i]
            );

            // number of free elements in the optimized tree array
            uint32_t free_indices_size = uint32_t(isum[new_node_count]);

            thrust::device_vector<uint32_t> d_free(d_valid.begin(), d_valid.begin() + free_indices_size);  // reuse the valid space as it is not needed any more

            // compute the free indices
            blocksPerGrid = (new_node_count + threadsPerBlock - 1) / threadsPerBlock;
            compute_free_indices_kernel << < blocksPerGrid, threadsPerBlock >> > (thrust::raw_pointer_cast(d_valid_sums.data()),
                thrust::raw_pointer_cast(isum.data()),
                thrust::raw_pointer_cast(d_free.data()),
                new_node_count);

            // get the sum of the first object that has to be moved
            uint32_t first_moved = d_valid_sums[new_node_count];

            blocksPerGrid = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;
            compact_tree_kernel << < blocksPerGrid, threadsPerBlock >> > (d_nodes,
                thrust::raw_pointer_cast(d_root_node.data()),
                thrust::raw_pointer_cast(d_valid_sums_aligned.data()),
                thrust::raw_pointer_cast(d_free.data()),
                first_moved,
                new_node_count,
                num_nodes);

            if (_shrink_to_fit) {
                BVHNode *d_nodes_old = d_nodes;
                cudaMalloc((void**)&d_nodes, new_node_count * sizeof(BVHNode));
                cudaMemcpy(d_nodes, d_nodes_old, new_node_count * sizeof(BVHNode), cudaMemcpyDeviceToDevice);
                num_nodes = new_node_count;
            }
        }

        // Wait for GPU to finish before exiting
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();  // Check for launch error
        if (err != cudaSuccess) {
            printf("CUDA update_kernel launch error: %s\n", cudaGetErrorString(err));
        }

        // fetch to root node's location in the tree
        _root_node = d_root_node[0];

    }

    //std::cout << "The LBVH tree is built " << _leaf_size << std::endl;
}


void lbvh_tree::Prepare_queries(float3* queries, uint32_t* morton_code, uint32_t* sorted_indices, int nb_queries) {
    int threadsPerBlock = 256;  // Safer default
    int blocksPerGrid = (nb_queries + threadsPerBlock - 1) / threadsPerBlock;

    // only for large queries : sort them in morton order to prevent too much warp divergence on tree traversal
    thrust::device_ptr<uint32_t> dev_id_ptr(sorted_indices);
    thrust::device_vector<uint32_t> q_sorted_indicese(dev_id_ptr, dev_id_ptr + nb_queries);
    if (_sort_queries) {
        //morton_codes = cp.empty(queries.shape[0], dtype = cp.uint64)
        compute_morton_points_kernel << < blocksPerGrid, threadsPerBlock >> > (queries, thrust::raw_pointer_cast(extent.data()), morton_code, nb_queries);
        // Wait for GPU to finish before exiting
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();  // Check for launch error
        if (err != cudaSuccess) {
            printf("CUDA compute_morton_points_kernel launch error: %s\n", cudaGetErrorString(err));
        }

        // Wrap raw device pointer with thrust::device_ptr
        thrust::device_ptr<uint32_t> dev_ptr(morton_code);
        // Create device_vector by copying from device_ptr range
        thrust::device_vector<uint32_t> q_morton_code(dev_ptr, dev_ptr + nb_queries);


        // Create device vector for indices: 0,1,2,...N-1
        thrust::sequence(q_sorted_indicese.begin(), q_sorted_indicese.end());

        // Sort Morton codes, reorder indices accordingly
        thrust::sort_by_key(q_morton_code.begin(), q_morton_code.end(), q_sorted_indicese.begin());
    }
    else {
        // Create device vector for indices: 0,1,2,...N-1
        thrust::sequence(q_sorted_indicese.begin(), q_sorted_indicese.end());
    }

    cudaMemcpy(sorted_indices, thrust::raw_pointer_cast(q_sorted_indicese.data()), nb_queries * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    //std::cout << "Prepare_queries done " << std::endl;
    return;
}


void lbvh_tree::Query_KNN(float3* queries, uint32_t* morton_codes, uint32_t* sorted_indices, uint32_t* indices_out, float* distances_out, uint32_t* n_neighbors_out, int nb_queries) {
    if (_num_nodes < 0) {
        std::cout << "Index has not been built yet. Call 'build' first." << std::endl;
    }

    //std::cout << "Start Prepare_queries." << std::endl;
    Prepare_queries(queries, morton_codes, sorted_indices, nb_queries);

    /*uint32_t* sorted_indices_h = (uint32_t*)malloc(nb_queries * sizeof(uint32_t));
    cudaMemcpy(sorted_indices_h, thrust::raw_pointer_cast(d_indices.data()), nb_queries * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nb_queries; i++) {
        std::cout << i << " -> " << sorted_indices_h[i] << std::endl;
    }

    std::cout << " Root node " << _root_node << std::endl;*/

    // use the maximum allowed threads per block from the kernel(depends on the number of registers)

    /*thrust::device_vector<uint32_t> indices_out(nb_queries * _knn);
    thrust::fill(indices_out.begin(), indices_out.end(), UINT_MAX);

    thrust::device_vector<float_t> distances_out(nb_queries * _knn);
    thrust::fill(distances_out.begin(), distances_out.end(), 1.0e32f);

    thrust::device_vector<uint32_t> n_neighbors_out(nb_queries);
    thrust::fill(n_neighbors_out.begin(), n_neighbors_out.end(), 0);*/

    int threadsPerBlock = 256;  // Safer default
    int blocksPerGrid = (nb_queries + threadsPerBlock - 1) / threadsPerBlock;

    query_knn_kernel <<< blocksPerGrid, threadsPerBlock >>> (d_nodes,
        queries, //thrust::raw_pointer_cast(d_pts.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        _root_node,
        _radius,
        queries,
        sorted_indices,
        nb_queries,
        // custom parameters
        indices_out,
        distances_out,
        n_neighbors_out
    );

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();  // Check for launch error
    if (err != cudaSuccess) {
        printf("CUDA query_knn_kernel launch error: %s\n", cudaGetErrorString(err));
    }
    //int tmp;
    //std::cin >> tmp;
}

__global__ void map2uint4_kernel(uint4* adjacencies, uint4* adjacencies_delaunay, unsigned int* indices_out_sorted, int N, int KVal, int KVal_d) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    for (int i = 0; i < KVal / 4; i++) {
        adjacencies[((KVal + KVal_d) / 4) * idx + i].x = indices_out_sorted[KVal * idx + 4 * i];
        adjacencies[((KVal + KVal_d) / 4) * idx + i].y = indices_out_sorted[KVal * idx + 4 * i + 1];
        adjacencies[((KVal + KVal_d) / 4) * idx + i].z = indices_out_sorted[KVal * idx + 4 * i + 2];
        adjacencies[((KVal + KVal_d) / 4) * idx + i].w = indices_out_sorted[KVal * idx + 4 * i + 3];
    }

    for (int i = 0; i < KVal_d / 4; i++) {
        adjacencies[((KVal + KVal_d) / 4) * idx + (KVal / 4) + i].x = adjacencies_delaunay[(KVal_d / 4) * idx + i].x;
        adjacencies[((KVal + KVal_d) / 4) * idx + (KVal / 4) + i].y = adjacencies_delaunay[(KVal_d / 4) * idx + i].y;
        adjacencies[((KVal + KVal_d) / 4) * idx + (KVal / 4) + i].z = adjacencies_delaunay[(KVal_d / 4) * idx + i].z;
        adjacencies[((KVal + KVal_d) / 4) * idx + (KVal / 4) + i].w = adjacencies_delaunay[(KVal_d / 4) * idx + i].w;
    }
}

void lbvh_tree::Remap2uint4(uint4* adjacencies, uint4* adjacencies_delaunay, unsigned int* indices_out_sorted, int CurrNbPts, int KVal, int KVal_d) {

    int threadsPerBlock = 256;  // Safer default
    int blocksPerGrid = (CurrNbPts + threadsPerBlock - 1) / threadsPerBlock;

    map2uint4_kernel << < blocksPerGrid, threadsPerBlock >> > (adjacencies, adjacencies_delaunay, indices_out_sorted, CurrNbPts, KVal, KVal_d);

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();  // Check for launch error
    if (err != cudaSuccess) {
        printf("CUDA update_kernel launch error: %s\n", cudaGetErrorString(err));
    }
}