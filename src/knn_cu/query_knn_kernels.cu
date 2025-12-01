#include "query_knn.cuh"
#include <stdio.h>

using namespace lbvh;

__global__ void query_knn_kernel(const BVHNode *nodes,
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
                                 )
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    StaticPriorityQueue<float, K> queue(max_radius);
    unsigned int query_idx = sorted_queries[idx];
    //printf("idx: %d, query_idx %d \n", idx, query_idx);
    query_knn(nodes, points, sorted_indices, root_index, &query_points[query_idx], queue);
    //printf("idx: %d, query_idx %d, -> %d, key: %d, dist: %f \n", idx, query_idx, queue.size(), queue[0].id, queue[0].key);
    __syncwarp(); // synchronize the warp before the write operation
    // write back the results at the correct position
    queue.write_results(&indices_out[query_idx * K], &distances_out[query_idx * K], &n_neighbors_out[query_idx]);
}