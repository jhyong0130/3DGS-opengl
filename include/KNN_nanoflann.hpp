
#ifndef __KNN_H
#define __KNN_H
#pragma once
#include "Utilities.h"
#include "CVT.hpp"
#include <nanoflann.hpp>
#include <omp.h>

// Define the KD-Tree type
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud>,
    PointCloud,
    3 /* dim */>;

/*KDTree buildKNN(PointCloud* cloud) {
    // Create and build the index
    size_t max_leaf = cloud->get_point_count() < 1000 ? 10 : 30;
    KDTree index(3, *cloud, {max_leaf});
    index.buildIndex();

    return index;
}*/

void KNNSearch(PointCloud* cloud, /*KDTree* index, */unsigned int* ret_index, float* out_dist_sqr, int num_results, int kval_d) {// Search parameters
    auto start = std::chrono::high_resolution_clock::now();
    int N = cloud->get_point_count();
    
    size_t max_leaf = N < 1000 ? 10 : 30;
    KDTree index(3 /*dim*/, *cloud, {max_leaf});
    index.buildIndex();

    int nb_thread = 32;
    int bach_size = (N / nb_thread) +1;
    omp_set_num_threads(nb_thread);
    int q;
    #pragma omp parallel for
    for (q = 0; q < nb_thread; ++q) {
        for (int b = 0; b < bach_size; ++b) {
            int idx_t = q * bach_size + b;
            if (idx_t < N) {
                const float query_pt[3] = { cloud->pts[idx_t].x, cloud->pts[idx_t].y, cloud->pts[idx_t].z };
                index.knnSearch(query_pt, num_results, &ret_index[(num_results + kval_d) * idx_t + kval_d], &out_dist_sqr[(num_results + kval_d) * idx_t + kval_d]);
            }
        }
    }

    int id_s = 0;
    for (size_t q = 0; q < N; ++q) {
        if (cloud->flags[q] > 0) {
            ret_index[(num_results+kval_d)*q + kval_d + num_results-1] = cloud->NbPts + id_s;
            id_s++;
        }
    }

    // Record end time
    auto end = std::chrono::high_resolution_clock::now();

    // Compute duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << duration.count() << std::endl;
}

#endif