
#ifndef __KNN_FAISS_H
#define __KNN_FAISS_H
#pragma once
#include "Utilities.h"
#include "CVT.hpp"
#include <faiss/IndexFlat.h>
#include <faiss/Index.h> 
#include <omp.h>


void KNNSearch_Faiss(PointCloud* cloud, /*KDTree* index, */unsigned int* ret_index, float* out_dist_sqr, int num_results, int kval_d) {// Search parameters
    auto start = std::chrono::high_resolution_clock::now();
    int N = cloud->get_point_count();

    cloud->flattenPoints();
    
    // Create index
    faiss::IndexFlatL2 index(3);  // L2 distance
    index.add(N, cloud->float_pts.data()); // Add vectors to index

    // Output arrays
    std::vector<faiss::idx_t> I(N * num_results); // indices
    std::vector<float> D(N * num_results);               // distances

    // Perform search
    index.search(N, cloud->float_pts.data(), num_results, D.data(), I.data());

    int id_s = 0;
    for (size_t q = 0; q < N; ++q) {
        for (size_t i = 0; i < num_results; ++i) {
            ret_index[(num_results+kval_d)*q + i] = I[q * num_results + i];

            if (cloud->flags[q] > 0) {
                ret_index[(num_results+kval_d)*q + kval_d + num_results-1] = cloud->NbPts + id_s;
                id_s++;
            }
        }
    }
    
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();

    // Compute duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << duration << std::endl;
}

#endif