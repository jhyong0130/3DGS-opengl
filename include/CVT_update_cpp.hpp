//
//  CVT_update_cpp.hpp
//  
//
//  Created by Diego Thomas on 2025/07/24.
//


#ifndef __CVT_UPDATE_H
#define __CVT_UPDATE_H
#pragma once
#include "Utilities.h"

// isocahedron
glm::vec3 isoca[60] {
    glm::vec3(0, 0.525731, 0.850651), glm::vec3(0.850651, 0, 0.525731), glm::vec3(0.525731, 0.850651, 0),
    glm::vec3(0, 0.525731, 0.850651), glm::vec3(-0.525731, 0.850651, 0), glm::vec3(0.525731, 0.850651, 0),
    glm::vec3(0, 0.525731, 0.850651), glm::vec3(0.850651, 0, 0.525731), glm::vec3(0, -0.525731, 0.850651),
    glm::vec3(0, 0.525731, 0.850651), glm::vec3(-0.850651, 0, 0.525731), glm::vec3(0, -0.525731, 0.850651),
    glm::vec3(0, 0.525731, 0.850651), glm::vec3(-0.850651, 0, 0.525731), glm::vec3(-0.525731, 0.850651, 0),
    glm::vec3(0, -0.525731, 0.850651), glm::vec3(-0.850651, 0, 0.525731), glm::vec3(-0.525731, -0.850651, 0),
    glm::vec3(0, -0.525731, 0.850651), glm::vec3(-0.525731, -0.850651, 0), glm::vec3(0.525731, -0.850651, 0),
    glm::vec3(0, -0.525731, 0.850651), glm::vec3(0.525731, -0.850651, 0), glm::vec3(0.850651, 0, 0.525731),
    glm::vec3(0.850651, 0, 0.525731), glm::vec3(0.525731, -0.850651, 0), glm::vec3(0.850651, 0, -0.525731),
    glm::vec3(0.850651, 0, 0.525731), glm::vec3(0.525731, 0.850651, 0), glm::vec3(0.850651, 0, -0.525731),
    glm::vec3(0.525731, 0.850651, 0), glm::vec3(0.850651, 0, -0.525731), glm::vec3(0, 0.525731, -0.850651),
    glm::vec3(0.525731, 0.850651, 0), glm::vec3(0, 0.525731, -0.850651), glm::vec3(-0.525731, 0.850651, 0),
    glm::vec3(0, 0.525731, -0.850651), glm::vec3(0.850651, 0, -0.525731), glm::vec3(0, -0.525731, -0.850651),
    glm::vec3(0, 0.525731, -0.850651), glm::vec3(0, -0.525731, -0.850651), glm::vec3(-0.850651, 0, -0.525731),
    glm::vec3(0, 0.525731, -0.850651), glm::vec3(-0.850651, 0, -0.525731), glm::vec3(-0.525731, 0.850651, 0),
    glm::vec3(-0.525731, 0.850651, 0), glm::vec3(-0.850651, 0, -0.525731), glm::vec3(-0.850651, 0, 0.525731),
    glm::vec3(-0.850651, 0, -0.525731), glm::vec3(-0.850651, 0, 0.525731), glm::vec3(-0.525731, -0.850651, 0),
    glm::vec3(-0.850651, 0, -0.525731), glm::vec3(-0.525731, -0.850651, 0), glm::vec3(0, -0.525731, -0.850651),
    glm::vec3(0, -0.525731, -0.850651), glm::vec3(0.525731, -0.850651, 0), glm::vec3(0.850651, 0, -0.525731),
    glm::vec3(0, -0.525731, -0.850651), glm::vec3(0.525731, -0.850651, 0), glm::vec3(-0.525731, -0.850651, 0)
};


glm::vec4 getIntersection(std::vector<glm::vec3> vertices, std::vector<unsigned int> adjacencies, glm::vec3 curr_point, glm::vec3 ray, int myIndex, int Kval) {
    float min_dist = 1.0e32;
    float second_dist = 1.0e32;
    glm::vec3 curr_ids = glm::vec3(-1,-1,-1);
    for (int j = 0; j < Kval; j++)
    {
        unsigned int neighborID = adjacencies[Kval*myIndex + j];
        if (int(neighborID) == myIndex) continue;

        glm::vec3 point = vertices[int(neighborID)];

        // Compute bisector
        glm::vec3 center = 0.5f*(curr_point + point);
        glm::vec3 dir = point - curr_point;
        if (length(dir) < 1e-6f) continue;
        glm::vec3 planeNormal = normalize(dir);
        
        //  Compute ray/bisector intersection
        float denom = dot(planeNormal, ray);
        if (denom < 1e-6f) continue; // Lines are parallel or coincident

        float t = dot(planeNormal, center - curr_point) / denom;
        
        //std::cout << dot((curr_point + t * ray)-center, planeNormal) << std::endl;
        
        if (t >= 0.0 && t < min_dist) {
            if (min_dist < second_dist) {
                second_dist = min_dist;
            }
            min_dist = t;
            curr_ids.x = j;
            curr_ids.y = -1;
            curr_ids.z = -1;
        } else if (abs(t - min_dist) < 1.0e-3 && curr_ids.y == -1) {
            curr_ids.y = j;
        } else if (abs(t - min_dist) < 1.0e-3) {
            curr_ids.z = j;
        }
        
        if (t > min_dist && t < second_dist) {
            second_dist = t;
        }
        //std::cout << "t " << t << std::endl;
        //std::cout << "min_dist " << min_dist << std::endl;
        //std::cout << "curr_ids " << curr_ids.x << ", " << curr_ids.y << ", " << curr_ids.z << std::endl;
    }
    
    //std::cout << "min_dist: " << min_dist << std::endl;
    //std::cout << "second_dist: " << second_dist << std::endl;
    
    /*if (min_dist > 1.0e10)
        return glm::vec4(0.0);
    else*/
    return glm::vec4(min_dist, float(curr_ids.x), float(curr_ids.y), float(curr_ids.z));
}

bool test_indices(glm::vec4 inter1, glm::vec4 inter2, glm::vec4 inter3) {
    bool inter1_2, inter1_3;
    
    if (inter1.z == -1.0f) {
        inter1_2 = ( inter1.y == inter2.y || inter1.y == inter2.z || inter1.y == inter2.w );
        inter1_3 = ( inter1.y == inter3.y || inter1.y == inter3.z || inter1.y == inter3.w );
    } else if (inter1.w == -1.0f){
        inter1_2 = ( inter1.y == inter2.y || inter1.y == inter2.z || inter1.y == inter2.w ) ||
                    ( inter1.z == inter2.y || inter1.z == inter2.z || inter1.z == inter2.w );
        inter1_3 = ( inter1.y == inter3.y || inter1.y == inter3.z || inter1.y == inter3.w ) ||
                    ( inter1.z == inter3.y || inter1.z == inter3.z || inter1.z == inter3.w );
    } else {
        inter1_2 = ( inter1.y == inter2.y || inter1.y == inter2.z || inter1.y == inter2.w ) ||
                    ( inter1.z == inter2.y || inter1.z == inter2.z || inter1.z == inter2.w ) ||
                    ( inter1.w == inter2.y || inter1.w == inter2.z || inter1.w == inter2.w ) ;
        inter1_3 = ( inter1.y == inter3.y || inter1.y == inter3.z || inter1.y == inter3.w ) ||
                        ( inter1.z == inter3.y || inter1.z == inter3.z || inter1.z == inter3.w ) ||
                        ( inter1.w == inter3.y || inter1.w == inter3.z || inter1.w == inter3.w ) ;
    }
    
    return inter1_2 && inter1_3;
}

glm::vec3 Clip(glm::vec3 p, glm::vec3 in_vec, float l, glm::vec3 ray, float thresh) {
    float norm = length(in_vec);
    if (norm < thresh) return in_vec;

    //return thresh*normalize(in_vec);

    float A = dot(ray, ray);
    float B = 2.0 * dot(p, ray);
    float C = dot(p, p) - thresh*thresh;

    float discriminant = B*B - 4.0*A*C;
    if (discriminant < 0.0) {
        // No real solution
        return in_vec;
    }

    float sqrtD = sqrt(discriminant);
    float l2_1 = (-B + sqrtD) / (2.0 * A);
    float l2_2 = (-B - sqrtD) / (2.0 * A);

    // Choose the l2 that is positive
    float l2 = (l2_1 > 0.0) ? l2_1 : l2_2;

    glm::vec3 q2 = p + l2 * ray;

    return q2;
}

glm::vec4 computeCentroid(std::vector<glm::vec3> vertices, std::vector<unsigned int> adjacencies, glm::vec3 curr_point, glm::vec3 ray1, glm::vec3 ray2, glm::vec3 ray3, int myIndex, int KVal, int lvl) {
    glm::vec4 inter1 = getIntersection(vertices, adjacencies, curr_point, ray1, myIndex, KVal);
    glm::vec4 inter2 = getIntersection(vertices, adjacencies, curr_point, ray2, myIndex, KVal);
    glm::vec4 inter3 = getIntersection(vertices, adjacencies, curr_point, ray3, myIndex, KVal);

    if (inter1.x == 0.0f || inter2.x == 0.0f || inter3.x == 0.0f)
        return glm::vec4(curr_point, 1.0f);

    glm::vec3 s1 = Clip(curr_point, curr_point + inter1.x*ray1, inter1.x, ray1, 1.5f);
    glm::vec3 s2 = Clip(curr_point, curr_point + inter2.x*ray2, inter2.x, ray2, 1.5f);
    glm::vec3 s3 = Clip(curr_point, curr_point + inter3.x*ray3, inter3.x, ray3, 1.5f);

    glm::vec3 ctd = glm::vec3(0.0);
    glm::vec3 ctd_tmp, AB, AC, AD;
    float volume = 0;
    float voltmp;
    
    /*std::cout << "inter1 " << inter1.y << ", " << inter1.z << ", " << inter1.w << std::endl;
    std::cout << "inter2 " << inter2.y << ", " << inter2.z << ", " << inter2.w << std::endl;
    std::cout << "inter3 " << inter3.y << ", " << inter3.z << ", " << inter3.w << std::endl;*/
    if (test_indices(inter1, inter2, inter3) || lvl > 1) {
        ctd = (curr_point + s1 + s2 + s3)/4.0f;

        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;
        
        /*std::cout << "curr_point " << curr_point.x << ", " << curr_point.y << ", " << curr_point.z << std::endl;
        std::cout << "s1 " << s1.x << ", " << s1.y << ", " << s1.z << std::endl;
        std::cout << "s2 " << s2.x << ", " << s2.y << ", " << s2.z << std::endl;
        std::cout << "s3 " << s3.x << ", " << s3.y << ", " << s3.z << std::endl;*/
        volume = abs(dot(AB, cross(AC, AD)) / 6.0f);
        //std::cout << "volume " << volume << std::endl;
        //std::cout << "terminate " << std::endl;
        return glm::vec4(ctd, volume);
    } else {
        /*std::cout << "inter1 " << inter1.y << ", " << inter1.z << ", " << inter1.w << std::endl;
        std::cout << "inter2 " << inter2.y << ", " << inter2.z << ", " << inter2.w << std::endl;
        std::cout << "inter3 " << inter3.y << ", " << inter3.z << ", " << inter3.w << std::endl;
        std::cout << "max_dist " << std::max(length(s1-s2), std::max(length(s1-s3), length(s2-s3))) << std::endl;
        std::cout << "split " << lvl << std::endl;*/
        // split tetrahedron in middle
        glm::vec3 new_s = (s1 + s2 + s3)/3.0f;
        glm::vec3 new_central_ray = normalize(new_s - curr_point);
        
        new_s = (s1 + s2)/2.0f;
        glm::vec3 new_e1_ray = normalize(new_s - curr_point);
        
        new_s = (s1 + s3)/2.0f;
        glm::vec3 new_e2_ray = normalize(new_s - curr_point);
        
        new_s = (s2 + s3)/2.0f;
        glm::vec3 new_e3_ray = normalize(new_s - curr_point);

        // compute splitted volumes of 6 sub tets
        //////////////// First tetrahedra ////////////////
        glm::vec4 ctmp = computeCentroid(vertices, adjacencies, curr_point, ray1, new_e1_ray, new_central_ray, myIndex, KVal, lvl+1);
        ctd = ctd + glm::vec3(ctmp.x, ctmp.y, ctmp.z)*ctmp.w;
        volume = volume + ctmp.w;
        
        //std::cout << "First tetrahedra " << std::endl;
        //int dummy;
        //std::cin >> dummy;
        
        //////////////// Second tetrahedra ////////////////
        ctmp = computeCentroid(vertices, adjacencies, curr_point, ray1, new_e2_ray, new_central_ray, myIndex, KVal, lvl+1);
        ctd = ctd + glm::vec3(ctmp.x, ctmp.y, ctmp.z)*ctmp.w;
        volume = volume + ctmp.w;
        
        //std::cout << "Second tetrahedra " << std::endl;
        //std::cin >> dummy;

        //////////////// Third tetrahedra ////////////////
        ctmp = computeCentroid(vertices, adjacencies, curr_point, ray2, new_e1_ray, new_central_ray, myIndex, KVal, lvl+1);
        ctd = ctd + glm::vec3(ctmp.x, ctmp.y, ctmp.z)*ctmp.w;
        volume = volume + ctmp.w;
        
        //std::cout << "Third tetrahedra " << std::endl;
        //std::cin >> dummy;
        
        
        //////////////// Fourth tetrahedra ////////////////
        ctmp = computeCentroid(vertices, adjacencies, curr_point, ray2, new_e3_ray, new_central_ray, myIndex, KVal, lvl+1);
        ctd = ctd + glm::vec3(ctmp.x, ctmp.y, ctmp.z)*ctmp.w;
        volume = volume + ctmp.w;
        
        //std::cout << "Fourth tetrahedra " << std::endl;
        //std::cin >> dummy;
        
        //////////////// Fifth tetrahedra ////////////////
        ctmp = computeCentroid(vertices, adjacencies, curr_point, ray3, new_e3_ray, new_central_ray, myIndex, KVal, lvl+1);
        ctd = ctd + glm::vec3(ctmp.x, ctmp.y, ctmp.z)*ctmp.w;
        volume = volume + ctmp.w;
        
        //std::cout << "Fifth tetrahedra " << std::endl;
        //std::cin >> dummy;
                
        //////////////// Sixth tetrahedra ////////////////
        ctmp = computeCentroid(vertices, adjacencies, curr_point, ray3, new_e2_ray, new_central_ray, myIndex, KVal, lvl+1);
        ctd = ctd + glm::vec3(ctmp.x, ctmp.y, ctmp.z)*ctmp.w;
        volume = volume + ctmp.w;
        
        //std::cout << "Sixth tetrahedra " << std::endl;
        //std::cin >> dummy;
        
        return glm::vec4(ctd/volume, volume);
    }
}

glm::vec3 Center(std::vector<glm::vec3> vertices, std::vector<unsigned int> adjacencies, glm::vec3 curr_point, int myIndex, int Kval)
{
    glm::vec3 center_out = glm::vec3(0.0f, 0.0f, 0.0f);
    float tot_area = 0.0f;
    glm::vec4 curr_center;
    
    // loop through all 20 tetrehedra that tesselate the sphere
    for (int j = 0; j < 20; j++) {
        //std::cout << "tet " << j << std::endl;
        curr_center = computeCentroid(vertices, adjacencies, curr_point, isoca[3*j], isoca[3*j+1], isoca[3*j+2], myIndex, Kval, 0);
        //std::cout << "curr_center " << curr_center.x << ", " << curr_center.y << ", " << curr_center.z << std::endl;
        center_out = center_out + glm::vec3(curr_center.x, curr_center.y, curr_center.z) * curr_center.w;
        tot_area = tot_area + curr_center.w;
    }
    
    if (tot_area > 0.0f) {
        center_out.x = center_out.x / tot_area;
        center_out.y = center_out.y / tot_area;
        center_out.z = center_out.z / tot_area;
    }
    float lambda = 1.0f;
    return (length(center_out) < 1.4f) ? (1.0f-lambda)*curr_point + lambda * center_out : curr_point;
}

std::vector<glm::vec3> CVTUpdate_cpp(std::vector<glm::vec3> vertices, std::vector<unsigned int> adjacencies, int Kval) {
    
    std::vector<glm::vec3> outPosition;
    
    int index = 0;
    for (const auto& v : vertices) {
        std::cout << "index " << index << std::endl;
        outPosition.push_back(Center(vertices, adjacencies, v, index, Kval));
        index++;
    }
    
    return outPosition;
}

#endif
