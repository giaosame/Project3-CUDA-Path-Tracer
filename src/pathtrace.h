#pragma once
#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree(Scene* scene);
void pathtrace(uchar4 *pbo, int frame, int iteration);
void preprocessGltfData(Scene* scene);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter, bool denoised = false);
void denoise(uchar4* pbo, int iter, int atrous_total_num_iters, float c_phi, float n_phi, float p_phi);
