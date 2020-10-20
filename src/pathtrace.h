#pragma once
#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree(Scene* scene);
void pathtrace(uchar4 *pbo, int frame, int iteration);
void preprocessGltfData(Scene* scene);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);
