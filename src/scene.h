#pragma once
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "gltf-loader.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    void collectLightGeoms();

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Geom> lightGeoms;
    std::vector<Material> materials;
    RenderState state;

    int getMeshesSize() const;
    int getTexturesSize() const;

    // class members for gltf meshes
    std::vector<gltf::Mesh<float>> gltfMeshes;
    std::vector<gltf::Material> gltfMaterials;
    std::vector<gltf::Texture> gltfTextures;

    std::vector<unsigned int> faces_offset;
    std::vector<unsigned int> vertices_offset;

    int total_faces;
    int total_vertices;
};
