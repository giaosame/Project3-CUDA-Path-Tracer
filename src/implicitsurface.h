#pragma once
#include "utilities.h"
#include "sceneStructs.h"

class ImplicitSurface
{
public:
	__host__ __device__
	static glm::vec3 computeSurfaceNormal(const Geom& geom, const glm::vec3& p)
	{
		const glm::vec3 delta_x(0.001f, 0, 0);
		const glm::vec3 delta_y(0, 0.001f, 0);
		const glm::vec3 delta_z(0, 0, 0.001f);

		glm::vec3 normal;
		if (geom.type == GeomType::HEART)
		{
			normal = glm::vec3(heartSDF(p + delta_x) - heartSDF(p - delta_x),
							   heartSDF(p + delta_y) - heartSDF(p - delta_y),
							   heartSDF(p + delta_z) - heartSDF(p - delta_z));
		}
		else if (geom.type == GeomType::TANGLECUBE)
		{
			normal = glm::vec3(tanglecubeSDF(p + delta_x) - tanglecubeSDF(p - delta_x),
							   tanglecubeSDF(p + delta_y) - tanglecubeSDF(p - delta_y),
							   tanglecubeSDF(p + delta_z) - tanglecubeSDF(p - delta_z));
		}
		else if (geom.type == GeomType::TORUS)
		{
			normal =  glm::vec3(torusSDF(p + delta_x) - torusSDF(p - delta_x),
								torusSDF(p + delta_y) - torusSDF(p - delta_y),
							    torusSDF(p + delta_z) - torusSDF(p - delta_z));
		}
		
		return glm::length(normal) > 0 ? glm::normalize(normal) : normal;
	}

	__host__ __device__
	static float torusSDF(const glm::vec3& p, const float r1 = 1.0f, const float r2 = 0.5f)
	{
		const float q = sqrt(p.x * p.x + p.z * p.z) - r1;
		const float len = sqrt(q * q + p.y * p.y);
		return len - r2;
	}

	__host__ __device__
	static float heartSDF(const glm::vec3& p, const float h = 1.0f, const float r1 =	1.0f, const float r2 = 0.0f)
	{
		const float x2 = p.x * p.x;
		const float y2 = p.y * p.y;
		const float z2 = p.z * p.z;
		const float z3 = p.z * z2;
		const float temp = x2 + 9.f * y2 / 4.f + z2 - 1.f;
		return temp * temp * temp - x2 * z3 - 9.f * y2 * z3 / 80.f;
	}

	__host__ __device__
	static float tanglecubeSDF(const glm::vec3& p)
	{
		// x^4 - 5x^2 + y^4 - 5y^2 + z^4 - 5z^2 + 11.8 = 0.
		const float x2 = p.x * p.x;
		const float y2 = p.y * p.y;
		const float z2 = p.z * p.z;
		return x2 * x2 - 5.f * x2 + y2 * y2 - 5.f * y2 + z2 * z2 - 5.f * z2 + 11.8f;
	}
};