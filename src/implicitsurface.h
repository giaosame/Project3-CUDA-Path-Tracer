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
		if (geom.type == GeomType::CONE)
		{
			normal = glm::vec3(coneSDF(p + delta_x) - coneSDF(p - delta_x),
							   coneSDF(p + delta_y) - coneSDF(p - delta_y),
							   coneSDF(p + delta_z) - coneSDF(p - delta_z));
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
	static float coneSDF(const glm::vec3& p, const float h = 1.0f, const float r1 =	1.0f, const float r2 = 0.0f)
	{
		const glm::vec2 q(glm::length(glm::vec2(p.x, p.z)), p.y);
		const glm::vec2 k1(r2, h);
		const glm::vec2 k2(r2 - r1, 2.f * h);
		const glm::vec2 ca(q.x - glm::min(q.x, (q.y < 0.f) ? r1 : r2), abs(q.y) - h);
		const glm::vec2 cb = q - k1 + k2 * glm::clamp(glm::dot(k1 - q, k2) / glm::dot(k2, k2), 0.f, 1.f);
		float s = (cb.x < 0.f && ca.y < 0.f) ? -1.f : 1.f;
		return s * sqrt(glm::min(glm::dot(ca, ca), glm::dot(cb, cb)));
	}

	__host__ __device__
	static float tanglecubeSDF(const glm::vec3& p)
	{
		// x^4 - 5x^2 + y^4 - 5y^2 + z^4 - 5z^2 + 11.8 = 0.
		const float x2 = p.x * p.x;
		const float y2 = p.y * p.y;
		const float z2 = p.z * p.z;
		return x2 * x2 - 5 * x2 + y2 * y2 - 5 * y2 + z2 * z2 - 5 * z2 + 11.8f;
	}
};