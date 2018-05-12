////////////////////////////////////////////////////////////////////
// Ray.h
//
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef __I23D_RAY_H__
#define __I23D_RAY_H__


// I N C L U D E S /////////////////////////////////////////////////


// D E F I N E S ///////////////////////////////////////////////////


namespace I23D {

// S T R U C T S ///////////////////////////////////////////////////

// Basic ray class
template <typename TYPE, int DIMS>
class TRay
{
public:
	typedef Eigen::Matrix<TYPE,DIMS+1,DIMS+1,Eigen::RowMajor> MATRIX;
	typedef Eigen::Matrix<TYPE,DIMS,1> VECTOR;
	typedef Eigen::Matrix<TYPE,DIMS,1> POINT;
	typedef I23D::TAABB<TYPE,DIMS> AABB;
	typedef I23D::TOBB<TYPE,DIMS> OBB;
	typedef I23D::TPlane<TYPE> PLANE;
	static const int numScalar = (2*DIMS);

	VECTOR	m_vDir;		// ray direction (normalized)
	POINT	m_pOrig;	// ray origin

	//---------------------------------------

	inline TRay() {}
	inline TRay(const POINT& pOrig, const VECTOR& vDir);
	inline TRay(const POINT& pt0, const POINT& pt1, bool bPoints);

	inline void Set(const POINT& pOrig, const VECTOR& vDir);
	inline void SetFromPoints(const POINT& pt0, const POINT& pt1);
	inline TYPE SetFromPointsLen(const POINT& pt0, const POINT& pt1);
	inline void DeTransform(const MATRIX&);			// move to matrix space

	bool Intersects(const POINT&, const POINT&, const POINT&,
					bool bCull, TYPE *t) const;
	bool Intersects(const POINT&, const POINT&, const POINT&,
					bool bCull, TYPE fL, TYPE *t) const;
	bool Intersects(const PLANE& plane, bool bCull,
					TYPE *t, POINT* pPtHit) const;
	inline TYPE IntersectsDist(const PLANE& plane) const;
	inline POINT Intersects(const PLANE& plane) const;
	bool Intersects(const PLANE& plane, bool bCull,
					TYPE fL, TYPE *t, POINT* pPtHit) const;
	bool Intersects(const AABB& aabb) const;
	bool Intersects(const AABB& aabb, TYPE& t) const;
	bool Intersects(const AABB& aabb, TYPE fL, TYPE *t) const;
	bool Intersects(const OBB& obb) const;
	bool Intersects(const OBB& obb, TYPE &t) const;
	bool Intersects(const OBB& obb, TYPE fL, TYPE *t) const;
	bool Intersects(const TRay& ray, TYPE& s) const;
	bool Intersects(const TRay& ray, POINT& p) const;
	bool Intersects(const TRay& ray, TYPE& s1, TYPE& s2) const;
	bool IntersectsAprox(const TRay& ray, POINT& p) const;
	bool IntersectsAprox2(const TRay& ray, POINT& p) const;

	inline TYPE CosAngle(const TRay&) const;
	inline bool Coplanar(const TRay&) const;
	inline bool Parallel(const TRay&) const;

	inline TYPE Classify(const POINT&) const;
	inline POINT ProjectPoint(const POINT&) const;
	bool DistanceSq(const POINT&, TYPE&) const;
	bool Distance(const POINT&, TYPE&) const;
	TYPE DistanceSq(const POINT&) const;
	TYPE Distance(const POINT&) const;

	TRay operator * (const MATRIX&) const;			// matrix multiplication
	inline TRay& operator *= (const MATRIX&);		// matrix multiplication

	inline TYPE& operator [] (BYTE i) { ASSERT(i<6); return m_vDir.data()[i]; }
	inline TYPE operator [] (BYTE i) const { ASSERT(i<6); return m_vDir.data()[i]; }
}; // class TRay
/*----------------------------------------------------------------*/


#include "Ray.inl"
/*----------------------------------------------------------------*/

} // namespace I23D

#endif // __I23D_RAY_H__
