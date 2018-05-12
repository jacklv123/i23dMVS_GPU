////////////////////////////////////////////////////////////////////
// ImageJPG.h
//
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef __I23D_IMAGEJPG_H__
#define __I23D_IMAGEJPG_H__


// D E F I N E S ///////////////////////////////////////////////////


// I N C L U D E S /////////////////////////////////////////////////

#include "Image.h"


namespace I23D {

// S T R U C T S ///////////////////////////////////////////////////

class GENERAL_API CImageJPG : public CImage
{
public:
	CImageJPG();
	virtual ~CImageJPG();

	void		Close();

	HRESULT		ReadHeader();
	HRESULT		ReadData(void*, PIXELFORMAT, UINT nStride, UINT lineWidth);
	HRESULT		WriteHeader(PIXELFORMAT, UINT width, UINT height, BYTE numLevels);
	HRESULT		WriteData(void*, PIXELFORMAT, UINT nStride, UINT lineWidth);

protected:
	void*		m_state;
}; // class CImageJPG
/*----------------------------------------------------------------*/

} // namespace I23D

#endif // __I23D_IMAGEJPG_H__
