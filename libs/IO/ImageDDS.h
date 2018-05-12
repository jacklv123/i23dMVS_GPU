////////////////////////////////////////////////////////////////////
// ImageDDS.h
//
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef __I23D_IMAGEDDS_H__
#define __I23D_IMAGEDDS_H__


// I N C L U D E S /////////////////////////////////////////////////

#include "Image.h"


namespace I23D {

// S T R U C T S ///////////////////////////////////////////////////

class GENERAL_API CImageDDS : public CImage
{
public:
	CImageDDS();
	virtual ~CImageDDS();

	HRESULT		ReadHeader();
	HRESULT		ReadData(void*, PIXELFORMAT, UINT nStride, UINT lineWidth);
	HRESULT		WriteHeader(PIXELFORMAT, UINT width, UINT height, BYTE numLevels);
	HRESULT		WriteData(void*, PIXELFORMAT, UINT nStride, UINT lineWidth);
}; // class CImageDDS
/*----------------------------------------------------------------*/

} // namespace I23D

#endif // __I23D_IMAGEDDS_H__
