////////////////////////////////////////////////////////////////////
// ImageSCI.h
//
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef __I23D_IMAGESCI_H__
#define __I23D_IMAGESCI_H__


// I N C L U D E S /////////////////////////////////////////////////

#include "Image.h"


namespace I23D {

// S T R U C T S ///////////////////////////////////////////////////

class GENERAL_API CImageSCI : public CImage
{
public:
	CImageSCI();
	virtual ~CImageSCI();

	HRESULT		ReadHeader();
	HRESULT		WriteHeader(PIXELFORMAT, UINT width, UINT height, BYTE numLevels);
}; // class CImageSCI
/*----------------------------------------------------------------*/

} // namespace I23D

#endif // __I23D_IMAGESCI_H__
