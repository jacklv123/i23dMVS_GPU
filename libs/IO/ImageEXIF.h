////////////////////////////////////////////////////////////////////
// ImageEXIF.h
//
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef __I23D_IMAGEEXIV_H__
#define __I23D_IMAGEEXIV_H__


// D E F I N E S ///////////////////////////////////////////////////

struct Exiv2Struct;


// I N C L U D E S /////////////////////////////////////////////////

#include "Image.h"


namespace I23D {

// S T R U C T S ///////////////////////////////////////////////////

class GENERAL_API CImageEXIF : public CImage
{
public:
	CImageEXIF();
	virtual ~CImageEXIF();

	void		Close();

	HRESULT		ReadHeader();
	HRESULT		ReadData(void*, PIXELFORMAT, UINT nStride, UINT lineWidth);
	HRESULT		WriteHeader(PIXELFORMAT, UINT width, UINT height, BYTE numLevels);
	HRESULT		WriteData(void*, PIXELFORMAT, UINT nStride, UINT lineWidth);

	bool		HasEXIF() const;
	bool		HasIPTC() const;
	bool		HasXMP() const;

	String		ReadKeyEXIF(const String& name, bool bInterpret=true) const;

	void		DumpAll();

protected:
	CAutoPtr<Exiv2Struct>	m_state;
}; // class CImageEXIF
/*----------------------------------------------------------------*/

} // namespace I23D

#endif // __I23D_IMAGEEXIV_H__
