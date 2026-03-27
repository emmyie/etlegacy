/**
 * @file dx12_image.cpp
 * @brief DX12-only image loader.
 *
 * Supports TGA (types 2 and 10), JPEG (via libjpeg), and PNG (self-contained
 * decoder using the puff() DEFLATE decompressor – no libpng dependency,
 * mirrors the approach of src/renderercommon/tr_image_png.c).
 * Pixel data is loaded via dx12.ri.FS_ReadFile and decoded into a plain
 * malloc()-allocated RGBA buffer.
 */

#include "dx12_image.h"

#ifdef _WIN32

#include <stdlib.h>   // malloc / free / abs
#include <string.h>   // memcpy / memcmp
#include <setjmp.h>   // setjmp / longjmp  (JPEG error recovery)
#include <jpeglib.h>  // libjpeg decode  (linked via renderer_libraries)

// puff() is compiled as C; forward-declare with C linkage so C++ name
// mangling does not produce a mismatched symbol at link time.
extern "C" int32_t puff(uint8_t *dest, uint32_t *destlen,
                         uint8_t *source, uint32_t *sourcelen);

#define Q3IMAGE_BYTESPERPIXEL 4  // RGBA format: 4 bytes per pixel

// ---------------------------------------------------------------------------
// TGA decoding
// ---------------------------------------------------------------------------

#define TGA_HEADER_SIZE 18

/**
 * @brief Read a little-endian 16-bit unsigned integer from two bytes.
 */
static unsigned short TGA_Read16(const byte *p)
{
	return (unsigned short)(p[0] | ((unsigned short)p[1] << 8));
}

/**
 * @brief Decode a TGA image in memory to RGBA.
 *
 * Supports:
 *   - Type  2: uncompressed true-color (24- or 32-bit)
 *   - Type 10: RLE-compressed true-color (24- or 32-bit)
 *
 * Output is always 4 bytes per pixel (RGBA), top-left origin.
 *
 * @param[in]  buf      Raw file bytes.
 * @param[in]  bufSize  Number of bytes in buf.
 * @param[out] pic      Set to a malloc()-allocated RGBA buffer on success.
 * @param[out] width    Image width in pixels.
 * @param[out] height   Image height in pixels.
 */
static void DX12_DecodeTGA(const byte *buf, int bufSize,
                            byte **pic, int *width, int *height)
{
	int  idLength;
	int  colorMapType;
	int  imageType;
	int  w;
	int  h;
	int  pixelDepth;
	int  imageDescriptor;
	int  bytesPerPixel;
	int  numPixels;
	int  topLeft;
	byte *out;
	byte *dst;
	const byte *src;
	const byte *end;

	*pic    = NULL;
	*width  = 0;
	*height = 0;

	if (bufSize < TGA_HEADER_SIZE)
	{
		return;
	}

	idLength        = buf[0];
	colorMapType    = buf[1];
	imageType       = buf[2];
	w               = (int)TGA_Read16(buf + 12);
	h               = (int)TGA_Read16(buf + 14);
	pixelDepth      = buf[16];
	imageDescriptor = buf[17];
	topLeft         = (imageDescriptor & 0x20) != 0;

	// Color-mapped images are not supported
	if (colorMapType != 0)
	{
		return;
	}

	// Only uncompressed (2) and RLE (10) true-color
	if (imageType != 2 && imageType != 10)
	{
		return;
	}

	// Only 24-bit (BGR) and 32-bit (BGRA)
	if (pixelDepth != 24 && pixelDepth != 32)
	{
		return;
	}

	if (w <= 0 || h <= 0)
	{
		return;
	}

	bytesPerPixel = pixelDepth / 8;
	numPixels     = w * h;
	out           = (byte *)malloc((size_t)numPixels * 4);
	if (!out)
	{
		return;
	}

	src = buf + TGA_HEADER_SIZE + idLength;
	end = buf + bufSize;
	dst = out;

	if (imageType == 2)
	{
		// Uncompressed: pixels stored linearly
		int i;

		for (i = 0; i < numPixels; i++)
		{
			if (src + bytesPerPixel > end)
			{
				free(out);
				return;
			}
			// TGA stores pixels as BGR(A); convert to RGBA
			dst[0] = src[2];
			dst[1] = src[1];
			dst[2] = src[0];
			dst[3] = (bytesPerPixel == 4) ? src[3] : 0xFF;
			src   += bytesPerPixel;
			dst   += 4;
		}
	}
	else
	{
		// RLE-compressed (type 10)
		int pixelsLeft = numPixels;

		while (pixelsLeft > 0 && src < end)
		{
			byte packetHeader = *src++;
			int  count        = (int)(packetHeader & 0x7F) + 1;

			if (packetHeader & 0x80)
			{
				// RLE packet: one pixel repeated count times
				byte r, g, b, a;
				int  i;

				if (src + bytesPerPixel > end)
				{
					free(out);
					return;
				}
				r    = src[2];
				g    = src[1];
				b    = src[0];
				a    = (bytesPerPixel == 4) ? src[3] : 0xFF;
				src += bytesPerPixel;

				for (i = 0; i < count && pixelsLeft > 0; i++, pixelsLeft--)
				{
					*dst++ = r;
					*dst++ = g;
					*dst++ = b;
					*dst++ = a;
				}
			}
			else
			{
				// Raw packet: count distinct pixels
				int i;

				for (i = 0; i < count && pixelsLeft > 0; i++, pixelsLeft--)
				{
					if (src + bytesPerPixel > end)
					{
						free(out);
						return;
					}
					*dst++ = src[2];
					*dst++ = src[1];
					*dst++ = src[0];
					*dst++ = (bytesPerPixel == 4) ? src[3] : 0xFF;
					src   += bytesPerPixel;
				}
			}
		}
	}

	// TGA origin: bit 5 of imageDescriptor = 0 means bottom-left; flip rows
	if (!topLeft)
	{
		int  rowBytes = w * 4;
		byte *rowBuf  = (byte *)malloc((size_t)rowBytes);

		if (rowBuf)
		{
			int row;

			for (row = 0; row < h / 2; row++)
			{
				byte *top    = out + row * rowBytes;
				byte *bottom = out + (h - 1 - row) * rowBytes;
				memcpy(rowBuf, top, (size_t)rowBytes);
				memcpy(top, bottom, (size_t)rowBytes);
				memcpy(bottom, rowBuf, (size_t)rowBytes);
			}
			free(rowBuf);
		}
	}

	*pic    = out;
	*width  = w;
	*height = h;
}

// ---------------------------------------------------------------------------
// JPEG decoding
// ---------------------------------------------------------------------------

/**
 * @brief Private error-manager for libjpeg – stores a setjmp recovery point.
 */
struct dx12JpegError
{
	struct jpeg_error_mgr pub;      ///< must be first member
	jmp_buf               jmpBuf;
};

/** @brief libjpeg error_exit callback: long-jump back to our recovery point. */
static void DX12_JpegErrorExit(j_common_ptr cinfo)
{
	struct dx12JpegError *err = (struct dx12JpegError *)cinfo->err;

	longjmp(err->jmpBuf, 1);
}

/**
 * @brief Decode a JPEG file in memory to RGBA.
 *
 * @param[in]  buf      Raw JPEG file bytes.
 * @param[in]  bufSize  Byte count.
 * @param[out] pic      malloc()-allocated RGBA output, or NULL on failure.
 * @param[out] width    Image width.
 * @param[out] height   Image height.
 */
static void DX12_DecodeJPG(const byte *buf, int bufSize,
                             byte **pic, int *width, int *height)
{
	struct jpeg_decompress_struct cinfo;
	struct dx12JpegError          jerr;
	int                           w, h, row;
	byte                         *out;

	*pic = NULL; *width = 0; *height = 0;

	if (!buf || bufSize <= 0)
	{
		return;
	}

	cinfo.err              = jpeg_std_error(&jerr.pub);
	jerr.pub.error_exit    = DX12_JpegErrorExit;

	if (setjmp(jerr.jmpBuf))
	{
		jpeg_destroy_decompress(&cinfo);
		return;
	}

	jpeg_create_decompress(&cinfo);
	jpeg_mem_src(&cinfo, (unsigned char *)buf, (unsigned long)bufSize);
	jpeg_read_header(&cinfo, TRUE);

	cinfo.out_color_space = JCS_RGB;
	jpeg_start_decompress(&cinfo);

	w = (int)cinfo.output_width;
	h = (int)cinfo.output_height;

	if (w <= 0 || h <= 0 || cinfo.output_components != 3)
	{
		jpeg_destroy_decompress(&cinfo);
		return;
	}

	out = (byte *)malloc((size_t)w * (size_t)h * 4);
	if (!out)
	{
		jpeg_destroy_decompress(&cinfo);
		return;
	}

	// Allocate a single-row scanline buffer via libjpeg's pool allocator
	{
		JSAMPARRAY rowbuf = (*cinfo.mem->alloc_sarray)(
			(j_common_ptr)&cinfo, JPOOL_IMAGE, (JDIMENSION)(w * 3), 1);

		row = 0;
		while (cinfo.output_scanline < cinfo.output_height)
		{
			byte *src;
			byte *dst;
			int   x;

			jpeg_read_scanlines(&cinfo, rowbuf, 1);
			src = (byte *)rowbuf[0];
			dst = out + (size_t)row * (size_t)w * 4;
			for (x = 0; x < w; x++)
			{
				dst[x * 4 + 0] = src[x * 3 + 0]; // R
				dst[x * 4 + 1] = src[x * 3 + 1]; // G
				dst[x * 4 + 2] = src[x * 3 + 2]; // B
				dst[x * 4 + 3] = 0xFF;            // A = opaque
			}
			row++;
		}
	}

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);

	*pic    = out;
	*width  = w;
	*height = h;
}

// ---------------------------------------------------------------------------
// PNG decoding – self-contained, no libpng dependency.
// Algorithm mirrors src/renderercommon/tr_image_png.c, adapted to use
// malloc/free and the puff() DEFLATE decompressor (puff.c is compiled into
// the same DX12 renderer target).
// ---------------------------------------------------------------------------

#ifndef INT_MAX
#define INT_MAX 0x1fffffff
#endif

// PNG specification constants (from the PNG spec / tr_image_png.c)
#define PNG_Signature      "\x89\x50\x4E\x47\xD\xA\x1A\xA"
#define PNG_Signature_Size (8)

struct PNG_ChunkHeader
{
	uint32_t Length;
	uint32_t Type;
};

#define PNG_ChunkHeader_Size (8)

typedef uint32_t PNG_ChunkCRC;

#define PNG_ChunkCRC_Size (4)

#define MAKE_CHUNKTYPE(a, b, c, d) (((a) << 24) | ((b) << 16) | ((c) << 8) | ((d)))

#define PNG_ChunkType_IHDR MAKE_CHUNKTYPE('I', 'H', 'D', 'R')
#define PNG_ChunkType_PLTE MAKE_CHUNKTYPE('P', 'L', 'T', 'E')
#define PNG_ChunkType_IDAT MAKE_CHUNKTYPE('I', 'D', 'A', 'T')
#define PNG_ChunkType_IEND MAKE_CHUNKTYPE('I', 'E', 'N', 'D')
#define PNG_ChunkType_tRNS MAKE_CHUNKTYPE('t', 'R', 'N', 'S')

struct PNG_Chunk_IHDR
{
	uint32_t Width;
	uint32_t Height;
	uint8_t  BitDepth;
	uint8_t  ColourType;
	uint8_t  CompressionMethod;
	uint8_t  FilterMethod;
	uint8_t  InterlaceMethod;
};

#define PNG_Chunk_IHDR_Size (13)

#define PNG_ColourType_Grey      (0)
#define PNG_ColourType_True      (2)
#define PNG_ColourType_Indexed   (3)
#define PNG_ColourType_GreyAlpha (4)
#define PNG_ColourType_TrueAlpha (6)

#define PNG_NumColourComponents_Grey      (1)
#define PNG_NumColourComponents_True      (3)
#define PNG_NumColourComponents_Indexed   (1)
#define PNG_NumColourComponents_GreyAlpha (2)
#define PNG_NumColourComponents_TrueAlpha (4)

#define PNG_BitDepth_1  (1)
#define PNG_BitDepth_2  (2)
#define PNG_BitDepth_4  (4)
#define PNG_BitDepth_8  (8)
#define PNG_BitDepth_16 (16)

#define PNG_CompressionMethod_0 (0)
#define PNG_FilterMethod_0      (0)

#define PNG_FilterType_None    (0)
#define PNG_FilterType_Sub     (1)
#define PNG_FilterType_Up      (2)
#define PNG_FilterType_Average (3)
#define PNG_FilterType_Paeth   (4)

#define PNG_InterlaceMethod_NonInterlaced (0)
#define PNG_InterlaceMethod_Interlaced    (1)

#define PNG_Adam7_NumPasses (7)

#define PNG_ZlibHeader_Size     (2)
#define PNG_ZlibCheckValue_Size (4)

// ---------------------------------------------------------------------------
// Buffered-file reader (wraps a raw in-memory byte buffer)
// ---------------------------------------------------------------------------

struct DX12_BufferedFile
{
	const byte *Buffer;
	int         Length;
	const byte *Ptr;
	int         BytesLeft;
};

static struct DX12_BufferedFile *DX12_OpenBufferedFile(const byte *data, int size)
{
	struct DX12_BufferedFile *BF;

	if (!data || size <= 0)
	{
		return NULL;
	}

	BF = (struct DX12_BufferedFile *)malloc(sizeof(struct DX12_BufferedFile));
	if (!BF)
	{
		return NULL;
	}

	BF->Buffer    = data;
	BF->Length    = size;
	BF->Ptr       = data;
	BF->BytesLeft = size;

	return BF;
}

static void DX12_CloseBufferedFile(struct DX12_BufferedFile *BF)
{
	if (BF)
	{
		free(BF);
	}
}

static const void *DX12_BufferedFileRead(struct DX12_BufferedFile *BF, unsigned Length)
{
	const void *RetVal;

	if (!(BF && Length))
	{
		return NULL;
	}

	if ((unsigned)BF->BytesLeft < Length)
	{
		return NULL;
	}

	RetVal         = BF->Ptr;
	BF->Ptr       += Length;
	BF->BytesLeft -= (int)Length;

	return RetVal;
}

static qboolean DX12_BufferedFileRewind(struct DX12_BufferedFile *BF, unsigned Offset)
{
	unsigned BytesRead;

	if (!BF)
	{
		return qfalse;
	}

	if (Offset == (unsigned) - 1)
	{
		BF->Ptr       = BF->Buffer;
		BF->BytesLeft = BF->Length;

		return qtrue;
	}

	BytesRead = (unsigned)(BF->Ptr - BF->Buffer);

	if (Offset > BytesRead)
	{
		return qfalse;
	}

	BF->Ptr       -= Offset;
	BF->BytesLeft += (int)Offset;

	return qtrue;
}

static qboolean DX12_BufferedFileSkip(struct DX12_BufferedFile *BF, unsigned Offset)
{
	if (!BF)
	{
		return qfalse;
	}

	if ((unsigned)BF->BytesLeft < Offset)
	{
		return qfalse;
	}

	BF->Ptr       += Offset;
	BF->BytesLeft -= (int)Offset;

	return qtrue;
}

// ---------------------------------------------------------------------------
// Chunk navigation
// ---------------------------------------------------------------------------

static qboolean DX12_FindChunk(struct DX12_BufferedFile *BF, uint32_t ChunkType)
{
	if (!BF)
	{
		return qfalse;
	}

	while (qtrue)
	{
		const struct PNG_ChunkHeader *CH;
		uint32_t                      Length;
		uint32_t                      Type;

		CH = (const struct PNG_ChunkHeader *)DX12_BufferedFileRead(BF, PNG_ChunkHeader_Size);
		if (!CH)
		{
			return qfalse;
		}

		Length = BigLong(CH->Length);
		Type   = BigLong(CH->Type);

		if (Type == ChunkType)
		{
			DX12_BufferedFileRewind(BF, PNG_ChunkHeader_Size);
			break;
		}
		else
		{
			if (Length)
			{
				if (!DX12_BufferedFileSkip(BF, Length + PNG_ChunkCRC_Size))
				{
					return qfalse;
				}
			}
		}
	}

	return qtrue;
}

// ---------------------------------------------------------------------------
// IDAT decompression
// ---------------------------------------------------------------------------

static uint32_t DX12_DecompressIDATs(struct DX12_BufferedFile *BF, uint8_t **Buffer)
{
	uint8_t                     *DecompressedData;
	uint8_t                     *CompressedData;
	uint8_t                     *CompressedDataPtr;
	uint32_t                     CompressedDataLength;
	const struct PNG_ChunkHeader *CH;
	uint32_t                     Length;
	uint32_t                     Type;
	int                          BytesToRewind;
	int32_t                      puffResult;
	uint8_t                     *puffDest;
	uint32_t                     puffDestLen;
	uint8_t                     *puffSrc;
	uint32_t                     puffSrcLen;

	if (!(BF && Buffer))
	{
		return 0;
	}

	DecompressedData     = NULL;
	*Buffer              = DecompressedData;
	CompressedData       = NULL;
	CompressedDataLength = 0;
	BytesToRewind        = 0;

	if (!DX12_FindChunk(BF, PNG_ChunkType_IDAT))
	{
		return 0;
	}

	// First pass: count total compressed bytes across all consecutive IDAT chunks.
	while (qtrue)
	{
		CH = (const struct PNG_ChunkHeader *)DX12_BufferedFileRead(BF, PNG_ChunkHeader_Size);
		if (!CH)
		{
			DX12_BufferedFileRewind(BF, (unsigned)BytesToRewind);

			return 0;
		}

		Length = BigLong(CH->Length);
		Type   = BigLong(CH->Type);

		if (!(Type == PNG_ChunkType_IDAT))
		{
			DX12_BufferedFileRewind(BF, PNG_ChunkHeader_Size);

			break;
		}

		BytesToRewind += PNG_ChunkHeader_Size;

		if (Length)
		{
			if (!DX12_BufferedFileSkip(BF, Length + PNG_ChunkCRC_Size))
			{
				DX12_BufferedFileRewind(BF, (unsigned)BytesToRewind);

				return 0;
			}

			BytesToRewind        += (int)(Length + PNG_ChunkCRC_Size);
			CompressedDataLength += Length;
		}
	}

	DX12_BufferedFileRewind(BF, (unsigned)BytesToRewind);

	CompressedData = (uint8_t *)malloc(CompressedDataLength);
	if (!CompressedData)
	{
		return 0;
	}

	CompressedDataPtr = CompressedData;

	// Second pass: collect the compressed bytes.
	while (qtrue)
	{
		const uint8_t *OrigCompressedData;

		CH = (const struct PNG_ChunkHeader *)DX12_BufferedFileRead(BF, PNG_ChunkHeader_Size);
		if (!CH)
		{
			free(CompressedData);

			return 0;
		}

		Length = BigLong(CH->Length);
		Type   = BigLong(CH->Type);

		if (!(Type == PNG_ChunkType_IDAT))
		{
			DX12_BufferedFileRewind(BF, PNG_ChunkHeader_Size);

			break;
		}

		if (Length)
		{
			OrigCompressedData = (const uint8_t *)DX12_BufferedFileRead(BF, Length);
			if (!OrigCompressedData)
			{
				free(CompressedData);

				return 0;
			}

			if (!DX12_BufferedFileSkip(BF, PNG_ChunkCRC_Size))
			{
				free(CompressedData);

				return 0;
			}

			memcpy(CompressedDataPtr, OrigCompressedData, Length);
			CompressedDataPtr += Length;
		}
	}

	// Skip the 2-byte zlib header and 4-byte Adler-32 check value.
	puffDest    = NULL;
	puffDestLen = 0;
	puffSrc     = CompressedData + PNG_ZlibHeader_Size;
	puffSrcLen  = CompressedDataLength - PNG_ZlibHeader_Size - PNG_ZlibCheckValue_Size;

	// First puff() call: calculate the decompressed size.
	puffResult = puff(puffDest, &puffDestLen, puffSrc, &puffSrcLen);
	if (!((puffResult == 0) && (puffDestLen > 0)))
	{
		free(CompressedData);

		return 0;
	}

	DecompressedData = (uint8_t *)malloc(puffDestLen);
	if (!DecompressedData)
	{
		free(CompressedData);

		return 0;
	}

	// Second puff() call: decompress into the allocated buffer.
	puffDest   = DecompressedData;
	puffSrc    = CompressedData + PNG_ZlibHeader_Size;
	puffSrcLen = CompressedDataLength - PNG_ZlibHeader_Size - PNG_ZlibCheckValue_Size;

	puffResult = puff(puffDest, &puffDestLen, puffSrc, &puffSrcLen);

	free(CompressedData);

	if (!((puffResult == 0) && (puffDestLen > 0)))
	{
		free(DecompressedData);

		return 0;
	}

	*Buffer = DecompressedData;

	return puffDestLen;
}

// ---------------------------------------------------------------------------
// PNG filters
// ---------------------------------------------------------------------------

static uint8_t DX12_PredictPaeth(uint8_t a, uint8_t b, uint8_t c)
{
	int p  = ((int)a) + ((int)b) - ((int)c);
	int pa = abs(p - ((int)a));
	int pb = abs(p - ((int)b));
	int pc = abs(p - ((int)c));

	if ((pa <= pb) && (pa <= pc))
	{
		return a;
	}
	else if (pb <= pc)
	{
		return b;
	}
	else
	{
		return c;
	}
}

static qboolean DX12_UnfilterImage(uint8_t *DecompressedData,
                                   uint32_t ImageHeight,
                                   uint32_t BytesPerScanline,
                                   uint32_t BytesPerPixel)
{
	uint8_t  *DecompPtr;
	uint8_t  FilterType;
	uint8_t  *PixelLeft, *PixelUp, *PixelUpLeft;
	uint32_t w, h, p;
	uint8_t  Zeros[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	if (!(DecompressedData && BytesPerPixel))
	{
		return qfalse;
	}

	if ((!ImageHeight) || (!BytesPerScanline))
	{
		return qtrue;
	}

	DecompPtr = DecompressedData;

	for (h = 0; h < ImageHeight; h++)
	{
		FilterType = *DecompPtr;
		DecompPtr++;

		PixelLeft = Zeros;

		if (h > 0)
		{
			PixelUp = DecompPtr - (BytesPerScanline + 1);
		}
		else
		{
			PixelUp = Zeros;
		}

		PixelUpLeft = Zeros;

		for (w = 0; w < (BytesPerScanline / BytesPerPixel); w++)
		{
			for (p = 0; p < BytesPerPixel; p++)
			{
				switch (FilterType)
				{
				case PNG_FilterType_None:
					break;

				case PNG_FilterType_Sub:
					DecompPtr[p] += PixelLeft[p];

					break;

				case PNG_FilterType_Up:
					DecompPtr[p] += PixelUp[p];

					break;

				case PNG_FilterType_Average:
					DecompPtr[p] += (uint8_t)((((uint16_t)PixelLeft[p]) + ((uint16_t)PixelUp[p])) / 2);

					break;

				case PNG_FilterType_Paeth:
					DecompPtr[p] += DX12_PredictPaeth(PixelLeft[p], PixelUp[p], PixelUpLeft[p]);

					break;

				default:
					return qfalse;
				}
			}

			PixelLeft = DecompPtr;

			if (h > 0)
			{
				PixelUpLeft = DecompPtr - (BytesPerScanline + 1);
			}

			DecompPtr += BytesPerPixel;

			if (h > 0)
			{
				PixelUp = DecompPtr - (BytesPerScanline + 1);
			}
		}
	}

	return qtrue;
}

// ---------------------------------------------------------------------------
// Pixel format conversion
// ---------------------------------------------------------------------------

static qboolean DX12_ConvertPixel(struct PNG_Chunk_IHDR *IHDR,
                                  byte *OutPtr,
                                  uint8_t *DecompPtr,
                                  qboolean HasTransparentColour,
                                  uint8_t *TransparentColour,
                                  uint8_t *OutPal)
{
	if (!(IHDR && OutPtr && DecompPtr && TransparentColour && OutPal))
	{
		return qfalse;
	}

	switch (IHDR->ColourType)
	{
	case PNG_ColourType_Grey:
	{
		switch (IHDR->BitDepth)
		{
		case PNG_BitDepth_1:
		case PNG_BitDepth_2:
		case PNG_BitDepth_4:
		{
			uint8_t Step      = (uint8_t)(0xFF / ((1 << IHDR->BitDepth) - 1));
			uint8_t GreyValue = (uint8_t)(DecompPtr[0] * Step);

			OutPtr[0] = GreyValue;
			OutPtr[1] = GreyValue;
			OutPtr[2] = GreyValue;
			OutPtr[3] = 0xFF;

			if (HasTransparentColour && TransparentColour[1] == DecompPtr[0])
			{
				OutPtr[3] = 0x00;
			}

			break;
		}

		case PNG_BitDepth_8:
		case PNG_BitDepth_16:
		{
			OutPtr[0] = DecompPtr[0];
			OutPtr[1] = DecompPtr[0];
			OutPtr[2] = DecompPtr[0];
			OutPtr[3] = 0xFF;

			if (HasTransparentColour)
			{
				if (IHDR->BitDepth == PNG_BitDepth_8)
				{
					if (TransparentColour[1] == DecompPtr[0])
					{
						OutPtr[3] = 0x00;
					}
				}
				else
				{
					if ((TransparentColour[0] == DecompPtr[0]) && (TransparentColour[1] == DecompPtr[1]))
					{
						OutPtr[3] = 0x00;
					}
				}
			}

			break;
		}

		default:
			return qfalse;
		}

		break;
	}

	case PNG_ColourType_True:
	{
		switch (IHDR->BitDepth)
		{
		case PNG_BitDepth_8:
		{
			OutPtr[0] = DecompPtr[0];
			OutPtr[1] = DecompPtr[1];
			OutPtr[2] = DecompPtr[2];
			OutPtr[3] = 0xFF;

			if (HasTransparentColour)
			{
				if ((TransparentColour[1] == DecompPtr[0]) &&
				    (TransparentColour[3] == DecompPtr[1]) &&
				    (TransparentColour[5] == DecompPtr[2]))
				{
					OutPtr[3] = 0x00;
				}
			}

			break;
		}

		case PNG_BitDepth_16:
		{
			// Use only the upper byte of each 16-bit channel
			OutPtr[0] = DecompPtr[0];
			OutPtr[1] = DecompPtr[2];
			OutPtr[2] = DecompPtr[4];
			OutPtr[3] = 0xFF;

			if (HasTransparentColour)
			{
				if ((TransparentColour[0] == DecompPtr[0]) && (TransparentColour[1] == DecompPtr[1]) &&
				    (TransparentColour[2] == DecompPtr[2]) && (TransparentColour[3] == DecompPtr[3]) &&
				    (TransparentColour[4] == DecompPtr[4]) && (TransparentColour[5] == DecompPtr[5]))
				{
					OutPtr[3] = 0x00;
				}
			}

			break;
		}

		default:
			return qfalse;
		}

		break;
	}

	case PNG_ColourType_Indexed:
	{
		OutPtr[0] = OutPal[DecompPtr[0] * Q3IMAGE_BYTESPERPIXEL + 0];
		OutPtr[1] = OutPal[DecompPtr[0] * Q3IMAGE_BYTESPERPIXEL + 1];
		OutPtr[2] = OutPal[DecompPtr[0] * Q3IMAGE_BYTESPERPIXEL + 2];
		OutPtr[3] = OutPal[DecompPtr[0] * Q3IMAGE_BYTESPERPIXEL + 3];

		break;
	}

	case PNG_ColourType_GreyAlpha:
	{
		switch (IHDR->BitDepth)
		{
		case PNG_BitDepth_8:
		{
			OutPtr[0] = DecompPtr[0];
			OutPtr[1] = DecompPtr[0];
			OutPtr[2] = DecompPtr[0];
			OutPtr[3] = DecompPtr[1];

			break;
		}

		case PNG_BitDepth_16:
		{
			// Use only the upper byte of each 16-bit channel
			OutPtr[0] = DecompPtr[0];
			OutPtr[1] = DecompPtr[0];
			OutPtr[2] = DecompPtr[0];
			OutPtr[3] = DecompPtr[2];

			break;
		}

		default:
			return qfalse;
		}

		break;
	}

	case PNG_ColourType_TrueAlpha:
	{
		switch (IHDR->BitDepth)
		{
		case PNG_BitDepth_8:
		{
			OutPtr[0] = DecompPtr[0];
			OutPtr[1] = DecompPtr[1];
			OutPtr[2] = DecompPtr[2];
			OutPtr[3] = DecompPtr[3];

			break;
		}

		case PNG_BitDepth_16:
		{
			// Use only the upper byte of each 16-bit channel
			OutPtr[0] = DecompPtr[0];
			OutPtr[1] = DecompPtr[2];
			OutPtr[2] = DecompPtr[4];
			OutPtr[3] = DecompPtr[6];

			break;
		}

		default:
			return qfalse;
		}

		break;
	}

	default:
		return qfalse;
	}

	return qtrue;
}

// ---------------------------------------------------------------------------
// Image decode – non-interlaced
// ---------------------------------------------------------------------------

static qboolean DX12_DecodeNonInterlaced(struct PNG_Chunk_IHDR *IHDR,
                                         byte *OutBuffer,
                                         uint8_t *DecompressedData,
                                         uint32_t DecompressedDataLength,
                                         qboolean HasTransparentColour,
                                         uint8_t *TransparentColour,
                                         uint8_t *OutPal)
{
	uint32_t IHDR_Width;
	uint32_t IHDR_Height;
	uint32_t BytesPerScanline, BytesPerPixel, PixelsPerByte;
	uint32_t w, h, p;
	byte     *OutPtr;
	uint8_t  *DecompPtr;

	if (!(IHDR && OutBuffer && DecompressedData && DecompressedDataLength && TransparentColour && OutPal))
	{
		return qfalse;
	}

	IHDR_Width  = BigLong(IHDR->Width);
	IHDR_Height = BigLong(IHDR->Height);

	switch (IHDR->ColourType)
	{
	case PNG_ColourType_Grey:
	{
		switch (IHDR->BitDepth)
		{
		case PNG_BitDepth_1:
		case PNG_BitDepth_2:
		case PNG_BitDepth_4:
			BytesPerPixel = 1;
			PixelsPerByte = 8 / IHDR->BitDepth;

			break;

		case PNG_BitDepth_8:
		case PNG_BitDepth_16:
			BytesPerPixel = (IHDR->BitDepth / 8) * PNG_NumColourComponents_Grey;
			PixelsPerByte = 1;

			break;

		default:
			return qfalse;
		}

		break;
	}

	case PNG_ColourType_True:
	{
		switch (IHDR->BitDepth)
		{
		case PNG_BitDepth_8:
		case PNG_BitDepth_16:
			BytesPerPixel = (IHDR->BitDepth / 8) * PNG_NumColourComponents_True;
			PixelsPerByte = 1;

			break;

		default:
			return qfalse;
		}

		break;
	}

	case PNG_ColourType_Indexed:
	{
		switch (IHDR->BitDepth)
		{
		case PNG_BitDepth_1:
		case PNG_BitDepth_2:
		case PNG_BitDepth_4:
			BytesPerPixel = 1;
			PixelsPerByte = 8 / IHDR->BitDepth;

			break;

		case PNG_BitDepth_8:
			BytesPerPixel = PNG_NumColourComponents_Indexed;
			PixelsPerByte = 1;

			break;

		default:
			return qfalse;
		}

		break;
	}

	case PNG_ColourType_GreyAlpha:
	{
		switch (IHDR->BitDepth)
		{
		case PNG_BitDepth_8:
		case PNG_BitDepth_16:
			BytesPerPixel = (IHDR->BitDepth / 8) * PNG_NumColourComponents_GreyAlpha;
			PixelsPerByte = 1;

			break;

		default:
			return qfalse;
		}

		break;
	}

	case PNG_ColourType_TrueAlpha:
	{
		switch (IHDR->BitDepth)
		{
		case PNG_BitDepth_8:
		case PNG_BitDepth_16:
			BytesPerPixel = (IHDR->BitDepth / 8) * PNG_NumColourComponents_TrueAlpha;
			PixelsPerByte = 1;

			break;

		default:
			return qfalse;
		}

		break;
	}

	default:
		return qfalse;
	}

	BytesPerScanline = (IHDR_Width * BytesPerPixel + (PixelsPerByte - 1)) / PixelsPerByte;

	if (!(DecompressedDataLength == ((BytesPerScanline + 1) * IHDR_Height)))
	{
		return qfalse;
	}

	if (!DX12_UnfilterImage(DecompressedData, IHDR_Height, BytesPerScanline, BytesPerPixel))
	{
		return qfalse;
	}

	OutPtr    = OutBuffer;
	DecompPtr = DecompressedData;

	for (h = 0; h < IHDR_Height; h++)
	{
		uint32_t CurrPixel;

		DecompPtr++;  // skip FilterType byte
		CurrPixel = 0;

		for (w = 0; w < (BytesPerScanline / BytesPerPixel); w++)
		{
			if (PixelsPerByte > 1)
			{
				for (p = 0; p < PixelsPerByte; p++)
				{
					if (CurrPixel < IHDR_Width)
					{
						uint8_t Mask        = (uint8_t)((1 << IHDR->BitDepth) - 1);
						uint32_t Shift      = (PixelsPerByte - 1 - p) * IHDR->BitDepth;
						uint8_t SinglePixel = (uint8_t)((DecompPtr[0] & (Mask << Shift)) >> Shift);

						if (!DX12_ConvertPixel(IHDR, OutPtr, &SinglePixel, HasTransparentColour, TransparentColour, OutPal))
						{
							return qfalse;
						}

						OutPtr += Q3IMAGE_BYTESPERPIXEL;
						CurrPixel++;
					}
				}
			}
			else
			{
				if (!DX12_ConvertPixel(IHDR, OutPtr, DecompPtr, HasTransparentColour, TransparentColour, OutPal))
				{
					return qfalse;
				}

				OutPtr += Q3IMAGE_BYTESPERPIXEL;
			}

			DecompPtr += BytesPerPixel;
		}
	}

	return qtrue;
}

// ---------------------------------------------------------------------------
// Image decode – Adam7 interlaced
// ---------------------------------------------------------------------------

static qboolean DX12_DecodeInterlaced(struct PNG_Chunk_IHDR *IHDR,
                                      byte *OutBuffer,
                                      uint8_t *DecompressedData,
                                      uint32_t DecompressedDataLength,
                                      qboolean HasTransparentColour,
                                      uint8_t *TransparentColour,
                                      uint8_t *OutPal)
{
	uint32_t IHDR_Width;
	uint32_t IHDR_Height;
	uint32_t BytesPerScanline[PNG_Adam7_NumPasses], BytesPerPixel, PixelsPerByte;
	uint32_t PassWidth[PNG_Adam7_NumPasses], PassHeight[PNG_Adam7_NumPasses];
	uint32_t WSkip[PNG_Adam7_NumPasses], WOffset[PNG_Adam7_NumPasses];
	uint32_t HSkip[PNG_Adam7_NumPasses], HOffset[PNG_Adam7_NumPasses];
	uint32_t w, h, p, a;
	byte     *OutPtr;
	uint8_t  *DecompPtr;
	uint32_t TargetLength;

	if (!(IHDR && OutBuffer && DecompressedData && DecompressedDataLength && TransparentColour && OutPal))
	{
		return qfalse;
	}

	IHDR_Width  = BigLong(IHDR->Width);
	IHDR_Height = BigLong(IHDR->Height);

	// Adam7 pass skip/offset tables
	WSkip[0] = 8; WOffset[0] = 0; HSkip[0] = 8; HOffset[0] = 0;
	WSkip[1] = 8; WOffset[1] = 4; HSkip[1] = 8; HOffset[1] = 0;
	WSkip[2] = 4; WOffset[2] = 0; HSkip[2] = 8; HOffset[2] = 4;
	WSkip[3] = 4; WOffset[3] = 2; HSkip[3] = 4; HOffset[3] = 0;
	WSkip[4] = 2; WOffset[4] = 0; HSkip[4] = 4; HOffset[4] = 2;
	WSkip[5] = 2; WOffset[5] = 1; HSkip[5] = 2; HOffset[5] = 0;
	WSkip[6] = 1; WOffset[6] = 0; HSkip[6] = 2; HOffset[6] = 1;

	// Pass dimensions
	PassWidth[0]  = (IHDR_Width  + 7) / 8; PassHeight[0] = (IHDR_Height + 7) / 8;
	PassWidth[1]  = (IHDR_Width  + 3) / 8; PassHeight[1] = (IHDR_Height + 7) / 8;
	PassWidth[2]  = (IHDR_Width  + 3) / 4; PassHeight[2] = (IHDR_Height + 3) / 8;
	PassWidth[3]  = (IHDR_Width  + 1) / 4; PassHeight[3] = (IHDR_Height + 3) / 4;
	PassWidth[4]  = (IHDR_Width  + 1) / 2; PassHeight[4] = (IHDR_Height + 1) / 4;
	PassWidth[5]  = (IHDR_Width  + 0) / 2; PassHeight[5] = (IHDR_Height + 1) / 2;
	PassWidth[6]  = (IHDR_Width  + 0) / 1; PassHeight[6] = (IHDR_Height + 0) / 2;

	switch (IHDR->ColourType)
	{
	case PNG_ColourType_Grey:
	{
		switch (IHDR->BitDepth)
		{
		case PNG_BitDepth_1:
		case PNG_BitDepth_2:
		case PNG_BitDepth_4:
			BytesPerPixel = 1;
			PixelsPerByte = 8 / IHDR->BitDepth;

			break;

		case PNG_BitDepth_8:
		case PNG_BitDepth_16:
			BytesPerPixel = (IHDR->BitDepth / 8) * PNG_NumColourComponents_Grey;
			PixelsPerByte = 1;

			break;

		default:
			return qfalse;
		}

		break;
	}

	case PNG_ColourType_True:
	{
		switch (IHDR->BitDepth)
		{
		case PNG_BitDepth_8:
		case PNG_BitDepth_16:
			BytesPerPixel = (IHDR->BitDepth / 8) * PNG_NumColourComponents_True;
			PixelsPerByte = 1;

			break;

		default:
			return qfalse;
		}

		break;
	}

	case PNG_ColourType_Indexed:
	{
		switch (IHDR->BitDepth)
		{
		case PNG_BitDepth_1:
		case PNG_BitDepth_2:
		case PNG_BitDepth_4:
			BytesPerPixel = 1;
			PixelsPerByte = 8 / IHDR->BitDepth;

			break;

		case PNG_BitDepth_8:
			BytesPerPixel = PNG_NumColourComponents_Indexed;
			PixelsPerByte = 1;

			break;

		default:
			return qfalse;
		}

		break;
	}

	case PNG_ColourType_GreyAlpha:
	{
		switch (IHDR->BitDepth)
		{
		case PNG_BitDepth_8:
		case PNG_BitDepth_16:
			BytesPerPixel = (IHDR->BitDepth / 8) * PNG_NumColourComponents_GreyAlpha;
			PixelsPerByte = 1;

			break;

		default:
			return qfalse;
		}

		break;
	}

	case PNG_ColourType_TrueAlpha:
	{
		switch (IHDR->BitDepth)
		{
		case PNG_BitDepth_8:
		case PNG_BitDepth_16:
			BytesPerPixel = (IHDR->BitDepth / 8) * PNG_NumColourComponents_TrueAlpha;
			PixelsPerByte = 1;

			break;

		default:
			return qfalse;
		}

		break;
	}

	default:
		return qfalse;
	}

	for (a = 0; a < PNG_Adam7_NumPasses; a++)
	{
		BytesPerScanline[a] = (PassWidth[a] * BytesPerPixel + (PixelsPerByte - 1)) / PixelsPerByte;
	}

	TargetLength = 0;

	for (a = 0; a < PNG_Adam7_NumPasses; a++)
	{
		TargetLength += ((BytesPerScanline[a] + (BytesPerScanline[a] ? 1 : 0)) * PassHeight[a]);
	}

	if (!(DecompressedDataLength == TargetLength))
	{
		return qfalse;
	}

	DecompPtr = DecompressedData;

	for (a = 0; a < PNG_Adam7_NumPasses; a++)
	{
		if (!DX12_UnfilterImage(DecompPtr, PassHeight[a], BytesPerScanline[a], BytesPerPixel))
		{
			return qfalse;
		}

		DecompPtr += ((BytesPerScanline[a] + (BytesPerScanline[a] ? 1 : 0)) * PassHeight[a]);
	}

	DecompPtr = DecompressedData;

	for (a = 0; a < PNG_Adam7_NumPasses; a++)
	{
		for (h = 0; h < PassHeight[a]; h++)
		{
			uint32_t CurrPixel;

			if (BytesPerScanline[a])
			{
				DecompPtr++;  // skip FilterType byte
			}

			CurrPixel = 0;

			for (w = 0; w < (BytesPerScanline[a] / BytesPerPixel); w++)
			{
				if (PixelsPerByte > 1)
				{
					for (p = 0; p < PixelsPerByte; p++)
					{
						if (CurrPixel < PassWidth[a])
						{
							uint8_t  Mask        = (uint8_t)((1 << IHDR->BitDepth) - 1);
							uint32_t Shift       = (PixelsPerByte - 1 - p) * IHDR->BitDepth;
							uint8_t  SinglePixel = (uint8_t)((DecompPtr[0] & (Mask << Shift)) >> Shift);

							OutPtr = OutBuffer + (((((h * HSkip[a]) + HOffset[a]) * IHDR_Width) + ((CurrPixel * WSkip[a]) + WOffset[a])) * Q3IMAGE_BYTESPERPIXEL);

							if (!DX12_ConvertPixel(IHDR, OutPtr, &SinglePixel, HasTransparentColour, TransparentColour, OutPal))
							{
								return qfalse;
							}

							CurrPixel++;
						}
					}
				}
				else
				{
					OutPtr = OutBuffer + (((((h * HSkip[a]) + HOffset[a]) * IHDR_Width) + ((w * WSkip[a]) + WOffset[a])) * Q3IMAGE_BYTESPERPIXEL);

					if (!DX12_ConvertPixel(IHDR, OutPtr, DecompPtr, HasTransparentColour, TransparentColour, OutPal))
					{
						return qfalse;
					}
				}

				DecompPtr += BytesPerPixel;
			}
		}
	}

	return qtrue;
}

// ---------------------------------------------------------------------------
// Top-level PNG decode entry point
// ---------------------------------------------------------------------------

/**
 * @brief Decode a PNG file in memory to RGBA.
 *
 * Self-contained: uses puff() for DEFLATE, no libpng dependency.
 * Mirrors the algorithm in src/renderercommon/tr_image_png.c.
 *
 * @param[in]  buf      Raw PNG file bytes.
 * @param[in]  bufSize  Byte count.
 * @param[out] pic      malloc()-allocated RGBA output, or NULL on failure.
 * @param[out] width    Image width.
 * @param[out] height   Image height.
 */
static void DX12_DecodePNG(const byte *buf, int bufSize,
                            byte **pic, int *width, int *height)
{
	struct DX12_BufferedFile   *ThePNG;
	byte                       *OutBuffer;
	const uint8_t              *Signature;
	const struct PNG_ChunkHeader *CH;
	uint32_t                    ChunkHeaderLength;
	uint32_t                    ChunkHeaderType;
	struct PNG_Chunk_IHDR      *IHDR;
	uint32_t                    IHDR_Width;
	uint32_t                    IHDR_Height;
	const PNG_ChunkCRC         *CRC;
	uint8_t                    *InPal;
	uint8_t                    *DecompressedData;
	uint32_t                    DecompressedDataLength;
	uint32_t                    i;

	uint8_t OutPal[1024];  // palette: 256 RGBA entries

	qboolean HasTransparentColour = qfalse;
	uint8_t  TransparentColour[6] = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };

	*pic    = NULL;
	*width  = 0;
	*height = 0;

	if (!buf || bufSize <= 0)
	{
		return;
	}

	ThePNG = DX12_OpenBufferedFile(buf, bufSize);
	if (!ThePNG)
	{
		return;
	}

	// Verify PNG signature
	Signature = (const uint8_t *)DX12_BufferedFileRead(ThePNG, PNG_Signature_Size);
	if (!Signature)
	{
		DX12_CloseBufferedFile(ThePNG);

		return;
	}

	if (memcmp(Signature, PNG_Signature, PNG_Signature_Size))
	{
		DX12_CloseBufferedFile(ThePNG);

		return;
	}

	// Read the first chunk header – must be IHDR
	CH = (const struct PNG_ChunkHeader *)DX12_BufferedFileRead(ThePNG, PNG_ChunkHeader_Size);
	if (!CH)
	{
		DX12_CloseBufferedFile(ThePNG);

		return;
	}

	ChunkHeaderLength = BigLong(CH->Length);
	ChunkHeaderType   = BigLong(CH->Type);

	if (!((ChunkHeaderType == PNG_ChunkType_IHDR) && (ChunkHeaderLength == PNG_Chunk_IHDR_Size)))
	{
		DX12_CloseBufferedFile(ThePNG);

		return;
	}

	IHDR = (struct PNG_Chunk_IHDR *)DX12_BufferedFileRead(ThePNG, PNG_Chunk_IHDR_Size);
	if (!IHDR)
	{
		DX12_CloseBufferedFile(ThePNG);

		return;
	}

	CRC = (const PNG_ChunkCRC *)DX12_BufferedFileRead(ThePNG, PNG_ChunkCRC_Size);
	if (!CRC)
	{
		DX12_CloseBufferedFile(ThePNG);

		return;
	}

	IHDR_Width  = BigLong(IHDR->Width);
	IHDR_Height = BigLong(IHDR->Height);

	if (!((IHDR_Width > 0) && (IHDR_Height > 0)) ||
	    IHDR_Width > (uint32_t)INT_MAX / Q3IMAGE_BYTESPERPIXEL / IHDR_Height)
	{
		DX12_CloseBufferedFile(ThePNG);

		return;
	}

	if (!((IHDR->CompressionMethod == PNG_CompressionMethod_0) &&
	      (IHDR->FilterMethod      == PNG_FilterMethod_0)))
	{
		DX12_CloseBufferedFile(ThePNG);

		return;
	}

	if (!((IHDR->InterlaceMethod == PNG_InterlaceMethod_NonInterlaced) ||
	      (IHDR->InterlaceMethod == PNG_InterlaceMethod_Interlaced)))
	{
		DX12_CloseBufferedFile(ThePNG);

		return;
	}

	// Read palette for indexed images
	if (IHDR->ColourType == PNG_ColourType_Indexed)
	{
		if (!DX12_FindChunk(ThePNG, PNG_ChunkType_PLTE))
		{
			DX12_CloseBufferedFile(ThePNG);

			return;
		}

		CH = (const struct PNG_ChunkHeader *)DX12_BufferedFileRead(ThePNG, PNG_ChunkHeader_Size);
		if (!CH)
		{
			DX12_CloseBufferedFile(ThePNG);

			return;
		}

		ChunkHeaderLength = BigLong(CH->Length);
		ChunkHeaderType   = BigLong(CH->Type);

		if (!(ChunkHeaderType == PNG_ChunkType_PLTE))
		{
			DX12_CloseBufferedFile(ThePNG);

			return;
		}

		if (ChunkHeaderLength % 3)
		{
			DX12_CloseBufferedFile(ThePNG);

			return;
		}

		InPal = (uint8_t *)DX12_BufferedFileRead(ThePNG, ChunkHeaderLength);
		if (!InPal)
		{
			DX12_CloseBufferedFile(ThePNG);

			return;
		}

		CRC = (const PNG_ChunkCRC *)DX12_BufferedFileRead(ThePNG, PNG_ChunkCRC_Size);
		if (!CRC)
		{
			DX12_CloseBufferedFile(ThePNG);

			return;
		}

		for (i = 0; i < 256; i++)
		{
			OutPal[i * Q3IMAGE_BYTESPERPIXEL + 0] = 0x00;
			OutPal[i * Q3IMAGE_BYTESPERPIXEL + 1] = 0x00;
			OutPal[i * Q3IMAGE_BYTESPERPIXEL + 2] = 0x00;
			OutPal[i * Q3IMAGE_BYTESPERPIXEL + 3] = 0xFF;
		}

		for (i = 0; i < (ChunkHeaderLength / 3); i++)
		{
			OutPal[i * Q3IMAGE_BYTESPERPIXEL + 0] = InPal[i * 3 + 0];
			OutPal[i * Q3IMAGE_BYTESPERPIXEL + 1] = InPal[i * 3 + 1];
			OutPal[i * Q3IMAGE_BYTESPERPIXEL + 2] = InPal[i * 3 + 2];
			OutPal[i * Q3IMAGE_BYTESPERPIXEL + 3] = 0xFF;
		}
	}

	// Optional tRNS chunk (transparency)
	if (DX12_FindChunk(ThePNG, PNG_ChunkType_tRNS))
	{
		uint8_t *Trans;

		CH = (const struct PNG_ChunkHeader *)DX12_BufferedFileRead(ThePNG, PNG_ChunkHeader_Size);
		if (!CH)
		{
			DX12_CloseBufferedFile(ThePNG);

			return;
		}

		ChunkHeaderLength = BigLong(CH->Length);
		ChunkHeaderType   = BigLong(CH->Type);

		if (!(ChunkHeaderType == PNG_ChunkType_tRNS))
		{
			DX12_CloseBufferedFile(ThePNG);

			return;
		}

		Trans = (uint8_t *)DX12_BufferedFileRead(ThePNG, ChunkHeaderLength);
		if (!Trans)
		{
			DX12_CloseBufferedFile(ThePNG);

			return;
		}

		CRC = (const PNG_ChunkCRC *)DX12_BufferedFileRead(ThePNG, PNG_ChunkCRC_Size);
		if (!CRC)
		{
			DX12_CloseBufferedFile(ThePNG);

			return;
		}

		switch (IHDR->ColourType)
		{
		case PNG_ColourType_Grey:
		{
			if (ChunkHeaderLength != 2)
			{
				DX12_CloseBufferedFile(ThePNG);

				return;
			}

			HasTransparentColour = qtrue;
			TransparentColour[0] = Trans[0];
			TransparentColour[1] = Trans[1];

			break;
		}

		case PNG_ColourType_True:
		{
			if (ChunkHeaderLength != 6)
			{
				DX12_CloseBufferedFile(ThePNG);

				return;
			}

			HasTransparentColour = qtrue;
			TransparentColour[0] = Trans[0];
			TransparentColour[1] = Trans[1];
			TransparentColour[2] = Trans[2];
			TransparentColour[3] = Trans[3];
			TransparentColour[4] = Trans[4];
			TransparentColour[5] = Trans[5];

			break;
		}

		case PNG_ColourType_Indexed:
		{
			if (ChunkHeaderLength > 256)
			{
				DX12_CloseBufferedFile(ThePNG);

				return;
			}

			HasTransparentColour = qtrue;

			for (i = 0; i < ChunkHeaderLength; i++)
			{
				OutPal[i * Q3IMAGE_BYTESPERPIXEL + 3] = Trans[i];
			}

			break;
		}

		default:
		{
			DX12_CloseBufferedFile(ThePNG);

			return;
		}
		}
	}

	// Rewind to just after the signature so DecompressIDATs can scan from there
	if (!DX12_BufferedFileRewind(ThePNG, (unsigned) - 1))
	{
		DX12_CloseBufferedFile(ThePNG);

		return;
	}

	if (!DX12_BufferedFileSkip(ThePNG, PNG_Signature_Size))
	{
		DX12_CloseBufferedFile(ThePNG);

		return;
	}

	// Decompress all IDAT chunks
	DecompressedDataLength = DX12_DecompressIDATs(ThePNG, &DecompressedData);
	if (!(DecompressedDataLength && DecompressedData))
	{
		DX12_CloseBufferedFile(ThePNG);

		return;
	}

	// Allocate output RGBA buffer
	OutBuffer = (byte *)malloc((size_t)IHDR_Width * (size_t)IHDR_Height * Q3IMAGE_BYTESPERPIXEL);
	if (!OutBuffer)
	{
		free(DecompressedData);
		DX12_CloseBufferedFile(ThePNG);

		return;
	}

	switch (IHDR->InterlaceMethod)
	{
	case PNG_InterlaceMethod_NonInterlaced:
	{
		if (!DX12_DecodeNonInterlaced(IHDR, OutBuffer, DecompressedData, DecompressedDataLength,
		                              HasTransparentColour, TransparentColour, OutPal))
		{
			free(OutBuffer);
			free(DecompressedData);
			DX12_CloseBufferedFile(ThePNG);

			return;
		}

		break;
	}

	case PNG_InterlaceMethod_Interlaced:
	{
		if (!DX12_DecodeInterlaced(IHDR, OutBuffer, DecompressedData, DecompressedDataLength,
		                           HasTransparentColour, TransparentColour, OutPal))
		{
			free(OutBuffer);
			free(DecompressedData);
			DX12_CloseBufferedFile(ThePNG);

			return;
		}

		break;
	}

	default:
	{
		free(OutBuffer);
		free(DecompressedData);
		DX12_CloseBufferedFile(ThePNG);

		return;
	}
	}

	*pic    = OutBuffer;
	*width  = (int)IHDR_Width;
	*height = (int)IHDR_Height;

	free(DecompressedData);
	DX12_CloseBufferedFile(ThePNG);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * @brief DX12_LoadImage
 * @param[in]  name   Game-path (with or without extension).
 * @param[out] pic    malloc()-allocated RGBA buffer, or NULL.
 * @param[out] width  Image width.
 * @param[out] height Image height.
 *
 * Tries extensions in order: .tga, .jpg, .png.
 */
void DX12_LoadImage(const char *name, byte **pic, int *width, int *height)
{
	static const char *s_exts[] = { "tga", "jpg", "png", NULL };
	char  localName[MAX_QPATH];
	char  tryName[MAX_QPATH];
	void *buf  = NULL;
	int   size = 0;
	int   i;

	*pic    = NULL;
	*width  = 0;
	*height = 0;

	if (!name || !name[0])
	{
		return;
	}

	// Strip any existing extension so we can try each one in order
	DX12_StripExtension(name, localName, sizeof(localName));

	for (i = 0; s_exts[i]; i++)
	{
		snprintf(tryName, sizeof(tryName), "%s.%s", localName, s_exts[i]);

		size = dx12.ri.FS_ReadFile(tryName, &buf);
		if (size <= 0 || !buf)
		{
			buf = NULL;
			continue;
		}

		if (i == 0)
		{
			DX12_DecodeTGA((const byte *)buf, size, pic, width, height);
		}
		else if (i == 1)
		{
			DX12_DecodeJPG((const byte *)buf, size, pic, width, height);
		}
		else if (i == 2)
		{
			DX12_DecodePNG((const byte *)buf, size, pic, width, height);
		}

		dx12.ri.FS_FreeFile(buf);
		buf = NULL;

		if (*pic)
		{
			return;
		}
	}
}

/**
 * @brief DX12_FreeImage
 * @param[in] pic  Buffer returned by DX12_LoadImage(), or NULL.
 */
void DX12_FreeImage(byte *pic)
{
	if (pic)
	{
		free(pic);
	}
}

#endif // _WIN32
