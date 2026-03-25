/**
 * @file dx12_image.cpp
 * @brief DX12-only image loader.
 *
 * Minimal TGA decoder (uncompressed type 2 and RLE type 10, 24-bit and
 * 32-bit) with no dependency on renderer_common, OpenGL, or any GL headers.
 * Pixel data is loaded via dx12.ri.FS_ReadFile and decoded into a plain
 * malloc()-allocated RGBA buffer.
 */

#include "dx12_image.h"

#ifdef _WIN32

#include <stdlib.h>   // malloc / free
#include <string.h>   // memcpy

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
 * Only TGA decoding is implemented; other formats fall through to the next
 * extension.
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

		// Decode based on extension index (0 = tga)
		if (i == 0)
		{
			DX12_DecodeTGA((const byte *)buf, size, pic, width, height);
		}
		// .jpg and .png not yet decoded natively; release and try next extension

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
