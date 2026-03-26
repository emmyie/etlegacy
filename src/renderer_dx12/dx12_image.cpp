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
#include <setjmp.h>   // setjmp / longjmp  (JPEG error recovery)
#include <jpeglib.h>  // libjpeg decode  (linked via renderer_libraries)
#include <png.h>      // libpng decode   (linked via renderer_libraries)

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
// PNG decoding
// ---------------------------------------------------------------------------

/**
 * @brief Per-read-operation state for the custom libpng memory reader.
 */
typedef struct
{
	const byte *data;
	int         size;
	int         pos;
} dx12PngReadState_t;

/**
 * @brief Custom read callback for libpng: reads bytes from a memory buffer.
 */
static void DX12_PngReadData(png_structp png_ptr, png_bytep outBytes,
                              png_size_t byteCount)
{
	dx12PngReadState_t *st = (dx12PngReadState_t *)png_get_io_ptr(png_ptr);
	int                 remaining;

	if (!st)
	{
		png_error(png_ptr, "NULL read state");
		return;
	}

	remaining = st->size - st->pos;
	if ((int)byteCount > remaining)
	{
		png_error(png_ptr, "PNG read past end of buffer");
		return;
	}

	memcpy(outBytes, st->data + st->pos, byteCount);
	st->pos += (int)byteCount;
}

/**
 * @brief Decode a PNG file in memory to RGBA.
 *
 * Handles: palette, greyscale, greyscale+alpha, RGB, RGBA; bit depths
 * 1/2/4/8/16 (all normalised to 8-bit RGBA on output).
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
	dx12PngReadState_t  st;
	png_structp         png;
	png_infop           info;
	int                 w, h, y;
	png_byte            colorType, bitDepth;
	byte               *out;
	png_bytep          *rowPtrs;

	*pic = NULL; *width = 0; *height = 0;

	if (!buf || bufSize < 8)
	{
		return;
	}

	// Validate PNG signature
	if (png_sig_cmp((png_const_bytep)buf, 0, 8))
	{
		return;
	}

	png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png)
	{
		return;
	}

	info = png_create_info_struct(png);
	if (!info)
	{
		png_destroy_read_struct(&png, NULL, NULL);
		return;
	}

	if (setjmp(png_jmpbuf(png)))
	{
		png_destroy_read_struct(&png, &info, NULL);
		return;
	}

	st.data = buf;
	st.size = bufSize;
	st.pos  = 0;
	png_set_read_fn(png, &st, DX12_PngReadData);

	png_read_info(png, info);

	w         = (int)png_get_image_width(png, info);
	h         = (int)png_get_image_height(png, info);
	colorType = png_get_color_type(png, info);
	bitDepth  = png_get_bit_depth(png, info);

	// Normalise to 8-bit RGBA
	if (bitDepth == 16)
	{
		png_set_strip_16(png);
	}
	if (colorType == PNG_COLOR_TYPE_PALETTE)
	{
		png_set_palette_to_rgb(png);
	}
	if (colorType == PNG_COLOR_TYPE_GRAY && bitDepth < 8)
	{
		png_set_expand_gray_1_2_4_to_8(png);
	}
	if (png_get_valid(png, info, PNG_INFO_tRNS))
	{
		png_set_tRNS_to_alpha(png);
	}
	if (colorType == PNG_COLOR_TYPE_RGB ||
	    colorType == PNG_COLOR_TYPE_GRAY ||
	    colorType == PNG_COLOR_TYPE_PALETTE)
	{
		png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
	}
	if (colorType == PNG_COLOR_TYPE_GRAY ||
	    colorType == PNG_COLOR_TYPE_GRAY_ALPHA)
	{
		png_set_gray_to_rgb(png);
	}

	png_read_update_info(png, info);

	if (w <= 0 || h <= 0)
	{
		png_destroy_read_struct(&png, &info, NULL);
		return;
	}

	out = (byte *)malloc((size_t)w * (size_t)h * 4);
	if (!out)
	{
		png_destroy_read_struct(&png, &info, NULL);
		return;
	}

	rowPtrs = (png_bytep *)malloc((size_t)h * sizeof(png_bytep));
	if (!rowPtrs)
	{
		free(out);
		png_destroy_read_struct(&png, &info, NULL);
		return;
	}

	for (y = 0; y < h; y++)
	{
		rowPtrs[y] = out + (size_t)y * (size_t)w * 4;
	}

	png_read_image(png, rowPtrs);
	free(rowPtrs);

	png_destroy_read_struct(&png, &info, NULL);

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
