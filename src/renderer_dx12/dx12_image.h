/**
 * @file dx12_image.h
 * @brief DX12-only image loader – no GL, no renderer_common dependency.
 *
 * Reads raw file bytes via dx12.ri.FS_ReadFile and decodes them to a
 * dx12.ri.Z_Malloc()-allocated RGBA (4 bytes per pixel, row-major, top-left origin)
 * pixel buffer.  Only TGA (types 2 and 10, 24-bit and 32-bit) is
 * implemented; other formats return NULL.
 */

#ifndef DX12_IMAGE_H
#define DX12_IMAGE_H

#ifdef _WIN32

#include "tr_dx12_local.h"

/**
 * @brief Load an image to an RGBA pixel buffer.
 * @param[in]  name   Game-path of the image (with or without extension).
 *                    Extensions tried in order: .tga, .jpg, .png.
 * @param[out] pic    Set to a dx12.ri.Z_Malloc()-allocated buffer on success, NULL
 *                    on failure.  Free with DX12_FreeImage().
 * @param[out] width  Image width in pixels.
 * @param[out] height Image height in pixels.
 */
void DX12_LoadImage(const char *name, byte **pic, int *width, int *height);

/**
 * @brief Free a pixel buffer previously returned by DX12_LoadImage().
 * @param[in] pic  Pointer returned by DX12_LoadImage(), or NULL (no-op).
 */
void DX12_FreeImage(byte *pic);

#endif // _WIN32
#endif // DX12_IMAGE_H
