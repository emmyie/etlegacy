/*
 * Wolfenstein: Enemy Territory GPL Source Code
 * Copyright (C) 1999-2010 id Software LLC, a ZeniMax Media company.
 *
 * ET: Legacy
 * Copyright (C) 2012-2024 ET:Legacy team <mail@etlegacy.com>
 *
 * This file is part of ET: Legacy - http://www.etlegacy.com
 *
 * ET: Legacy is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ET: Legacy is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ET: Legacy. If not, see <http://www.gnu.org/licenses/>.
 *
 * In addition, Wolfenstein: Enemy Territory GPL Source Code is also
 * subject to certain additional terms. You should have received a copy
 * of these additional terms immediately following the terms and conditions
 * of the GNU General Public License which accompanied the source code.
 * If not, please request a copy in writing from id Software at the address below.
 *
 * id Software LLC, c/o ZeniMax Media Inc., Suite 120, Rockville, Maryland 20850 USA.
 */
/**
 * @file dx12_image.h
 * @brief DX12-only image loader – no GL, no renderer_common dependency.
 *
 * Reads raw file bytes via dx12.ri.FS_ReadFile and decodes them to a
 * malloc()-allocated RGBA (4 bytes per pixel, row-major, top-left origin)
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
 * @param[out] pic    Set to a malloc()-allocated buffer on success, NULL
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
