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
 * @file dx12_poly.h
 * @brief DX12 2D drawing helpers – no GL, no renderer_common dependency.
 *
 * All functions write vertices into the per-frame ring-buffer
 * (dx12.quadVBMapped / dx12.quadVBOffset) and issue draw calls against the
 * already-open command list.  They must only be called between
 * DX12_BeginFrameRender() and R_DX12_SwapBuffers().
 */

#ifndef DX12_POLY_H
#define DX12_POLY_H

#ifdef _WIN32

#include "tr_dx12_local.h"

/**
 * @brief Draw a textured quad with per-corner modulate color.
 *
 * @param x,y       Top-left corner in screen pixels (0→vidWidth, 0→vidHeight).
 * @param w,h       Width and height in screen pixels.
 * @param s1,t1     UV for the top-left corner.
 * @param s2,t2     UV for the bottom-right corner.
 * @param hShader   Texture handle from DX12_RegisterTexture().
 */
void DX12_DrawStretchPic(float x, float y, float w, float h,
                         float s1, float t1, float s2, float t2,
                         qhandle_t hShader);

/**
 * @brief Draw a textured quad with a gradient modulate across two corners.
 *
 * @param gradientColor  RGBA [0,1] color applied to the gradient side.
 * @param gradientType   0 = left-right gradient; any other value = top-bottom.
 */
void DX12_DrawStretchPicGradient(float x, float y, float w, float h,
                                 float s1, float t1, float s2, float t2,
                                 qhandle_t hShader, const float *gradientColor,
                                 int gradientType);

/**
 * @brief Draw an array of 2D polygon vertices as a triangle fan.
 *
 * @param polys     Array of numverts polyVert_t vertices (screen-space XY,
 *                  UV, and byte modulate color).
 * @param numverts  Total vertex count (≥ 3).
 * @param hShader   Texture handle.
 */
void DX12_Add2dPolys(polyVert_t *polys, int numverts, qhandle_t hShader);

#endif // _WIN32
#endif // DX12_POLY_H
