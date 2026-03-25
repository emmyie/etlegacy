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
 * @file dx12_shader.h
 * @brief DX12 texture registry – no GL, no renderer_common dependency.
 *
 * A flat array of up to DX12_MAX_TEXTURES entries.  Index 0 is always a
 * 1×1 opaque-white fallback texture used whenever a shader handle is invalid.
 * Handles returned by DX12_RegisterTexture() are indices into this array.
 */

#ifndef DX12_SHADER_H
#define DX12_SHADER_H

#ifdef _WIN32

#include "tr_dx12_local.h"

/**
 * @struct dx12ShaderEntry_t
 * @brief One slot in the DX12 texture registry.
 */
typedef struct
{
	char          name[MAX_QPATH]; ///< Game-path used to load this texture
	int           width;           ///< Original image width
	int           height;          ///< Original image height
	dx12Texture_t tex;             ///< D3D12 resource + SRV handles
	qboolean      valid;           ///< qtrue once successfully uploaded
} dx12ShaderEntry_t;

extern dx12ShaderEntry_t dx12Shaders[DX12_MAX_TEXTURES];
extern int               dx12NumShaders;

#endif // _WIN32
#endif // DX12_SHADER_H
