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
 * @file dx12_shader.cpp
 * @brief DX12 texture registry.
 *
 * Wraps DX12_LoadImage + DX12_CreateTextureFromRGBA into a simple name-keyed
 * table of up to DX12_MAX_TEXTURES entries.  Index 0 is a permanent 1×1
 * opaque-white fallback.
 *
 * @note Texture registration must happen while no frame is open (i.e. before
 *       the first DX12_BeginFrameRender call or after R_DX12_SwapBuffers),
 *       because DX12_CreateTextureFromRGBA resets the main command allocator.
 */

#include "dx12_shader.h"
#include "dx12_image.h"

#ifdef _WIN32

dx12ShaderEntry_t dx12Shaders[DX12_MAX_TEXTURES];
int               dx12NumShaders = 0;

// ---------------------------------------------------------------------------
// DX12_InitTextures
// ---------------------------------------------------------------------------

/**
 * @brief DX12_InitTextures
 *
 * Clears the registry and pre-loads the fallback white texture at slot 0.
 * Must be called after R_DX12_Init() (device + SRV heap must exist).
 */
void DX12_InitTextures(void)
{
	byte white[4] = { 255, 255, 255, 255 };

	Com_Memset(dx12Shaders, 0, sizeof(dx12Shaders));
	dx12NumShaders = 0;

	// Slot 0: 1×1 opaque-white fallback
	dx12Texture_t fallback = DX12_CreateTextureFromRGBA(white, 1, 1, 0);

	if (fallback.resource)
	{
		Q_strncpyz(dx12Shaders[0].name, "__white__", sizeof(dx12Shaders[0].name));
		dx12Shaders[0].width  = 1;
		dx12Shaders[0].height = 1;
		dx12Shaders[0].tex    = fallback;
		dx12Shaders[0].valid  = qtrue;
		dx12NumShaders        = 1;
	}
	else
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_InitTextures: white fallback texture failed\n");
		dx12NumShaders = 1; // still reserve slot 0
	}
}

// ---------------------------------------------------------------------------
// DX12_ShutdownTextures
// ---------------------------------------------------------------------------

/**
 * @brief DX12_ShutdownTextures
 *
 * Releases all D3D12 texture resources.  Must be called while the GPU is
 * idle (after DX12_WaitForGpu / R_DX12_Shutdown's wait).
 */
void DX12_ShutdownTextures(void)
{
	int i;

	for (i = 0; i < dx12NumShaders; i++)
	{
		if (dx12Shaders[i].valid && dx12Shaders[i].tex.resource)
		{
			dx12Shaders[i].tex.resource->Release();
			dx12Shaders[i].tex.resource = NULL;
		}
		dx12Shaders[i].valid = qfalse;
	}
	dx12NumShaders = 0;
}

// ---------------------------------------------------------------------------
// DX12_RegisterTexture
// ---------------------------------------------------------------------------

/**
 * @brief DX12_RegisterTexture
 * @param[in] name  Game-path of the image (with or without extension).
 * @return          Handle (registry index) ≥ 1, or 0 on failure.
 *
 * Deduplicates by name (case-insensitive).  Loads via DX12_LoadImage and
 * uploads to GPU via DX12_CreateTextureFromRGBA.
 */
qhandle_t DX12_RegisterTexture(const char *name)
{
	int           i;
	byte         *pic    = NULL;
	int           width  = 0;
	int           height = 0;
	int           slot;
	dx12Texture_t tex;

	if (!name || !name[0])
	{
		return 0;
	}

	// Deduplicate: return existing handle if already loaded
	for (i = 1; i < dx12NumShaders; i++)
	{
		if (dx12Shaders[i].valid && !Q_stricmp(dx12Shaders[i].name, name))
		{
			return (qhandle_t)i;
		}
	}

	if (dx12NumShaders >= DX12_MAX_TEXTURES)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_RegisterTexture: texture registry full\n");
		return 0;
	}

	// Guard against registration during an open frame
	if (dx12.frameOpen)
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "DX12_RegisterTexture: '%s' registered mid-frame; deferring is not supported\n",
		               name);
		return 0;
	}

	DX12_LoadImage(name, &pic, &width, &height);

	if (!pic || width <= 0 || height <= 0)
	{
		dx12.ri.Printf(PRINT_DEVELOPER, "DX12_RegisterTexture: could not load '%s'\n", name);
		return 0;
	}

	slot = dx12NumShaders;
	tex  = DX12_CreateTextureFromRGBA((const byte *)pic, width, height, slot);
	DX12_FreeImage(pic);

	if (!tex.resource)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_RegisterTexture: GPU upload failed for '%s'\n", name);
		return 0;
	}

	Q_strncpyz(dx12Shaders[slot].name, name, sizeof(dx12Shaders[slot].name));
	dx12Shaders[slot].width  = width;
	dx12Shaders[slot].height = height;
	dx12Shaders[slot].tex    = tex;
	dx12Shaders[slot].valid  = qtrue;
	dx12NumShaders++;

	return (qhandle_t)slot;
}

// ---------------------------------------------------------------------------
// DX12_GetTexture
// ---------------------------------------------------------------------------

/**
 * @brief DX12_GetTexture
 * @param[in] handle  Registry index returned by DX12_RegisterTexture().
 * @return            Pointer to the dx12Texture_t for this handle, or the
 *                    white fallback (slot 0) if the handle is out-of-range
 *                    or invalid.  Returns NULL if the registry is empty.
 */
dx12Texture_t *DX12_GetTexture(qhandle_t handle)
{
	int idx = (int)handle;

	if (idx >= 0 && idx < dx12NumShaders && dx12Shaders[idx].valid)
	{
		return &dx12Shaders[idx].tex;
	}

	// Fall back to the white texture at slot 0
	if (dx12NumShaders > 0 && dx12Shaders[0].valid)
	{
		return &dx12Shaders[0].tex;
	}

	return NULL;
}

#endif // _WIN32
