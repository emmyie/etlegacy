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

extern dx12Material_t dx12Materials[DX12_MAX_MATERIALS];
extern int            dx12NumMaterials;

#endif // _WIN32
#endif // DX12_SHADER_H
