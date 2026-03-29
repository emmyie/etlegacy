/**
 * @file dx12_shader.h
 * @brief DX12 texture registry – no GL, no renderer_common dependency.
 *
 * A flat array of up to DX12_MAX_TEXTURES entries.  Indices 0–2 are always
 * built-in fallback textures: slot 0 (white), slot 1 (black), slot 2
 * (noshader checkerboard).  Handles returned by DX12_RegisterTexture() are
 * indices into this array.
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

// ---------------------------------------------------------------------------
// Shader remap table  (populated by RE_DX12_RemapShader)
// ---------------------------------------------------------------------------

/** Maximum simultaneous shader remaps. */
#define DX12_MAX_SHADER_REMAPS 32

/**
 * @struct dx12ShaderRemap_t
 * @brief One entry in the shader remap table.
 */
typedef struct
{
	char     oldName[MAX_QPATH]; ///< Original shader name (stripped extension)
	char     newName[MAX_QPATH]; ///< Replacement shader name
	float    timeOffset;         ///< Optional time offset passed by the caller
	qboolean active;             ///< qtrue when this slot is occupied
} dx12ShaderRemap_t;

extern dx12ShaderRemap_t dx12ShaderRemaps[DX12_MAX_SHADER_REMAPS];
extern int               dx12NumShaderRemaps;

/**
 * @brief Normalise a game-path in-place: replaces every backslash with a
 *        forward slash.  ET: Legacy uses forward-slash separators internally,
 *        but callers may supply Windows-style paths.
 * @param[in,out] path  NUL-terminated path string; modified in place.
 */
void DX12_FixPath(char *path);

/**
 * @brief Add or update a shader remap entry.
 * @param[in] oldName     Original shader name.
 * @param[in] newName     Replacement shader name.
 * @param[in] timeOffset  Optional time offset.
 */
void DX12_AddShaderRemap(const char *oldName, const char *newName, float timeOffset);

/**
 * @brief Look up the remap table for @p name.
 * @return The remapped name, or @p name itself if no remap is found.
 */
const char *DX12_GetRemappedShader(const char *name);

// ---------------------------------------------------------------------------
// Dynamic (runtime-loaded) shader registration
// ---------------------------------------------------------------------------

/**
 * @brief Parse a shader block from @p shadertext and register it as a DX12
 *        material, exactly like DX12_RegisterMaterial but sourced from memory
 *        instead of the filesystem.
 * @param[in] name        Shader name (used for cache key).
 * @param[in] shadertext  Full shader script text (may contain multiple blocks).
 * @return qtrue if the material was successfully parsed and registered.
 */
qboolean DX12_RegisterMaterialFromText(const char *name, const char *shadertext);

/**
 * @brief Copy a sub-region of the current back-buffer render target into a
 *        DX12 texture slot, (re-)allocating the texture resource if its
 *        dimensions do not match (w × h).  Mirrors glCopyTexImage2D.
 *
 * @param[in,out] entry   Texture registry entry to update.
 * @param[in]     srvSlot SRV heap slot index for this entry.
 * @param[in]     x       Left edge of source rectangle (back-buffer coords).
 * @param[in]     y       Top  edge of source rectangle (back-buffer coords).
 * @param[in]     w       Width  of the copy region in pixels.
 * @param[in]     h       Height of the copy region in pixels.
 */
void DX12_CopyRenderTargetToTexture(dx12ShaderEntry_t *entry, int srvSlot,
                                    int x, int y, int w, int h);

#endif // _WIN32
#endif // DX12_SHADER_H
