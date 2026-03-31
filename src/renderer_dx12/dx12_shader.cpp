/**
 * @file dx12_shader.cpp
 * @brief DX12 texture registry.
 *
 * Wraps DX12_LoadImage + DX12_CreateTextureFromRGBA into a simple name-keyed
 * table of up to DX12_MAX_TEXTURES entries.  Indices 0–2 are permanent
 * built-in fallback textures (white, black, noshader checkerboard).
 *
 * When a shader name (e.g. "ui/assets/et_clouds") does not correspond to an
 * image file on disk, DX12_RegisterTexture falls back to a basic shader-script
 * parser that scans the materials/ and scripts/ directories for a matching
 * shader definition and extracts the first-stage map texture path.  This lets
 * the renderer display at least the primary texture of complex multi-stage
 * shaders even though animation and blending are not implemented.
 *
 * @note DX12_CreateTextureFromRGBA uses the dedicated upload command
 *       allocator/list (dx12.uploadCmdAllocator / dx12.uploadCmdList), not the
 *       per-frame rendering allocator.  Registration is therefore safe at any
 *       time, including while a frame is open.
 */

#include "dx12_shader.h"
#include "dx12_image.h"
#include "dx12_world.h"

#ifdef _WIN32

#include <string.h>   // strncpy, strchr, etc.
#include <stdlib.h>   // atof

dx12ShaderEntry_t dx12Shaders[DX12_MAX_TEXTURES];
int               dx12NumShaders = 0;

dx12Material_t dx12Materials[DX12_MAX_MATERIALS];
int            dx12NumMaterials = 0;

// ---------------------------------------------------------------------------
// DX12_FixPath
// ---------------------------------------------------------------------------

/**
 * @brief Replace every backslash in @p path with a forward slash.
 *
 * ET: Legacy uses '/' as its canonical path separator.  Callers (e.g. the
 * UI or mod DLL on Windows) sometimes supply paths with '\\' separators; the
 * VFS never resolves those correctly, so we normalise early.
 */
void DX12_FixPath(char *path)
{
	char *p;

	if (!path)
	{
		return;
	}
	for (p = path; *p; ++p)
	{
		if (*p == '\\')
		{
			*p = '/';
		}
	}
}

// ---------------------------------------------------------------------------
// Shader remap table
// ---------------------------------------------------------------------------

dx12ShaderRemap_t dx12ShaderRemaps[DX12_MAX_SHADER_REMAPS];
int               dx12NumShaderRemaps = 0;

/**
 * @brief Add or update a shader remap entry.
 */
void DX12_AddShaderRemap(const char *oldName, const char *newName, float timeOffset)
{
	int i;
	char strippedOld[MAX_QPATH];
	char strippedNew[MAX_QPATH];

	if (!oldName || !oldName[0] || !newName || !newName[0])
	{
		return;
	}

	COM_StripExtension(oldName, strippedOld, sizeof(strippedOld));
	COM_StripExtension(newName, strippedNew, sizeof(strippedNew));

	// Update existing entry if old name matches
	for (i = 0; i < dx12NumShaderRemaps; i++)
	{
		if (dx12ShaderRemaps[i].active &&
		    !DX12_Stricmp(dx12ShaderRemaps[i].oldName, strippedOld))
		{
			Q_strncpyz(dx12ShaderRemaps[i].newName, strippedNew,
			           sizeof(dx12ShaderRemaps[i].newName));
			dx12ShaderRemaps[i].timeOffset = timeOffset;
			return;
		}
	}

	if (dx12NumShaderRemaps >= DX12_MAX_SHADER_REMAPS)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_AddShaderRemap: remap table full\n");
		return;
	}

	Q_strncpyz(dx12ShaderRemaps[dx12NumShaderRemaps].oldName, strippedOld,
	           sizeof(dx12ShaderRemaps[0].oldName));
	Q_strncpyz(dx12ShaderRemaps[dx12NumShaderRemaps].newName, strippedNew,
	           sizeof(dx12ShaderRemaps[0].newName));
	dx12ShaderRemaps[dx12NumShaderRemaps].timeOffset = timeOffset;
	dx12ShaderRemaps[dx12NumShaderRemaps].active     = qtrue;
	dx12NumShaderRemaps++;
}

/**
 * @brief Look up the remap table for @p name.
 * @return The remapped name, or NULL if no remap found.
 */
const char *DX12_GetRemappedShader(const char *name)
{
	int  i;
	char stripped[MAX_QPATH];

	if (!name || !name[0])
	{
		return NULL;
	}

	COM_StripExtension(name, stripped, sizeof(stripped));

	for (i = 0; i < dx12NumShaderRemaps; i++)
	{
		if (dx12ShaderRemaps[i].active &&
		    !DX12_Stricmp(dx12ShaderRemaps[i].oldName, stripped))
		{
			return dx12ShaderRemaps[i].newName;
		}
	}
	return NULL;
}

// ---------------------------------------------------------------------------
// One-time "missing asset" warning deduplication.
// Prevents thousands of repeated "could not load" / "could not resolve"
// messages when the same texture or material name fails on every frame that
// requests it (e.g. sky shaders, decals, player skins).
//
// SHD_WarnOnce() returns qtrue on the FIRST call for a given @p name and
// qfalse on every subsequent call.  The caller should print its warning and
// return 0 only when the function returns qtrue.
// ---------------------------------------------------------------------------

#define DX12_MAX_MISSING_WARN 512

static char s_missingNames[DX12_MAX_MISSING_WARN][MAX_QPATH];
static int  s_numMissingNames = 0;

/**
 * @brief Returns qtrue (and records @p name) on the first call for a given
 *        asset name; returns qfalse on every subsequent call for the same name.
 *
 * @param[in] name  Asset name to check (case-insensitive).
 * @return qtrue if this is the first occurrence, qfalse if already recorded.
 */
static qboolean SHD_WarnOnce(const char *name)
{
	int i;

	for (i = 0; i < s_numMissingNames; i++)
	{
		if (!DX12_Stricmp(s_missingNames[i], name))
		{
			return qfalse;
		}
	}

	if (s_numMissingNames < DX12_MAX_MISSING_WARN)
	{
		Q_strncpyz(s_missingNames[s_numMissingNames], name, MAX_QPATH);
		s_numMissingNames++;
	}

	return qtrue;
}

// ---------------------------------------------------------------------------
// DX12_CreateSolidTexture / DX12_InitTextures
// ---------------------------------------------------------------------------

/**
 * @brief Create a 1×1 solid-colour texture in the shader registry at a fixed
 *        @p slot.  Does not modify dx12NumShaders; the caller is responsible
 *        for advancing that counter to reserve the slot.
 *
 * @param[in] r     Red   component (0–255).
 * @param[in] g     Green component (0–255).
 * @param[in] b     Blue  component (0–255).
 * @param[in] name  Registry name stored in dx12Shaders[slot].name.
 * @param[in] slot  Index into dx12Shaders[] to populate.
 */
static void DX12_CreateSolidTexture(byte r, byte g, byte b, const char *name, int slot)
{
	byte          pixel[4] = { r, g, b, 255 };
	dx12Texture_t fallback = DX12_CreateTextureFromRGBA(pixel, 1, 1, slot);

	if (fallback.resource)
	{
		Q_strncpyz(dx12Shaders[slot].name, name, sizeof(dx12Shaders[slot].name));
		dx12Shaders[slot].width  = 1;
		dx12Shaders[slot].height = 1;
		dx12Shaders[slot].tex    = fallback;
		dx12Shaders[slot].valid  = qtrue;
	}
	else
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_InitTextures: '%s' fallback texture failed\n", name);
	}
}

/**
 * @brief DX12_InitTextures
 *
 * Clears the registry and pre-loads three built-in fallback textures:
 *   slot 0 – 1×1 opaque-white   ("__white__")
 *   slot 1 – 1×1 opaque-black   ("__black__")
 *   slot 2 – 4×4 magenta/purple checkerboard ("noshader")
 *
 * Must be called after R_DX12_Init() (device + SRV heap must exist).
 */
void DX12_InitTextures(void)
{
	Com_Memset(dx12Shaders, 0, sizeof(dx12Shaders));
	dx12NumShaders = 0;

	Com_Memset(dx12Materials, 0, sizeof(dx12Materials));
	dx12NumMaterials = 0;

	// Reset the one-time missing-asset warning table so that every map load
	// gets a fresh set of warnings for its own assets.
	s_numMissingNames = 0;

	// Slot 0: 1×1 opaque-white fallback
	DX12_CreateSolidTexture(255, 255, 255, "__white__", 0);
	dx12NumShaders = 1; // always reserve slot 0

	// Slot 1: 1×1 opaque-black fallback (used by *black virtual textures)
	DX12_CreateSolidTexture(0, 0, 0, "__black__", 1);
	dx12NumShaders = 2; // always reserve slot 1

	// Slot 2: 4×4 magenta/purple checkerboard – "noshader" fallback.
	// GL creates this texture in memory; DX12 must do the same so that
	// DX12_RegisterTexture("noshader") finds it in the dedup table and never
	// tries to load "noshader.tga" from disk.
	{
		byte          pixels[4 * 4 * 4];
		int           i;
		dx12Texture_t fallback;

		for (i = 0; i < 16; i++)
		{
			byte c             = (byte)(((i ^ (i > 2)) & 1) ? 255 : 128);
			pixels[i * 4 + 0] = c;
			pixels[i * 4 + 1] = 0;
			pixels[i * 4 + 2] = c;
			pixels[i * 4 + 3] = 255;
		}

		fallback = DX12_CreateTextureFromRGBA(pixels, 4, 4, 2);

		if (fallback.resource)
		{
			Q_strncpyz(dx12Shaders[2].name, "noshader", sizeof(dx12Shaders[2].name));
			dx12Shaders[2].width  = 4;
			dx12Shaders[2].height = 4;
			dx12Shaders[2].tex    = fallback;
			dx12Shaders[2].valid  = qtrue;
		}
		else
		{
			dx12.ri.Printf(PRINT_WARNING, "DX12_InitTextures: noshader fallback texture failed\n");
		}
		dx12NumShaders = 3; // always reserve slot 2
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

	// Clear material cache first (materials hold texture-handle indices, not GPU resources)
	Com_Memset(dx12Materials, 0, sizeof(dx12Materials));
	dx12NumMaterials = 0;

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

	// Zero the whole table so that stale names / SRV indices from the previous
	// map session cannot be accidentally reused if a future code path checks an
	// entry without consulting .valid first.
	Com_Memset(dx12Shaders, 0, sizeof(dx12Shaders));
}

// ---------------------------------------------------------------------------
// Shader-script texture resolution
// ---------------------------------------------------------------------------

/**
 * @brief Skip a single character if it is a whitespace byte.
 *
 * Returns the incremented pointer when the current character is a space,
 * horizontal tab, carriage return, or newline; otherwise returns @p p
 * unchanged.
 */
static const char *SH_SkipWhite(const char *p, const char *end)
{
	while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n'))
	{
		p++;
	}
	return p;
}

/**
 * @brief Advance past a @c // line comment (the @c // has already been
 *        consumed by the caller).
 */
static const char *SH_SkipLineComment(const char *p, const char *end)
{
	while (p < end && *p != '\n')
	{
		p++;
	}
	return p;
}

/**
 * @brief Advance past a @c /\* block comment (the opening @c /\* has already
 *        been consumed by the caller).
 */
static const char *SH_SkipBlockComment(const char *p, const char *end)
{
	while (p < end - 1)
	{
		if (p[0] == '*' && p[1] == '/')
		{
			return p + 2;
		}
		p++;
	}
	return end;
}

/**
 * @brief Skip all whitespace and comment sequences starting at @p p.
 */
static const char *SH_SkipWC(const char *p, const char *end)
{
	for (;;)
	{
		p = SH_SkipWhite(p, end);
		if (p >= end)
		{
			return end;
		}
		if (p + 1 < end && p[0] == '/' && p[1] == '/')
		{
			p = SH_SkipLineComment(p + 2, end);
		}
		else if (p + 1 < end && p[0] == '/' && p[1] == '*')
		{
			p = SH_SkipBlockComment(p + 2, end);
		}
		else
		{
			break;
		}
	}
	return p;
}

/**
 * @brief Read one whitespace-delimited token from the shader source.
 *
 * The characters @c { and @c } are treated as individual single-character
 * tokens; all other tokens end at whitespace or a brace.  The result is
 * written to @p tok (NUL-terminated) and the returned pointer points to the
 * first character after the consumed token.
 *
 * @param[in]  p      Source position.
 * @param[in]  end    One past the last byte of the source buffer.
 * @param[out] tok    Output buffer for the token string.
 * @param[in]  maxLen Capacity of @p tok including the terminating NUL.
 * @return            Pointer to the first byte after the token.
 */
static const char *SH_ReadToken(const char *p, const char *end,
                                 char *tok, int maxLen)
{
	int i = 0;

	tok[0] = '\0';
	p      = SH_SkipWC(p, end);

	if (p >= end)
	{
		return p;
	}

	// Braces are stand-alone single-character tokens
	if (*p == '{' || *p == '}')
	{
		tok[0] = *p++;
		tok[1] = '\0';
		return p;
	}

	while (p < end && i < maxLen - 1)
	{
		char c = *p;
		if (c == ' ' || c == '\t' || c == '\r' || c == '\n'
		    || c == '{' || c == '}')
		{
			break;
		}
		tok[i++] = c;
		p++;
	}
	tok[i] = '\0';
	return p;
}

/**
 * @brief Search a single shader-file buffer for a shader whose name equals
 *        @p shaderName and return the first map or clampmap texture path
 *        found in its first stage.
 *
 * The function implements a simple recursive-descent over the token stream:
 *   - Top level: @c name @c { … @c }  blocks are shaders.
 *   - Depth 1  : @c { … @c }  blocks are stages.
 *   - Depth 2  : @c map or @c clampmap followed by the texture path.
 *
 * Special virtual texture names ($lightmap, $whiteimage, $white) are
 * ignored so that the next map directive in the stage can be used instead.
 *
 * @param[in]  buf        Raw text content of the shader file.
 * @param[in]  bufLen     Number of bytes in @p buf.
 * @param[in]  shaderName Shader name to look for (case-insensitive).
 * @param[out] outPath    Destination buffer for the resolved texture path.
 * @param[in]  outLen     Capacity of @p outPath including the NUL terminator.
 * @return     qtrue when a texture path was successfully written to @p outPath.
 */
static qboolean SH_FindTextureInBuffer(const char *buf, int bufLen,
                                        const char *shaderName,
                                        char *outPath, int outLen)
{
	const char *p   = buf;
	const char *end = buf + bufLen;
	char        tok[MAX_QPATH];

	while (p < end)
	{
		int depth;

		p = SH_ReadToken(p, end, tok, sizeof(tok));
		if (!tok[0])
		{
			break;
		}

		// Skip any lone '{' or '}' at the top level (defensive)
		if (tok[0] == '{' || tok[0] == '}')
		{
			continue;
		}

		// tok is a potential shader name; the next non-comment token must be '{'
		{
			const char *saved = p;
			char        brace[4];

			p = SH_ReadToken(p, end, brace, sizeof(brace));
			if (brace[0] != '{')
			{
				// Not a shader block – backtrack and keep scanning
				p = saved;
				continue;
			}

			// Check whether this shader name matches the one we want
			if (DX12_Stricmp(tok, shaderName) != 0)
			{
				// Not the shader we need – skip to the matching '}'
				depth = 1;
				while (p < end && depth > 0)
				{
					p = SH_ReadToken(p, end, tok, sizeof(tok));
					if (tok[0] == '{') { depth++; }
					else if (tok[0] == '}') { depth--; }
				}
				continue;
			}

			// We are inside the matching shader block (depth 1).
			// Scan the first stage block (the first nested '{…}') for a
			// map or clampmap directive.
			depth = 1;
			while (p < end && depth > 0)
			{
				p = SH_ReadToken(p, end, tok, sizeof(tok));
				if (!tok[0])
				{
					break;
				}

				if (tok[0] == '{')
				{
					depth++;
				}
				else if (tok[0] == '}')
				{
					depth--;
					// After the first stage closes (depth back to 1) we stop
					// looking for maps so we only use the first stage.
					if (depth == 1)
					{
						break;
					}
				}
				else if (depth == 2
				         && (!DX12_Stricmp(tok, "map")
				             || !DX12_Stricmp(tok, "clampmap")))
				{
					// The next token is the texture path
					p = SH_ReadToken(p, end, tok, sizeof(tok));
					if (!tok[0] || tok[0] == '{' || tok[0] == '}')
					{
						break;
					}
					// Skip virtual/special textures
					if (tok[0] == '$')
					{
						continue;
					}
					Q_strncpyz(outPath, tok, outLen);
					return qtrue;
				}
			}
			return qfalse;
		}
	}
	return qfalse;
}

/**
 * @brief Scan all @c .shader files in @p dir for a definition of
 *        @p shaderName and write the resolved texture path to @p outPath.
 *
 * @param[in]  dir        Shader directory to search (e.g. "materials").
 * @param[in]  shaderName Shader name to look for.
 * @param[out] outPath    Destination buffer for the resolved texture path.
 * @param[in]  outLen     Capacity of @p outPath.
 * @return     qtrue when the shader was found and a path was written.
 */
static qboolean SH_ScanDir(const char *dir, const char *shaderName,
                             char *outPath, int outLen)
{
	char       **fileList;
	int          numFiles = 0;
	int          i;
	qboolean     found    = qfalse;

	fileList = dx12.ri.FS_ListFiles(dir, ".shader", &numFiles);
	if (!fileList || numFiles <= 0)
	{
		if (fileList)
		{
			dx12.ri.FS_FreeFileList(fileList);
		}
		return qfalse;
	}

	for (i = 0; i < numFiles && !found; i++)
	{
		char  fullPath[MAX_QPATH];
		void *buf  = NULL;
		int   size = 0;

		snprintf(fullPath, sizeof(fullPath), "%s/%s", dir, fileList[i]);

		size = dx12.ri.FS_ReadFile(fullPath, &buf);
		if (size <= 0 || !buf)
		{
			continue;
		}

		found = SH_FindTextureInBuffer((const char *)buf, size,
		                               shaderName, outPath, outLen);
		dx12.ri.FS_FreeFile(buf);
	}

	dx12.ri.FS_FreeFileList(fileList);
	return found;
}

/**
 * @brief Attempt to resolve a shader name to an image-file path by scanning
 *        the shader-script directories.
 *
 * Searches "materials" first, then "scripts", stopping as soon as the
 * shader is found.
 *
 * @param[in]  shaderName  Shader name as registered by the game code.
 * @param[out] outPath     Receives the resolved image path on success.
 * @param[in]  outLen      Capacity of @p outPath.
 * @return     qtrue when an image path has been written to @p outPath.
 */
static qboolean DX12_FindShaderTexture(const char *shaderName,
                                        char *outPath, int outLen)
{
	if (SH_ScanDir("materials", shaderName, outPath, outLen))
	{
		return qtrue;
	}
	if (SH_ScanDir("scripts", shaderName, outPath, outLen))
	{
		return qtrue;
	}
	return qfalse;
}

// ---------------------------------------------------------------------------
// DX12_RegisterTexture
// ---------------------------------------------------------------------------

/**
 * @brief DX12_RegisterTexture
 * @param[in] name  Game-path of the image or shader (with or without
 *                  extension).
 * @return          Handle (registry index) ≥ 1, or 0 on failure.
 *
 * Deduplicates by name (case-insensitive).  First tries to load @p name
 * directly as an image file (extensions: .tga, .jpg, .png).  If that fails,
 * scans the shader-script directories for a definition whose name matches
 * @p name and tries to load the first-stage map texture found there.  The
 * registered entry always uses @p name as its key (not the resolved path) so
 * that duplicate detection works correctly when the caller reuses the same
 * shader name across multiple registration calls.
 */
qhandle_t DX12_RegisterTexture(const char *name)
{
	int           i;
	byte         *pic    = NULL;
	int           width  = 0;
	int           height = 0;
	int           slot;
	dx12Texture_t tex;
	char          fixedName[MAX_OSPATH];

	if (!name || !name[0])
	{
		return 2; // slot 2 = noshader
	}

	Q_strncpyz(fixedName, name, sizeof(fixedName));
	DX12_FixPath(fixedName);
	// Strip any leading slash — the VFS uses relative paths only.
	name = fixedName;
	while (*name == '/')
	{
		name++;
	}
	if (!name[0])
	{
		return 2;
	}

	// Deduplicate: return existing handle if already loaded
	for (i = 1; i < dx12NumShaders; i++)
	{
		if (dx12Shaders[i].valid && !DX12_Stricmp(dx12Shaders[i].name, name))
		{
			return (qhandle_t)i;
		}
	}

	if (dx12NumShaders >= DX12_MAX_TEXTURES)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_RegisterTexture: texture registry full\n");
		return 2; // slot 2 = noshader
	}

	// Virtual '*white' textures: register as a named alias sharing slot 0's
	// GPU descriptor (resource pointer is left NULL to avoid double-release on
	// shutdown).  DX12_GetTexture will fall through to the heap-start descriptor
	// which is always the white fallback.
	if (name[0] == '*' && !DX12_Stricmp(name + 1, "white"))
	{
		slot = dx12NumShaders;
		Q_strncpyz(dx12Shaders[slot].name, name, sizeof(dx12Shaders[slot].name));
		dx12Shaders[slot].width       = 1;
		dx12Shaders[slot].height      = 1;
		dx12Shaders[slot].valid       = qtrue;
		// Share the GPU-visible descriptor of slot 0 (white fallback).
		// resource is intentionally left NULL so shutdown does not double-release.
		dx12Shaders[slot].tex.resource  = NULL;
		dx12Shaders[slot].tex.cpuHandle = dx12Shaders[0].tex.cpuHandle;
		dx12Shaders[slot].tex.gpuHandle = dx12Shaders[0].tex.gpuHandle;
		dx12NumShaders++;
		return (qhandle_t)slot;
	}

	// Other virtual names that also map to the white fallback (slot 0):
	// $whiteimage and $white are standard Q3 virtual names for a solid-white texture.
	if (!DX12_Stricmp(name, "$whiteimage") ||
	    !DX12_Stricmp(name, "$white"))
	{
		slot = dx12NumShaders;
		Q_strncpyz(dx12Shaders[slot].name, name, sizeof(dx12Shaders[slot].name));
		dx12Shaders[slot].width       = 1;
		dx12Shaders[slot].height      = 1;
		dx12Shaders[slot].valid       = qtrue;
		dx12Shaders[slot].tex.resource  = NULL;
		dx12Shaders[slot].tex.cpuHandle = dx12Shaders[0].tex.cpuHandle;
		dx12Shaders[slot].tex.gpuHandle = dx12Shaders[0].tex.gpuHandle;
		dx12NumShaders++;
		return (qhandle_t)slot;
	}

	// *black maps to the opaque-black fallback at slot 1.
	if (name[0] == '*' && !DX12_Stricmp(name + 1, "black"))
	{
		slot = dx12NumShaders;
		Q_strncpyz(dx12Shaders[slot].name, name, sizeof(dx12Shaders[slot].name));
		dx12Shaders[slot].width      = 1;
		dx12Shaders[slot].height     = 1;
		dx12Shaders[slot].valid      = qtrue;
		// Alias to slot 1 (black fallback); resource left NULL to avoid double-release.
		dx12Shaders[slot].tex.resource  = NULL;
		dx12Shaders[slot].tex.cpuHandle = (dx12NumShaders > 1 && dx12Shaders[1].valid)
		                                  ? dx12Shaders[1].tex.cpuHandle
		                                  : dx12Shaders[0].tex.cpuHandle;
		dx12Shaders[slot].tex.gpuHandle = (dx12NumShaders > 1 && dx12Shaders[1].valid)
		                                  ? dx12Shaders[1].tex.gpuHandle
		                                  : dx12Shaders[0].tex.gpuHandle;
		dx12NumShaders++;
		return (qhandle_t)slot;
	}

	// *N - BSP lightmap index (e.g. *0, *1, ...).
	// Lightmaps are uploaded by DX12_LoadWorld into dx12World.lightmapHandles[].
	if (name[0] == '*' && name[1] >= '0' && name[1] <= '9')
	{
		const char *p         = name + 1;
		qboolean    allDigits = qtrue;
		while (*p) { if (*p < '0' || *p > '9') { allDigits = qfalse; break; } p++; }
		if (allDigits)
		{
			int lmIdx = atoi(name + 1);
			if (lmIdx >= 0 && lmIdx < dx12World.numLightmaps)
			{
				return dx12World.lightmapHandles[lmIdx];
			}
			return 2; // out-of-range lightmap index -> noshader
		}
	}

	// "noshader", "noshader.tga", "noshader.jpg", etc. -> built-in slot 2.
	{
		char baseName[MAX_QPATH];
		COM_StripExtension(name, baseName, sizeof(baseName));
		if (!DX12_Stricmp(baseName, "noshader"))
		{
			return 2;
		}
	}

	// Try loading the name directly as an image file
	DX12_LoadImageSmart(name, &pic, &width, &height);

	// If direct loading failed, try resolving via shader scripts
	if (!pic || width <= 0 || height <= 0)
	{
		char resolved[MAX_QPATH];

		if (DX12_FindShaderTexture(name, resolved, sizeof(resolved)))
		{
			dx12.ri.Printf(PRINT_DEVELOPER,
			               "DX12_RegisterTexture: resolved shader '%s' -> '%s'\n",
			               name, resolved);
			DX12_LoadImageSmart(resolved, &pic, &width, &height);
		}
	}

	if (!pic || width <= 0 || height <= 0)
	{
		if (SHD_WarnOnce(name))
		{
			dx12.ri.Printf(PRINT_DEVELOPER, "DX12_RegisterTexture: could not load '%s'\n", name);
		}

		// Register the name as a noshader alias so subsequent lookups for the same
		// missing texture return immediately without repeating expensive FS searches.
		if (dx12NumShaders < DX12_MAX_TEXTURES && dx12NumShaders >= 3 && dx12Shaders[2].valid)
		{
			slot = dx12NumShaders;
			Q_strncpyz(dx12Shaders[slot].name, name, sizeof(dx12Shaders[slot].name));
			dx12Shaders[slot].width         = dx12Shaders[2].width;
			dx12Shaders[slot].height        = dx12Shaders[2].height;
			dx12Shaders[slot].valid         = qtrue;
			dx12Shaders[slot].tex.resource  = NULL; // shared alias – do not release on shutdown
			dx12Shaders[slot].tex.cpuHandle = dx12Shaders[2].tex.cpuHandle;
			dx12Shaders[slot].tex.gpuHandle = dx12Shaders[2].tex.gpuHandle;
			dx12NumShaders++;
			return (qhandle_t)slot;
		}
		return 2; // slot 2 = noshader
	}

	slot = dx12NumShaders;
	tex  = DX12_CreateTextureFromRGBA((const byte *)pic, width, height, slot);
	DX12_FreeImage(pic);

	if (!tex.resource)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_RegisterTexture: GPU upload failed for '%s'\n", name);
		return 2; // slot 2 = noshader
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

	// Material handle: return the stage-0 texture
	if (idx >= DX12_MATERIAL_HANDLE_BASE)
	{
		int midx = idx - DX12_MATERIAL_HANDLE_BASE;

		if (midx >= 0 && midx < dx12NumMaterials && dx12Materials[midx].valid
		    && dx12Materials[midx].numStages > 0
		    && dx12Materials[midx].stages[0].active)
		{
			// Recurse with the raw texture handle (always < DX12_MATERIAL_HANDLE_BASE)
			return DX12_GetTexture(dx12Materials[midx].stages[0].texHandle);
		}
		// Fall through to white fallback
		idx = 0;
	}

	if (idx >= 0 && idx < dx12NumShaders && dx12Shaders[idx].valid)
	{
		// Alias entries (failed-load stand-ins) have resource==NULL but gpuHandle
		// pointing to the noshader checkerboard.  Return the real noshader slot so
		// callers that check resource!=NULL still get a valid GPU handle.
		if (!dx12Shaders[idx].tex.resource && idx != 2
		    && dx12NumShaders > 2 && dx12Shaders[2].valid && dx12Shaders[2].tex.resource)
		{
			return &dx12Shaders[2].tex;
		}
		return &dx12Shaders[idx].tex;
	}

	// Fall back to the white texture at slot 0
	if (dx12NumShaders > 0 && dx12Shaders[0].valid)
	{
		return &dx12Shaders[0].tex;
	}

	return NULL;
}

// ---------------------------------------------------------------------------
// Material parser helpers
// ---------------------------------------------------------------------------

/**
 * @brief Read one float token from the stream and return its value.
 *        Advances *pp past the consumed token.
 */
static float SH_ParseFloat(const char **pp, const char *end)
{
	char tok[64];

	*pp = SH_ReadToken(*pp, end, tok, sizeof(tok));
	return tok[0] ? (float)atof(tok) : 0.0f;
}

/**
 * @brief Map an ET blendfunc token (GL_ONE, SRC_ALPHA, etc.) to a D3D12_BLEND.
 *        Returns D3D12_BLEND_ONE for any unrecognised token.
 */
static D3D12_BLEND SH_ParseBlendFactor(const char *tok)
{
	if (!DX12_Stricmp(tok, "GL_ZERO")                || !DX12_Stricmp(tok, "ZERO"))
	{ return D3D12_BLEND_ZERO; }
	if (!DX12_Stricmp(tok, "GL_ONE")                 || !DX12_Stricmp(tok, "ONE"))
	{ return D3D12_BLEND_ONE; }
	if (!DX12_Stricmp(tok, "GL_SRC_COLOR")           || !DX12_Stricmp(tok, "SRC_COLOR"))
	{ return D3D12_BLEND_SRC_COLOR; }
	if (!DX12_Stricmp(tok, "GL_ONE_MINUS_SRC_COLOR") || !DX12_Stricmp(tok, "ONE_MINUS_SRC_COLOR"))
	{ return D3D12_BLEND_INV_SRC_COLOR; }
	if (!DX12_Stricmp(tok, "GL_DST_COLOR")           || !DX12_Stricmp(tok, "DST_COLOR"))
	{ return D3D12_BLEND_DEST_COLOR; }
	if (!DX12_Stricmp(tok, "GL_ONE_MINUS_DST_COLOR") || !DX12_Stricmp(tok, "ONE_MINUS_DST_COLOR"))
	{ return D3D12_BLEND_INV_DEST_COLOR; }
	if (!DX12_Stricmp(tok, "GL_SRC_ALPHA")           || !DX12_Stricmp(tok, "SRC_ALPHA"))
	{ return D3D12_BLEND_SRC_ALPHA; }
	if (!DX12_Stricmp(tok, "GL_ONE_MINUS_SRC_ALPHA") || !DX12_Stricmp(tok, "ONE_MINUS_SRC_ALPHA"))
	{ return D3D12_BLEND_INV_SRC_ALPHA; }
	if (!DX12_Stricmp(tok, "GL_DST_ALPHA")           || !DX12_Stricmp(tok, "DST_ALPHA"))
	{ return D3D12_BLEND_DEST_ALPHA; }
	if (!DX12_Stricmp(tok, "GL_ONE_MINUS_DST_ALPHA") || !DX12_Stricmp(tok, "ONE_MINUS_DST_ALPHA"))
	{ return D3D12_BLEND_INV_DEST_ALPHA; }
	if (!DX12_Stricmp(tok, "GL_SRC_ALPHA_SATURATE")  || !DX12_Stricmp(tok, "SRC_ALPHA_SATURATE"))
	{ return D3D12_BLEND_SRC_ALPHA_SAT; }
	return D3D12_BLEND_ONE;
}

/**
 * @brief Parse wave-function parameters: <func> <base> <amplitude> <phase> <freq>
 *        into @p wave.  Unknown func tokens default to sin.
 */
static void SH_ParseWaveParams(const char **pp, const char *end, dx12Wave_t *wave)
{
	char wfunc[32];

	*pp = SH_ReadToken(*pp, end, wfunc, sizeof(wfunc));

	if (!DX12_Stricmp(wfunc, "square"))
	{
		wave->func = DX12_WAVE_SQUARE;
	}
	else if (!DX12_Stricmp(wfunc, "triangle"))
	{
		wave->func = DX12_WAVE_TRIANGLE;
	}
	else if (!DX12_Stricmp(wfunc, "sawtooth"))
	{
		wave->func = DX12_WAVE_SAWTOOTH;
	}
	else if (!DX12_Stricmp(wfunc, "inversesawtooth"))
	{
		wave->func = DX12_WAVE_INVERSE_SAWTOOTH;
	}
	else
	{
		wave->func = DX12_WAVE_SIN;
	}

	wave->base      = SH_ParseFloat(pp, end);
	wave->amplitude = SH_ParseFloat(pp, end);
	wave->phase     = SH_ParseFloat(pp, end);
	wave->frequency = SH_ParseFloat(pp, end);
}

/**
 * @brief Parse one stage block from just after its opening '{' to (and
 *        including) the closing '}'.
 *
 * @param p      Source position (first token after the opening @c {).
 * @param end    One past the last byte of the source buffer.
 * @param stage  Stage struct to populate.
 * @return       Pointer to the first byte after the closing @c }.
 */
static const char *SH_ParseStage(const char *p, const char *end,
                                  dx12MaterialStage_t *stage)
{
	char tok[MAX_QPATH];

	stage->active   = qtrue;
	stage->srcBlend = D3D12_BLEND_ONE;
	stage->dstBlend = D3D12_BLEND_ZERO;

	while (p < end)
	{
		const char *saved;

		p = SH_ReadToken(p, end, tok, sizeof(tok));

		if (!tok[0] || tok[0] == '}')
		{
			break;
		}

		if (tok[0] == '{')
		{
			continue; // malformed – skip stray brace
		}

		// map / clampmap <path>
		if (!DX12_Stricmp(tok, "map") || !DX12_Stricmp(tok, "clampmap"))
		{
			char path[MAX_QPATH];

			p = SH_ReadToken(p, end, path, sizeof(path));
			if (!DX12_Stricmp(path, "$lightmap"))
			{
				// $lightmap is handled by useLightmap on the first diffuse stage.
				// Executing it separately multiplies the framebuffer by vertex colour
				// (which is zero for all lightmapped BSP surfaces), turning everything black.
				stage->active = qfalse;
			}
			else if (path[0] && path[0] != '$' && path[0] != '{' && path[0] != '}')
			{
				stage->texHandle = DX12_RegisterTexture(path);
			}
			continue;
		}

		// animMap <fps> <frame0> [frame1 …]
		if (!DX12_Stricmp(tok, "animMap"))
		{
			int fi;

			p              = SH_ReadToken(p, end, tok, sizeof(tok));
			stage->animFps = tok[0] ? (float)atof(tok) : 1.0f;

			for (fi = 0; fi < DX12_MAX_ANIM_FRAMES; fi++)
			{
				char frame[MAX_QPATH];

				saved = p;
				p     = SH_ReadToken(p, end, frame, sizeof(frame));

				if (!frame[0] || frame[0] == '{' || frame[0] == '}')
				{
					p = saved;
					break;
				}

				// Frame paths must contain '.' or '/' – otherwise it is
				// likely a directive keyword on the next line
				if (!strchr(frame, '.') && !strchr(frame, '/')
				    && !strchr(frame, '\\'))
				{
					p = saved;
					break;
				}

				stage->animFrames[fi] = DX12_RegisterTexture(frame);
				stage->animNumFrames++;
			}

			// Use the first frame as the primary texture if map was absent
			if (stage->animNumFrames > 0 && !stage->texHandle)
			{
				stage->texHandle = stage->animFrames[0];
			}
			continue;
		}

		// blendfunc <shorthand | srcFactor dstFactor>
		if (!DX12_Stricmp(tok, "blendfunc"))
		{
			char src[64];

			p = SH_ReadToken(p, end, src, sizeof(src));

			if (!DX12_Stricmp(src, "add"))
			{
				stage->srcBlend = D3D12_BLEND_ONE;
				stage->dstBlend = D3D12_BLEND_ONE;
			}
			else if (!DX12_Stricmp(src, "filter"))
			{
				stage->srcBlend = D3D12_BLEND_DEST_COLOR;
				stage->dstBlend = D3D12_BLEND_ZERO;
			}
			else if (!DX12_Stricmp(src, "blend"))
			{
				stage->srcBlend = D3D12_BLEND_SRC_ALPHA;
				stage->dstBlend = D3D12_BLEND_INV_SRC_ALPHA;
			}
			else
			{
				char dst[64];

				stage->srcBlend = SH_ParseBlendFactor(src);
				p               = SH_ReadToken(p, end, dst, sizeof(dst));
				stage->dstBlend = SH_ParseBlendFactor(dst);
			}
			continue;
		}

		// alphaFunc <GE128 | GT0 | LT128>
		// Maps to a PS clip() threshold stored in alphaTestThreshold.
		//   GE128  – keep pixels with alpha >= 0.5 (most common, used by foliage/fences)
		//   GT0    – keep pixels with alpha > 0 (discard fully transparent)
		//   LT128  – keep pixels with alpha < 0.5 (less common; stored as negative)
		if (!DX12_Stricmp(tok, "alphaFunc"))
		{
			char func[32];

			p = SH_ReadToken(p, end, func, sizeof(func));

			if (!DX12_Stricmp(func, "GE128"))
			{
				stage->alphaTestThreshold = 0.5f;
			}
			else if (!DX12_Stricmp(func, "GT0"))
			{
				stage->alphaTestThreshold = 0.004f; // just above zero
			}
			else if (!DX12_Stricmp(func, "LT128"))
			{
				// Negative threshold signals "clip if alpha >= |threshold|"
				stage->alphaTestThreshold = -0.5f;
			}
			continue;
		}

		// tcMod <scroll|rotate|stretch> [params…]
		if (!DX12_Stricmp(tok, "tcMod"))
		{
			char type[64];

			p = SH_ReadToken(p, end, type, sizeof(type));

			if (stage->numTcMods < DX12_MAX_TCMODS)
			{
				dx12TcMod_t *tm = &stage->tcMods[stage->numTcMods];

				if (!DX12_Stricmp(type, "scroll"))
				{
					tm->type      = DX12_TMOD_SCROLL;
					tm->scroll[0] = SH_ParseFloat(&p, end);
					tm->scroll[1] = SH_ParseFloat(&p, end);
					stage->numTcMods++;
				}
				else if (!DX12_Stricmp(type, "rotate"))
				{
					tm->type        = DX12_TMOD_ROTATE;
					tm->rotateSpeed = SH_ParseFloat(&p, end);
					stage->numTcMods++;
				}
				else if (!DX12_Stricmp(type, "stretch"))
				{
					tm->type = DX12_TMOD_STRETCH;
					SH_ParseWaveParams(&p, end, &tm->stretch);
					stage->numTcMods++;
				}
				else if (!DX12_Stricmp(type, "scale"))
				{
					tm->type     = DX12_TMOD_SCALE;
					tm->scale[0] = SH_ParseFloat(&p, end);
					tm->scale[1] = SH_ParseFloat(&p, end);
					if (tm->scale[0] == 0.0f) { tm->scale[0] = 1.0f; }
					if (tm->scale[1] == 0.0f) { tm->scale[1] = 1.0f; }
					stage->numTcMods++;
				}
				else if (!DX12_Stricmp(type, "turb"))
				{
					// tcMod turb <base> <amplitude> <phase> <frequency>
					// Pack into dx12Wave_t: base, amplitude, phase, frequency
					tm->type           = DX12_TMOD_TURB;
					tm->turb.base      = SH_ParseFloat(&p, end);
					tm->turb.amplitude = SH_ParseFloat(&p, end);
					tm->turb.phase     = SH_ParseFloat(&p, end);
					tm->turb.frequency = SH_ParseFloat(&p, end);
					tm->turb.func      = DX12_WAVE_SIN;
					stage->numTcMods++;
				}
				// Other tcMod types are recognised but ignored
			}
			continue;
		}

		// rgbGen <identity|identityLighting|vertex|exactVertex|entity|oneMinusEntity|wave|const>
		if (!DX12_Stricmp(tok, "rgbGen"))
		{
			char type[64];

			p = SH_ReadToken(p, end, type, sizeof(type));

			if (!DX12_Stricmp(type, "identity"))
			{
				stage->rgbGen = DX12_CGEN_IDENTITY;
			}
			else if (!DX12_Stricmp(type, "identityLighting"))
			{
				stage->rgbGen = DX12_CGEN_IDENTITY;
			}
			else if (!DX12_Stricmp(type, "vertex"))
			{
				stage->rgbGen = DX12_CGEN_VERTEX;
			}
			else if (!DX12_Stricmp(type, "exactVertex"))
			{
				stage->rgbGen = DX12_CGEN_EXACT_VERTEX;
			}
			else if (!DX12_Stricmp(type, "entity"))
			{
				stage->rgbGen = DX12_CGEN_ENTITY;
			}
			else if (!DX12_Stricmp(type, "oneMinusEntity"))
			{
				stage->rgbGen = DX12_CGEN_ONE_MINUS_ENTITY;
			}
			else if (!DX12_Stricmp(type, "wave"))
			{
				stage->rgbGen = DX12_CGEN_WAVEFORM;
				SH_ParseWaveParams(&p, end, &stage->rgbWave);
			}
			else if (!DX12_Stricmp(type, "const"))
			{
				// const ( r g b ) – parse three floats in optional parens
				char  tmp[64];
				float r, g, b;

				p = SH_ReadToken(p, end, tmp, sizeof(tmp));
				if (tmp[0] == '(')
				{
					r = SH_ParseFloat(&p, end);
					g = SH_ParseFloat(&p, end);
					b = SH_ParseFloat(&p, end);
					p = SH_ReadToken(p, end, tmp, sizeof(tmp)); // consume ')'
				}
				else
				{
					r = tmp[0] ? (float)atof(tmp) : 0.0f;
					g = SH_ParseFloat(&p, end);
					b = SH_ParseFloat(&p, end);
				}

				stage->rgbGen           = DX12_CGEN_CONST;
				stage->constantColor[0] = (byte)(r * 255.0f);
				stage->constantColor[1] = (byte)(g * 255.0f);
				stage->constantColor[2] = (byte)(b * 255.0f);
			}
			// Unrecognised rgbGen sub-types are silently ignored
			continue;
		}

		// alphaGen <identity|vertex|entity|wave|const>
		if (!DX12_Stricmp(tok, "alphaGen"))
		{
			char type[64];

			p = SH_ReadToken(p, end, type, sizeof(type));

			if (!DX12_Stricmp(type, "identity"))
			{
				stage->alphaGen = DX12_AGEN_IDENTITY;
			}
			else if (!DX12_Stricmp(type, "vertex"))
			{
				stage->alphaGen = DX12_AGEN_VERTEX;
			}
			else if (!DX12_Stricmp(type, "entity"))
			{
				stage->alphaGen = DX12_AGEN_ENTITY;
			}
			else if (!DX12_Stricmp(type, "wave"))
			{
				stage->alphaGen = DX12_AGEN_WAVEFORM;
				SH_ParseWaveParams(&p, end, &stage->alphaWave);
			}
			else if (!DX12_Stricmp(type, "const"))
			{
				stage->alphaGen = DX12_AGEN_CONST;
				stage->constantColor[3] = (byte)(SH_ParseFloat(&p, end) * 255.0f);
			}
			// Unrecognised alphaGen sub-types are silently ignored
			continue;
		}

		// tcGen <base|texture|lightmap|environment>
		if (!DX12_Stricmp(tok, "tcGen"))
		{
			char type[64];

			p = SH_ReadToken(p, end, type, sizeof(type));

			if (!DX12_Stricmp(type, "lightmap"))
			{
				stage->tcGen = DX12_TCGEN_LIGHTMAP;
			}
			else if (!DX12_Stricmp(type, "environment"))
			{
				stage->tcGen = DX12_TCGEN_ENVIRONMENT;
			}
			else
			{
				// "base", "texture", or anything else → default mesh UVs
				stage->tcGen = DX12_TCGEN_TEXTURE;
			}
			continue;
		}

		// stage <type> – PBR stage type qualifier used in ETL/renderer2 material files.
		// Only "diffusemap" and "liquidmap" map to a visible DX12 diffuse stage.
		// All other types (bumpmap, specularmap, normalmap, attenuationMapXY, …) are
		// PBR-only passes that DX12 doesn't render.  Mark the stage inactive and drain
		// the remaining tokens so no texture is wastefully loaded.
		if (!DX12_Stricmp(tok, "stage"))
		{
			char type[64];

			p = SH_ReadToken(p, end, type, sizeof(type));
			if (DX12_Stricmp(type, "diffusemap") != 0 && DX12_Stricmp(type, "liquidmap") != 0)
			{
				stage->active = qfalse;
				while (p < end)
				{
					p = SH_ReadToken(p, end, tok, sizeof(tok));
					if (!tok[0] || tok[0] == '}')
					{
						break;
					}
				}
				return p;
			}
			continue;
		}

		// bumpmap / normalmap / specularmap / heightmap <path> inside a stage block:
		// PBR texture references that DX12 does not use.  Consume the path argument
		// so it is not misread as the next directive keyword.
		if (!DX12_Stricmp(tok, "bumpmap") || !DX12_Stricmp(tok, "normalmap")
		    || !DX12_Stricmp(tok, "specularmap") || !DX12_Stricmp(tok, "heightmap"))
		{
			char path[MAX_QPATH];

			p = SH_ReadToken(p, end, path, sizeof(path));
			continue;
		}

		// All other stage directives are silently skipped
	}

	return p;
}

// ---------------------------------------------------------------------------
// Full material parser
// ---------------------------------------------------------------------------

/**
 * @brief Search a single shader-file buffer for a shader whose name equals
 *        @p shaderName and build a full dx12Material_t from it.
 *
 * Parses outer-level directives (surfaceparm) and delegates each stage
 * block to SH_ParseStage().
 *
 * @param[in]  buf         Raw text content of the shader file.
 * @param[in]  bufLen      Number of bytes in @p buf.
 * @param[in]  shaderName  Shader name to look for (case-insensitive).
 * @param[out] out         Material struct to populate on success.
 * @return     qtrue when the shader was found and @p out was filled.
 */
static qboolean SH_ParseMaterialInBuffer(const char *buf, int bufLen,
                                          const char *shaderName,
                                          dx12Material_t *out)
{
	const char *p   = buf;
	const char *end = buf + bufLen;
	char        tok[MAX_QPATH];

	Com_Memset(out, 0, sizeof(*out));

	while (p < end)
	{
		p = SH_ReadToken(p, end, tok, sizeof(tok));

		if (!tok[0])
		{
			break;
		}

		// Skip stray braces at the top level (defensive)
		if (tok[0] == '{' || tok[0] == '}')
		{
			continue;
		}

		// tok is a potential shader name; the next non-comment token must be '{'
		{
			const char *saved = p;
			char        brace[4];

			p = SH_ReadToken(p, end, brace, sizeof(brace));

			if (brace[0] != '{')
			{
				p = saved;
				continue;
			}

			if (DX12_Stricmp(tok, shaderName) != 0)
			{
				// Not the shader we want – skip to the matching '}'
				int depth = 1;

				while (p < end && depth > 0)
				{
					p = SH_ReadToken(p, end, tok, sizeof(tok));
					if (tok[0] == '{')
					{
						depth++;
					}
					else if (tok[0] == '}')
					{
						depth--;
					}
				}
				continue;
			}

			// We are inside the matching shader block.
			Q_strncpyz(out->name, shaderName, sizeof(out->name));

			while (p < end)
			{
				p = SH_ReadToken(p, end, tok, sizeof(tok));

				if (!tok[0] || tok[0] == '}')
				{
					break; // end of shader block
				}

				if (tok[0] == '{')
				{
					// Stage block
					if (out->numStages < DX12_MAX_MATERIAL_STAGES)
					{
						p = SH_ParseStage(p, end, &out->stages[out->numStages]);
						if (out->stages[out->numStages].active)
						{
							out->numStages++;
						}
					}
					else
					{
						// Excess stages – skip
						int depth = 1;

						while (p < end && depth > 0)
						{
							p = SH_ReadToken(p, end, tok, sizeof(tok));
							if (tok[0] == '{')
							{
								depth++;
							}
							else if (tok[0] == '}')
							{
								depth--;
							}
						}
					}
					continue;
				}

				// surfaceparm <keyword>
				if (!DX12_Stricmp(tok, "surfaceparm"))
				{
					char sparm[MAX_QPATH];

					p = SH_ReadToken(p, end, sparm, sizeof(sparm));

					if (!DX12_Stricmp(sparm, "sky"))
					{
						out->isSky = qtrue;
					}
					else if (!DX12_Stricmp(sparm, "fog"))
					{
						out->isFog = qtrue;
					}
					else if (!DX12_Stricmp(sparm, "trans")
					         || !DX12_Stricmp(sparm, "translucent"))
					{
						out->isTranslucent = qtrue;
					}
					else if (!DX12_Stricmp(sparm, "nodraw"))
					{
						out->isNodraw = qtrue;
					}
					else if (!DX12_Stricmp(sparm, "noimpact"))
					{
						out->noImpact = qtrue;
					}
					else if (!DX12_Stricmp(sparm, "nomarks"))
					{
						out->noMarks = qtrue;
					}
					continue;
				}

				// diffusemap <path> – top-level PBR diffuse shorthand (ETL material format).
				// Creates stage-0 with the given texture so materials with no explicit stage
				// blocks still render correctly instead of falling back to blank/white.
				if (!DX12_Stricmp(tok, "diffusemap"))
				{
					char path[MAX_QPATH];

					p = SH_ReadToken(p, end, path, sizeof(path));
					if (path[0] && out->numStages == 0)
					{
						dx12MaterialStage_t *s = &out->stages[0];

						s->active    = qtrue;
						s->srcBlend  = D3D12_BLEND_ONE;
						s->dstBlend  = D3D12_BLEND_ZERO;
						s->texHandle = DX12_RegisterTexture(path);
						if (s->texHandle)
						{
							out->numStages = 1;
						}
					}
					continue;
				}

				// implicitMap / implicitMask / implicitBlend <path|-> – shorthand that creates
				// a standard lightmapped stage from a single texture.  The special path "-"
				// means "use the shader name as the texture path".  Treated as diffusemap for DX12.
				if (!DX12_Stricmp(tok, "implicitMap") || !DX12_Stricmp(tok, "implicitMask")
				    || !DX12_Stricmp(tok, "implicitBlend"))
				{
					char path[MAX_QPATH];

					p = SH_ReadToken(p, end, path, sizeof(path));
					if (out->numStages == 0)
					{
						const char          *texPath = (path[0] && path[0] != '-') ? path : shaderName;
						dx12MaterialStage_t *s       = &out->stages[0];

						s->active    = qtrue;
						s->srcBlend  = D3D12_BLEND_ONE;
						s->dstBlend  = D3D12_BLEND_ZERO;
						s->texHandle = DX12_RegisterTexture(texPath);
						if (s->texHandle)
						{
							out->numStages = 1;
						}
					}
					continue;
				}

				// bumpmap / normalmap / specularmap / heightmap <path> at the outer shader level:
				// PBR texture references that DX12 doesn't render.  Consume the path argument to
				// prevent it from being misread as a directive keyword on the next iteration.
				if (!DX12_Stricmp(tok, "bumpmap") || !DX12_Stricmp(tok, "normalmap")
				    || !DX12_Stricmp(tok, "specularmap") || !DX12_Stricmp(tok, "heightmap"))
				{
					char path[MAX_QPATH];

					p = SH_ReadToken(p, end, path, sizeof(path));
					continue;
				}

				// cull <mode> - face culling override.
// "none", "twosided", "disable" render both faces.
// "back" (default) or "front" are ignored (default PSO handles back-face cull).
if (!DX12_Stricmp(tok, "cull"))
{
char mode[32];

p = SH_ReadToken(p, end, mode, sizeof(mode));
if (!DX12_Stricmp(mode, "none") || !DX12_Stricmp(mode, "twosided")
    || !DX12_Stricmp(mode, "disable"))
{
out->isDoubleSided = qtrue;
}
continue;
}

// sort <value> - render order hint.
// Named or numeric sort keys in the translucent range promote the material
// to isTranslucent so it is drawn after opaque geometry.
if (!DX12_Stricmp(tok, "sort"))
{
char sortVal[32];

p = SH_ReadToken(p, end, sortVal, sizeof(sortVal));
if (sortVal[0])
{
if (!DX12_Stricmp(sortVal, "nearest") || !DX12_Stricmp(sortVal, "additive")
    || !DX12_Stricmp(sortVal, "banner") || !DX12_Stricmp(sortVal, "underwater")
    || !DX12_Stricmp(sortVal, "translucent"))
{
out->isTranslucent = qtrue;
}
else
{
// Numeric sort keys >= 6 are in the translucent range.
int sv = atoi(sortVal);
if (sv >= 6)
{
out->isTranslucent = qtrue;
}
}
}
continue;
}

// skyParms <outerbox> <cloudheight> <innerbox>
// Load the 6 outer-box face textures (named <outerbox>_rt/bk/lf/ft/up/dn).
// "-" means no sky box (ignore).  Matched to renderer1 tr_shader.c ParseSkyParms.
if (!DX12_Stricmp(tok, "skyParms"))
{
    char skyName[MAX_QPATH];
    char tmp[MAX_QPATH];
    int  fi;
    static const char *suf[6] = { "rt", "bk", "lf", "ft", "up", "dn" };

    p = SH_ReadToken(p, end, skyName, sizeof(skyName));
    // consume cloudheight and innerbox tokens
    { char discard[32]; p = SH_ReadToken(p, end, discard, sizeof(discard)); }
    { char discard[MAX_QPATH]; p = SH_ReadToken(p, end, discard, sizeof(discard)); }

    if (skyName[0] && skyName[0] != '-')
    {
        for (fi = 0; fi < 6; fi++)
        {
            Com_sprintf(tmp, sizeof(tmp), "%s_%s.tga", skyName, suf[fi]);
            out->skyOuterBox[fi] = DX12_RegisterTexture(tmp);
            if (!out->skyOuterBox[fi])
            {
                // try without .tga extension (game might provide it differently)
                Com_sprintf(tmp, sizeof(tmp), "%s_%s", skyName, suf[fi]);
                out->skyOuterBox[fi] = DX12_RegisterTexture(tmp);
            }
        }
    }
    continue;
}

// All other outer-level directives (deformvertexes, fogparms, etc.) are skipped
			}

			out->valid = qtrue;
			return qtrue;
		}
	}

	return qfalse;
}

/**
 * @brief Scan all .shader files in @p dir for a definition of @p shaderName.
 *
 * @param[in]  dir         Shader directory to search (e.g. "materials").
 * @param[in]  shaderName  Shader name to look for.
 * @param[out] out         Material struct to populate on success.
 * @return     qtrue when the shader was found and @p out was filled.
 */
static qboolean SH_ScanDirForMaterial(const char *dir, const char *shaderName,
                                       dx12Material_t *out)
{
	char     **fileList;
	int        numFiles = 0;
	int        i;
	qboolean   found    = qfalse;

	fileList = dx12.ri.FS_ListFiles(dir, ".shader", &numFiles);
	if (!fileList || numFiles <= 0)
	{
		if (fileList)
		{
			dx12.ri.FS_FreeFileList(fileList);
		}
		return qfalse;
	}

	for (i = 0; i < numFiles && !found; i++)
	{
		char  fullPath[MAX_QPATH];
		void *buf  = NULL;
		int   size = 0;

		snprintf(fullPath, sizeof(fullPath), "%s/%s", dir, fileList[i]);
		size = dx12.ri.FS_ReadFile(fullPath, &buf);
		if (size <= 0 || !buf)
		{
			continue;
		}

		found = SH_ParseMaterialInBuffer((const char *)buf, size,
		                                 shaderName, out);
		dx12.ri.FS_FreeFile(buf);
	}

	dx12.ri.FS_FreeFileList(fileList);
	return found;
}

/**
 * @brief Attempt to parse a full material by scanning the shader-script
 *        directories (materials/ first, then scripts/).
 *
 * @param[in]  shaderName  Shader name to look for.
 * @param[out] out         Material struct to populate on success.
 * @return     qtrue when the shader was found and @p out was filled.
 */
static qboolean DX12_FindMaterialScript(const char *shaderName, dx12Material_t *out)
{
	if (SH_ScanDirForMaterial("materials", shaderName, out))
	{
		return qtrue;
	}
	if (SH_ScanDirForMaterial("scripts", shaderName, out))
	{
		return qtrue;
	}
	return qfalse;
}

// ---------------------------------------------------------------------------
// DX12_RegisterMaterial
// ---------------------------------------------------------------------------

/**
 * @brief DX12_RegisterMaterial
 * @param[in] name  Shader or image-file name to register.
 * @return          Material handle in [DX12_MATERIAL_HANDLE_BASE, …),
 *                  or 0 on failure.
 *
 * Searches the material cache first (deduplication by name, case-insensitive).
 * On a miss, tries to parse a full dx12Material_t from the shader-script
 * directories.  If no script is found, falls back to a single-stage material
 * built from a direct image load (matching the legacy DX12_RegisterTexture
 * behaviour for plain image references).
 */
qhandle_t DX12_RegisterMaterial(const char *name)
{
	int            i;
	int            slot;
	dx12Material_t mat;
	const char    *resolvedName;

	if (!name || !name[0])
	{
		return 0;
	}

	// Apply shader remap table (mirrors GL's r_remappedShader path).
	resolvedName = DX12_GetRemappedShader(name);
	if (!resolvedName)
	{
		resolvedName = name;
	}

	// Deduplicate: return existing handle if already cached
	for (i = 0; i < dx12NumMaterials; i++)
	{
		if (dx12Materials[i].valid && !DX12_Stricmp(dx12Materials[i].name, resolvedName))
		{
			return (qhandle_t)(DX12_MATERIAL_HANDLE_BASE + i);
		}
	}

	if (dx12NumMaterials >= DX12_MAX_MATERIALS)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_RegisterMaterial: material cache full\n");
		return 0;
	}

	Com_Memset(&mat, 0, sizeof(mat));

	// Try to parse a full material from .shader script files
	if (!DX12_FindMaterialScript(resolvedName, &mat))
	{
		// Fall back: treat the name as a plain image path and build a 1-stage material
		qhandle_t texHandle = DX12_RegisterTexture(resolvedName);

		// Build a 1-stage material from the texture (always succeeds: failed loads
		// return a noshader alias slot so no null-check is needed here).
		Q_strncpyz(mat.name, resolvedName, sizeof(mat.name));
		mat.stages[0].active   = qtrue;
		mat.stages[0].texHandle = texHandle;
		mat.stages[0].srcBlend = D3D12_BLEND_ONE;
		mat.stages[0].dstBlend = D3D12_BLEND_ZERO;
		mat.numStages          = 1;
		mat.valid              = qtrue;
	}
	else
	{
		dx12.ri.Printf(PRINT_DEVELOPER,
		               "DX12_RegisterMaterial: parsed shader '%s' (%d stage(s))\n",
		               resolvedName, mat.numStages);
	}

	slot              = dx12NumMaterials;
	dx12Materials[slot] = mat;
	dx12NumMaterials++;

	return (qhandle_t)(DX12_MATERIAL_HANDLE_BASE + slot);
}

// ---------------------------------------------------------------------------
// DX12_RegisterMaterialFromText
// ---------------------------------------------------------------------------

/**
 * @brief Parse a shader block from an in-memory text buffer and register it
 *        as a DX12 material.  Used by RE_DX12_LoadDynamicShader.
 *
 * @param[in] name        Shader name (cache key).
 * @param[in] shadertext  Full shader script text.
 * @return qtrue on success, qfalse on parse failure or cache overflow.
 */
qboolean DX12_RegisterMaterialFromText(const char *name, const char *shadertext)
{
	dx12Material_t mat;
	int            slot;

	if (!name || !name[0] || !shadertext || !shadertext[0])
	{
		return qfalse;
	}

	if (dx12NumMaterials >= DX12_MAX_MATERIALS)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_RegisterMaterialFromText: material cache full\n");
		return qfalse;
	}

	Com_Memset(&mat, 0, sizeof(mat));

	if (!SH_ParseMaterialInBuffer(shadertext, (int)strlen(shadertext), name, &mat))
	{
		dx12.ri.Printf(PRINT_DEVELOPER,
		               "DX12_RegisterMaterialFromText: failed to parse shader '%s'\n", name);
		return qfalse;
	}

	// Deduplicate: update existing entry if already cached
	{
		int i;
		for (i = 0; i < dx12NumMaterials; i++)
		{
			if (dx12Materials[i].valid && !DX12_Stricmp(dx12Materials[i].name, name))
			{
				dx12Materials[i] = mat;
				return qtrue;
			}
		}
	}

	slot                = dx12NumMaterials;
	dx12Materials[slot] = mat;
	dx12NumMaterials++;

	dx12.ri.Printf(PRINT_DEVELOPER,
	               "DX12_RegisterMaterialFromText: registered '%s' (%d stage(s))\n",
	               name, mat.numStages);
	return qtrue;
}

// ---------------------------------------------------------------------------
// DX12_GetMaterial
// ---------------------------------------------------------------------------

/**
 * @brief DX12_GetMaterial
 * @param[in] handle  Handle returned by DX12_RegisterMaterial().
 * @return            Pointer to the cached dx12Material_t, or NULL if
 *                    the handle is out of range or the slot is invalid.
 */
dx12Material_t *DX12_GetMaterial(qhandle_t handle)
{
	int idx = (int)handle - DX12_MATERIAL_HANDLE_BASE;

	if (idx >= 0 && idx < dx12NumMaterials && dx12Materials[idx].valid)
	{
		return &dx12Materials[idx];
	}
	return NULL;
}

#endif // _WIN32
