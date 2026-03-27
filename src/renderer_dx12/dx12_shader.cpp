/**
 * @file dx12_shader.cpp
 * @brief DX12 texture registry.
 *
 * Wraps DX12_LoadImage + DX12_CreateTextureFromRGBA into a simple name-keyed
 * table of up to DX12_MAX_TEXTURES entries.  Index 0 is a permanent 1×1
 * opaque-white fallback.
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

#ifdef _WIN32

#include <string.h>   // strncpy, strchr, etc.
#include <stdlib.h>   // atof

dx12ShaderEntry_t dx12Shaders[DX12_MAX_TEXTURES];
int               dx12NumShaders = 0;

dx12Material_t dx12Materials[DX12_MAX_MATERIALS];
int            dx12NumMaterials = 0;

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
	byte black[4] = {   0,   0,   0, 255 };

	Com_Memset(dx12Shaders, 0, sizeof(dx12Shaders));
	dx12NumShaders = 0;

	Com_Memset(dx12Materials, 0, sizeof(dx12Materials));
	dx12NumMaterials = 0;

	// Reset the one-time missing-asset warning table so that every map load
	// gets a fresh set of warnings for its own assets.
	s_numMissingNames = 0;

	// Slot 0: 1×1 opaque-white fallback
	{
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

	// Slot 1: 1×1 opaque-black fallback (used by *black virtual textures)
	{
		dx12Texture_t fallback = DX12_CreateTextureFromRGBA(black, 1, 1, 1);

		if (fallback.resource)
		{
			Q_strncpyz(dx12Shaders[1].name, "__black__", sizeof(dx12Shaders[1].name));
			dx12Shaders[1].width  = 1;
			dx12Shaders[1].height = 1;
			dx12Shaders[1].tex    = fallback;
			dx12Shaders[1].valid  = qtrue;
			dx12NumShaders        = 2;
		}
		else
		{
			dx12.ri.Printf(PRINT_WARNING, "DX12_InitTextures: black fallback texture failed\n");
			dx12NumShaders = 2; // still reserve slot 1
		}
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

	if (!name || !name[0])
	{
		return 0;
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
		return 0;
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

	// Try loading the name directly as an image file
	DX12_LoadImage(name, &pic, &width, &height);

	// If direct loading failed, try resolving via shader scripts
	if (!pic || width <= 0 || height <= 0)
	{
		char resolved[MAX_QPATH];

		if (DX12_FindShaderTexture(name, resolved, sizeof(resolved)))
		{
			dx12.ri.Printf(PRINT_DEVELOPER,
			               "DX12_RegisterTexture: resolved shader '%s' -> '%s'\n",
			               name, resolved);
			DX12_LoadImage(resolved, &pic, &width, &height);
		}
	}

	if (!pic || width <= 0 || height <= 0)
	{
		if (SHD_WarnOnce(name))
		{
			dx12.ri.Printf(PRINT_DEVELOPER, "DX12_RegisterTexture: could not load '%s'\n", name);
		}
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
			if (path[0] && path[0] != '$' && path[0] != '{' && path[0] != '}')
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
					char wfunc[32];

					tm->type = DX12_TMOD_STRETCH;
					p        = SH_ReadToken(p, end, wfunc, sizeof(wfunc));

					if (!DX12_Stricmp(wfunc, "square"))
					{
						tm->stretch.func = DX12_WAVE_SQUARE;
					}
					else if (!DX12_Stricmp(wfunc, "triangle"))
					{
						tm->stretch.func = DX12_WAVE_TRIANGLE;
					}
					else if (!DX12_Stricmp(wfunc, "sawtooth"))
					{
						tm->stretch.func = DX12_WAVE_SAWTOOTH;
					}
					else if (!DX12_Stricmp(wfunc, "inversesawtooth"))
					{
						tm->stretch.func = DX12_WAVE_INVERSE_SAWTOOTH;
					}
					else
					{
						tm->stretch.func = DX12_WAVE_SIN; // default: sin
					}

					tm->stretch.base      = SH_ParseFloat(&p, end);
					tm->stretch.amplitude = SH_ParseFloat(&p, end);
					tm->stretch.phase     = SH_ParseFloat(&p, end);
					tm->stretch.frequency = SH_ParseFloat(&p, end);
					stage->numTcMods++;
				}
				// Other tcMod types (scale, turb, etc.) are recognised but ignored
			}
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
					continue;
				}

				// All other outer-level directives (cull, sort, deformvertexes, etc.) are skipped
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

	if (!name || !name[0])
	{
		return 0;
	}

	// Deduplicate: return existing handle if already cached
	for (i = 0; i < dx12NumMaterials; i++)
	{
		if (dx12Materials[i].valid && !DX12_Stricmp(dx12Materials[i].name, name))
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
	if (!DX12_FindMaterialScript(name, &mat))
	{
		// Fall back: treat the name as a plain image path and build a 1-stage material
		qhandle_t texHandle = DX12_RegisterTexture(name);

		if (!texHandle)
		{
			if (SHD_WarnOnce(name))
			{
				dx12.ri.Printf(PRINT_DEVELOPER,
				               "DX12_RegisterMaterial: could not resolve '%s'\n", name);
			}
			return 0;
		}

		Q_strncpyz(mat.name, name, sizeof(mat.name));
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
		               name, mat.numStages);
	}

	slot              = dx12NumMaterials;
	dx12Materials[slot] = mat;
	dx12NumMaterials++;

	return (qhandle_t)(DX12_MATERIAL_HANDLE_BASE + slot);
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
