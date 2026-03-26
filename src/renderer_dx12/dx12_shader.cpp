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
 * @note Texture registration must happen while no frame is open (i.e. before
 *       the first DX12_BeginFrame call or after DX12_EndFrame),
 *       because DX12_CreateTextureFromRGBA resets the main command allocator.
 */

#include "dx12_shader.h"
#include "dx12_image.h"

#ifdef _WIN32

#include <string.h>  // strncpy, strchr, etc.

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

	// Guard against registration during an open frame
	if (dx12.frameOpen)
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "DX12_RegisterTexture: '%s' registered mid-frame; deferring is not supported\n",
		               name);
		return 0;
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
