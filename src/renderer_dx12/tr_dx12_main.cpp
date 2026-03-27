/**
 * @file tr_dx12_main.cpp
 * @brief DirectX 12 renderer – main entry point and refexport_t implementations
 */

#include "tr_dx12_local.h"
#include "dx12_shader.h"
#include "dx12_poly.h"
#include "dx12_world.h"
#include "dx12_scene.h"
#include "dx12_model.h"

#ifdef _WIN32

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

#include <string.h>  // memcpy

dx12Globals_t dx12;

// ---------------------------------------------------------------------------
// Scratch texture pool – used by RE_DX12_UploadCinematic / DrawStretchRaw
// ---------------------------------------------------------------------------

/** Maximum cinematic client slots, mirroring the GL renderer's scratchImage[]. */
#define DX12_MAX_SCRATCH_IMAGES 8

/**
 * @brief SRV heap slots reserved for scratch textures.
 *
 * These occupy the last DX12_MAX_SCRATCH_IMAGES entries of the SRV heap
 * (indices DX12_MAX_TEXTURES-8 … DX12_MAX_TEXTURES-1).  Regular texture
 * registration stops well before hitting this range in practice.
 */
#define DX12_SCRATCH_SRV_BASE (DX12_MAX_TEXTURES - DX12_MAX_SCRATCH_IMAGES)

/**
 * @struct dx12ScratchTex_t
 * @brief One per-client scratch texture slot for cinematic / video frames.
 */
typedef struct
{
	ID3D12Resource              *resource;  ///< GPU texture resource (or NULL)
	D3D12_CPU_DESCRIPTOR_HANDLE  cpuHandle; ///< SRV CPU handle in the SRV heap
	D3D12_GPU_DESCRIPTOR_HANDLE  gpuHandle; ///< SRV GPU handle for binding
	int                          width;     ///< Current texture width in pixels
	int                          height;    ///< Current texture height in pixels
	qboolean                     valid;     ///< qtrue once successfully created
} dx12ScratchTex_t;

static dx12ScratchTex_t dx12ScratchTex[DX12_MAX_SCRATCH_IMAGES];

// ---------------------------------------------------------------------------
// Scratch texture cleanup (called from R_DX12_Shutdown in tr_dx12_backend.cpp)
// ---------------------------------------------------------------------------

/**
 * @brief DX12_ShutdownScratchTextures
 *
 * Releases all GPU resources held by the cinematic scratch texture pool.
 * Must be called before the D3D12 device is destroyed.
 */
void DX12_ShutdownScratchTextures(void)
{
	int i;

	for (i = 0; i < DX12_MAX_SCRATCH_IMAGES; i++)
	{
		if (dx12ScratchTex[i].resource)
		{
			dx12ScratchTex[i].resource->Release();
			dx12ScratchTex[i].resource = NULL;
		}
		dx12ScratchTex[i].valid = qfalse;
	}
}

// ---------------------------------------------------------------------------
// Stub implementations required by refexport_t
// ---------------------------------------------------------------------------

static void RE_DX12_BeginRegistration(glconfig_t *config)
{
	cvar_t *r_mode;
	cvar_t *r_customwidth;
	cvar_t *r_customheight;

	if (!config)
	{
		return;
	}

	Com_Memset(config, 0, sizeof(*config));

	// Make sure core DX12 is initialized before any texture/shader registration
	if ( !dx12.initialized )
	{
		R_DX12_Init( );   // creates device, queues, heaps, PSO, VB, textures, etc.
	}

	r_mode        = dx12.ri.Cvar_Get("r_mode", "4", 0);
	r_customwidth = dx12.ri.Cvar_Get("r_customwidth", "1280", 0);
	r_customheight = dx12.ri.Cvar_Get("r_customheight", "720", 0);

	// Pick a sensible default resolution
	if (r_mode && r_mode->integer == -2)
	{
		config->vidWidth  = r_customwidth ? r_customwidth->integer : 1280;
		config->vidHeight = r_customheight ? r_customheight->integer : 720;
	}
	else
	{
		config->vidWidth  = 1280;
		config->vidHeight = 720;
	}

	config->windowWidth  = config->vidWidth;
	config->windowHeight = config->vidHeight;

	Q_strncpyz(config->renderer_string, "DirectX 12", sizeof(config->renderer_string));
	Q_strncpyz(config->vendor_string, "Microsoft", sizeof(config->vendor_string));
	Q_strncpyz(config->version_string, "12.0", sizeof(config->version_string));

	config->colorBits    = 32;
	config->depthBits    = 24;
	config->stencilBits  = 8;
	config->isFullscreen = qfalse;
	config->windowAspect = (float)config->vidWidth / (float)config->vidHeight;
	config->displayAspect = config->windowAspect;

	dx12.vidWidth  = config->vidWidth;
	dx12.vidHeight = config->vidHeight;
}

static void RE_DX12_EndRegistration(void)
{
	// Nothing to do for a minimal implementation
}

static void RE_DX12_RenderScene(const refdef_t *fd)
{
	DX12_RenderScene(fd);
}

static void RE_DX12_SetColor(const float *rgba)
{
	if (rgba)
	{
		dx12.color2D[0] = rgba[0];
		dx12.color2D[1] = rgba[1];
		dx12.color2D[2] = rgba[2];
		dx12.color2D[3] = rgba[3];
	}
	else
	{
		dx12.color2D[0] = 1.0f;
		dx12.color2D[1] = 1.0f;
		dx12.color2D[2] = 1.0f;
		dx12.color2D[3] = 1.0f;
	}
}

static void RE_DX12_DrawStretchPic(float x, float y, float w, float h,
                                    float s1, float t1, float s2, float t2,
                                    qhandle_t hShader)
{
	DX12_DrawStretchPic(x, y, w, h, s1, t1, s2, t2, hShader);
}

static void RE_DX12_DrawRotatedPic(float x, float y, float w, float h,
                                    float s1, float t1, float s2, float t2,
                                    qhandle_t hShader, float angle)
{
	DX12_DrawRotatedPic(x, y, w, h, s1, t1, s2, t2, hShader, angle);
}

static void RE_DX12_DrawStretchPicGradient(float x, float y, float w, float h,
                                            float s1, float t1, float s2, float t2,
                                            qhandle_t hShader, const float *gradientColor,
                                            int gradientType)
{
	DX12_DrawStretchPicGradient(x, y, w, h, s1, t1, s2, t2,
	                            hShader, gradientColor, gradientType);
}

static void RE_DX12_Add2dPolys(polyVert_t *polys, int numverts, qhandle_t hShader)
{
	DX12_Add2dPolys(polys, numverts, hShader);
}

static void RE_DX12_UploadCinematic(int w, int h, int cols, int rows,
                                     const byte *data, int client, qboolean dirty)
{
	dx12ScratchTex_t *scratch;
	int               srvSlot;
	dx12Texture_t     tex;

	(void)w; (void)h; // w/h are display dimensions, not texture size; use cols/rows

	if (!dx12.initialized || !dx12.device)
	{
		return;
	}

	if (client < 0 || client >= DX12_MAX_SCRATCH_IMAGES)
	{
		return;
	}

	if (!data || cols <= 0 || rows <= 0)
	{
		return;
	}

	scratch = &dx12ScratchTex[client];
	srvSlot = DX12_SCRATCH_SRV_BASE + client;

	// Nothing to do if the texture is up-to-date
	if (scratch->valid && !dirty &&
	    scratch->width == cols && scratch->height == rows)
	{
		return;
	}

	// Release the old GPU resource before (re-)creating the texture
	if (scratch->resource)
	{
		scratch->resource->Release();
		scratch->resource = NULL;
		scratch->valid    = qfalse;
	}

	// Create (or recreate) the scratch texture via the shared upload helper
	tex = DX12_CreateTextureFromRGBA(data, cols, rows, srvSlot);

	if (tex.resource)
	{
		scratch->resource  = tex.resource;
		scratch->cpuHandle = tex.cpuHandle;
		scratch->gpuHandle = tex.gpuHandle;
		scratch->width     = cols;
		scratch->height    = rows;
		scratch->valid     = qtrue;
	}
	else
	{
		dx12.ri.Printf(PRINT_DEVELOPER,
		               "RE_DX12_UploadCinematic: upload failed for client %d (%dx%d)\n",
		               client, cols, rows);
	}
}

static void RE_DX12_DrawStretchRaw(int x, int y, int w, int h, int cols, int rows,
                                    const byte *data, int client, qboolean dirty)
{
	dx12ScratchTex_t *scratch;
	float             nx1, ny1, nx2, ny2;
	float             r, g, b, a;
	dx12QuadVertex_t  corners[4];

	RE_DX12_UploadCinematic(w, h, cols, rows, data, client, dirty);

	if (!dx12.frameOpen)
	{
		return;
	}

	if (client < 0 || client >= DX12_MAX_SCRATCH_IMAGES)
	{
		return;
	}

	scratch = &dx12ScratchTex[client];
	if (!scratch->valid)
	{
		return;
	}

	// NDC conversion (matches dx12_poly.cpp's NDC_X / NDC_Y macros)
	nx1 = ((float)x / (float)dx12.vidWidth) * 2.0f - 1.0f;
	ny1 = 1.0f - ((float)y / (float)dx12.vidHeight) * 2.0f;
	nx2 = ((float)(x + w) / (float)dx12.vidWidth) * 2.0f - 1.0f;
	ny2 = 1.0f - ((float)(y + h) / (float)dx12.vidHeight) * 2.0f;

	r = dx12.color2D[0];
	g = dx12.color2D[1];
	b = dx12.color2D[2];
	a = dx12.color2D[3];

	// TL (top-left)
	corners[0].pos[0] = nx1; corners[0].pos[1] = ny1;
	corners[0].uv[0]  = 0.0f; corners[0].uv[1] = 0.0f;
	corners[0].color[0] = r; corners[0].color[1] = g;
	corners[0].color[2] = b; corners[0].color[3] = a;

	// TR (top-right)
	corners[1].pos[0] = nx2; corners[1].pos[1] = ny1;
	corners[1].uv[0]  = 1.0f; corners[1].uv[1] = 0.0f;
	corners[1].color[0] = r; corners[1].color[1] = g;
	corners[1].color[2] = b; corners[1].color[3] = a;

	// BL (bottom-left)
	corners[2].pos[0] = nx1; corners[2].pos[1] = ny2;
	corners[2].uv[0]  = 0.0f; corners[2].uv[1] = 1.0f;
	corners[2].color[0] = r; corners[2].color[1] = g;
	corners[2].color[2] = b; corners[2].color[3] = a;

	// BR (bottom-right)
	corners[3].pos[0] = nx2; corners[3].pos[1] = ny2;
	corners[3].uv[0]  = 1.0f; corners[3].uv[1] = 1.0f;
	corners[3].color[0] = r; corners[3].color[1] = g;
	corners[3].color[2] = b; corners[3].color[3] = a;

	DX12_Begin2DBatch(scratch->gpuHandle, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	DX12_AddQuadToBatch(corners);
	DX12_Flush2DBatch();
}

static void RE_DX12_BeginFrame(void)
{
	DX12_BeginFrame();
}

static void RE_DX12_EndFrame(int *frontEndMsec, int *backEndMsec)
{
	if (frontEndMsec)
	{
		*frontEndMsec = 0;
	}
	if (backEndMsec)
	{
		*backEndMsec = 0;
	}

	// Flush any pending batched 2D draw calls before closing the command list
	DX12_Flush2D();

	DX12_EndFrame();
}

// ---------------------------------------------------------------------------
// Minimal model registry
// ---------------------------------------------------------------------------

/** Maximum number of simultaneously registered models (matches classic renderer). */
#define DX12_MAX_MOD_KNOWN 2048

static char dx12ModelNames[DX12_MAX_MOD_KNOWN][MAX_QPATH];
static int  dx12NumModels = 0;

/**
 * @brief RE_DX12_RegisterModel
 *
 * Returns a stable, non-zero handle for the given model filename.
 * The DX12 renderer does not yet render skeletal models, but a valid handle
 * is required by the animation system (bg_animgroup.c / USE_MDXFILE) so that
 * animation pools are correctly keyed and populated.
 */
static qhandle_t RE_DX12_RegisterModel(const char *name)
{
	int i;
	int slot;

	if (!name || !name[0])
	{
		return 0;
	}

	// Return an existing handle if already registered
	for (i = 0; i < dx12NumModels; i++)
	{
		if (!Q_stricmp(dx12ModelNames[i], name))
		{
			return (qhandle_t)(i + 1);
		}
	}

	// Register a new slot
	if (dx12NumModels >= DX12_MAX_MOD_KNOWN)
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "RE_DX12_RegisterModel: model table full, dropping '%s'\n",
		               name);
		return 0;
	}

	slot = dx12NumModels;
	Q_strncpyz(dx12ModelNames[slot], name, MAX_QPATH);
	dx12NumModels++;

	// Attempt to load MD3 geometry into GPU buffers.
	// MDX/MDM (skeletal) files are silently skipped: the handle is still
	// valid for the animation system even without GPU geometry.
	DX12_LoadMD3(slot, name);

	return (qhandle_t)(slot + 1);
}

static qhandle_t RE_DX12_RegisterModelAllLODs(const char *name)
{
	return RE_DX12_RegisterModel(name);
}

static qhandle_t RE_DX12_RegisterSkin(const char *name)
{
	(void)name; return 0;
}

static qhandle_t RE_DX12_RegisterShader(const char *name)
{
	return DX12_RegisterMaterial(name);
}

static qhandle_t RE_DX12_RegisterShaderNoMip(const char *name)
{
	return DX12_RegisterMaterial(name);
}

// ---------------------------------------------------------------------------
// RE_DX12_RegisterFont helpers
// ---------------------------------------------------------------------------

/** @brief Read a little-endian 32-bit int and advance the read pointer. */
static int DX12_ReadInt32(const unsigned char **p)
{
	int v = (int)((*p)[0])
	        | ((int)((*p)[1]) << 8)
	        | ((int)((*p)[2]) << 16)
	        | ((int)((*p)[3]) << 24);
	*p += 4;
	return v;
}

/** @brief Read a little-endian IEEE 754 float and advance the read pointer. */
static float DX12_ReadFloat32(const unsigned char **p)
{
	float v;
	memcpy(&v, *p, sizeof(float));
	*p += sizeof(float);
	return v;
}

static void RE_DX12_RegisterFont(const char *fontName, int pointSize, void *font, qboolean extended)
{
	fontInfo_t         *fi   = (fontInfo_t *)font;
	char                datName[MAX_QPATH];
	void               *faceData  = NULL;
	int                 len;
	const unsigned char *p;
	int                 i;

	/**
	 * GLYPH_OLD_FORMAT = sizeof(fontInfo_t) = 20548 bytes:
	 *   256 × (7 ints + 4 floats + 4 bytes handle + 32 bytes shaderName) = 20480
	 *   + 4 bytes glyphScale + 64 bytes datName = 20548
	 */
	const int GLYPH_OLD_FORMAT = 20548;

	(void)extended; // Extended (UTF-8) font loading is not yet implemented

	if (!fi || !fontName || !fontName[0])
	{
		return;
	}

	Com_Memset(fi, 0, sizeof(fontInfo_t));

	snprintf(datName, sizeof(datName), "fonts/%s_%i.dat", fontName, pointSize);

	len = dx12.ri.FS_ReadFile(datName, &faceData);

	if (len <= 0 || !faceData)
	{
		dx12.ri.Printf(PRINT_DEVELOPER,
		               "RE_DX12_RegisterFont: font file '%s' not found\n", datName);
		return;
	}

	if (len != GLYPH_OLD_FORMAT)
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "RE_DX12_RegisterFont: '%s' has unexpected size %d (expected %d)\n",
		               datName, len, GLYPH_OLD_FORMAT);
		dx12.ri.FS_FreeFile(faceData);
		return;
	}

	p = (const unsigned char *)faceData;

	// Parse glyph data for all 256 ASCII code points
	for (i = 0; i < GLYPHS_ASCII_PER_FONT; i++)
	{
		glyphInfo_t *g = &fi->glyphs[i];

		g->height      = DX12_ReadInt32(&p);
		g->top         = DX12_ReadInt32(&p);
		g->bottom      = DX12_ReadInt32(&p);
		g->pitch       = DX12_ReadInt32(&p);
		g->xSkip       = DX12_ReadInt32(&p);
		g->imageWidth  = DX12_ReadInt32(&p);
		g->imageHeight = DX12_ReadInt32(&p);
		g->s           = DX12_ReadFloat32(&p);
		g->t           = DX12_ReadFloat32(&p);
		g->s2          = DX12_ReadFloat32(&p);
		g->t2          = DX12_ReadFloat32(&p);
		p             += 4; // skip stale handle from the original file

		memcpy(g->shaderName, p, sizeof(g->shaderName));
		p += sizeof(g->shaderName);
	}

	// Read the scale factor that follows the glyph array
	fi->glyphScale = DX12_ReadFloat32(&p);

	dx12.ri.FS_FreeFile(faceData);
	faceData = NULL;

	Q_strncpyz(fi->datName, datName, sizeof(fi->datName));

	// Register a DX12 texture for every glyph shader.  DX12_RegisterTexture
	// deduplicates by name so multiple font sizes that share atlas pages are
	// handled correctly.
	for (i = GLYPH_START; i <= GLYPH_ASCII_END; i++)
	{
		if (fi->glyphs[i].shaderName[0])
		{
			fi->glyphs[i].glyph = DX12_RegisterTexture(fi->glyphs[i].shaderName);
		}
	}

	dx12.ri.Printf(PRINT_DEVELOPER,
	               "RE_DX12_RegisterFont: loaded '%s' (scale %.3f)\n",
	               datName, fi->glyphScale);
}

static void RE_DX12_LoadWorld(const char *name)
{
	DX12_LoadWorld(name);
}

static qboolean RE_DX12_GetSkinModel(qhandle_t skinid, const char *type, char *name)
{
	(void)skinid; (void)type; (void)name; return qfalse;
}

static qhandle_t RE_DX12_GetShaderFromModel(qhandle_t modelid, int surfnum, int withlightmap)
{
	(void)modelid; (void)surfnum; (void)withlightmap; return 0;
}

static void RE_DX12_SetWorldVisData(const byte *vis)
{
	(void)vis;
}

static void RE_DX12_ClearScene(void)
{
	DX12_ClearScene();
}

static void RE_DX12_AddRefEntityToScene(const refEntity_t *re)
{
	DX12_AddEntityToScene(re);
}

static int RE_DX12_LightForPoint(vec3_t point, vec3_t ambientLight, vec3_t directedLight, vec3_t lightDir)
{
	(void)point; (void)ambientLight; (void)directedLight; (void)lightDir; return 0;
}

static void RE_DX12_AddPolyToScene(qhandle_t hShader, int numVerts, const polyVert_t *verts)
{
	DX12_AddScenePoly(hShader, numVerts, verts);
}

static void RE_DX12_AddPolysToScene(qhandle_t hShader, int numVerts, const polyVert_t *verts, int numPolys)
{
	int i;

	for (i = 0; i < numPolys; i++)
	{
		DX12_AddScenePoly(hShader, numVerts, verts + i * numVerts);
	}
}

static void RE_DX12_AddLightToScene(const vec3_t org, float radius, float intensity,
                                     float r, float g, float b, qhandle_t hShader, int flags)
{
	(void)org; (void)radius; (void)intensity;
	(void)r; (void)g; (void)b; (void)hShader; (void)flags;
}

static void RE_DX12_AddCoronaToScene(const vec3_t org, float r, float g, float b,
                                      float scale, int id, qboolean visible)
{
	(void)org; (void)r; (void)g; (void)b; (void)scale; (void)id; (void)visible;
}

static void RE_DX12_SetFog(int fogvar, int var1, int var2, float r, float g, float b, float density)
{
	(void)fogvar; (void)var1; (void)var2; (void)r; (void)g; (void)b; (void)density;
}

static int RE_DX12_MarkFragments(int numPoints, const vec3_t *points, const vec3_t projection,
                                  int maxPoints, vec3_t pointBuffer, int maxFragments,
                                  markFragment_t *fragmentBuffer)
{
	(void)numPoints; (void)points; (void)projection;
	(void)maxPoints; (void)pointBuffer; (void)maxFragments; (void)fragmentBuffer;
	return 0;
}

static void RE_DX12_ProjectDecal(qhandle_t hShader, int numPoints, vec3_t *points,
                                  vec4_t projection, vec4_t color, int lifeTime, int fadeTime)
{
	(void)hShader; (void)numPoints; (void)points;
	(void)projection; (void)color; (void)lifeTime; (void)fadeTime;
}

static void RE_DX12_ClearDecals(void)
{
}

static int RE_DX12_LerpTag(orientation_t *tag, const refEntity_t *refent,
                            const char *tagName, int startIndex)
{
	return DX12_LerpTag(tag, refent, tagName, startIndex);
}

static void RE_DX12_ModelBounds(qhandle_t model, vec3_t mins, vec3_t maxs)
{
	int idx = (int)model - 1;

	if (idx >= 0 && idx < DX12_MAX_MODELS && dx12ModelData[idx].valid)
	{
		VectorCopy(dx12ModelData[idx].mins, mins);
		VectorCopy(dx12ModelData[idx].maxs, maxs);
	}
	else
	{
		VectorClear(mins);
		VectorClear(maxs);
	}
}

static void RE_DX12_RemapShader(const char *oldShader, const char *newShader, const char *offsetTime)
{
	(void)oldShader; (void)newShader; (void)offsetTime;
}

static void RE_DX12_DrawDebugPolygon(int color, int numpoints, float *points)
{
	(void)color; (void)numpoints; (void)points;
}

static void RE_DX12_DrawDebugText(const vec3_t org, float r, float g, float b,
                                   const char *text, qboolean neverOcclude)
{
	(void)org; (void)r; (void)g; (void)b; (void)text; (void)neverOcclude;
}

static qboolean RE_DX12_GetEntityToken(char *buffer, size_t size)
{
	auto s = COM_Parse((char** ) & dx12World.entityParsePoint );
	Q_strncpyz(buffer, s, size);

	if (!dx12World.entityParsePoint || !s[0])
	{
		// Reset to allow repeated calls (e.g. after a full parse pass)
		dx12World.entityParsePoint = dx12World.entityString;
		return qfalse;
	}

	return qtrue;
}

static void RE_DX12_AddPolyBufferToScene(polyBuffer_t *pPolyBuffer)
{
	int          i;
	polyVert_t   tri[3];

	if (!pPolyBuffer || pPolyBuffer->numIndicies < 3)
	{
		return;
	}

	// Decompose indexed triangles into individual 3-vertex polygon calls.
	// DX12_AddScenePoly accepts any convex polygon; feeding 3 verts gives a
	// single triangle, which is the primitive unit of a polyBuffer's index list.
	for (i = 0; i + 2 < pPolyBuffer->numIndicies; i += 3)
	{
		unsigned int i0 = pPolyBuffer->indicies[i];
		unsigned int i1 = pPolyBuffer->indicies[i + 1];
		unsigned int i2 = pPolyBuffer->indicies[i + 2];

		if (i0 >= (unsigned)pPolyBuffer->numVerts ||
		    i1 >= (unsigned)pPolyBuffer->numVerts ||
		    i2 >= (unsigned)pPolyBuffer->numVerts)
		{
			continue;
		}

		tri[0].xyz[0]       = pPolyBuffer->xyz[i0][0];
		tri[0].xyz[1]       = pPolyBuffer->xyz[i0][1];
		tri[0].xyz[2]       = pPolyBuffer->xyz[i0][2];
		tri[0].st[0]        = pPolyBuffer->st[i0][0];
		tri[0].st[1]        = pPolyBuffer->st[i0][1];
		tri[0].modulate[0]  = pPolyBuffer->color[i0][0];
		tri[0].modulate[1]  = pPolyBuffer->color[i0][1];
		tri[0].modulate[2]  = pPolyBuffer->color[i0][2];
		tri[0].modulate[3]  = pPolyBuffer->color[i0][3];

		tri[1].xyz[0]       = pPolyBuffer->xyz[i1][0];
		tri[1].xyz[1]       = pPolyBuffer->xyz[i1][1];
		tri[1].xyz[2]       = pPolyBuffer->xyz[i1][2];
		tri[1].st[0]        = pPolyBuffer->st[i1][0];
		tri[1].st[1]        = pPolyBuffer->st[i1][1];
		tri[1].modulate[0]  = pPolyBuffer->color[i1][0];
		tri[1].modulate[1]  = pPolyBuffer->color[i1][1];
		tri[1].modulate[2]  = pPolyBuffer->color[i1][2];
		tri[1].modulate[3]  = pPolyBuffer->color[i1][3];

		tri[2].xyz[0]       = pPolyBuffer->xyz[i2][0];
		tri[2].xyz[1]       = pPolyBuffer->xyz[i2][1];
		tri[2].xyz[2]       = pPolyBuffer->xyz[i2][2];
		tri[2].st[0]        = pPolyBuffer->st[i2][0];
		tri[2].st[1]        = pPolyBuffer->st[i2][1];
		tri[2].modulate[0]  = pPolyBuffer->color[i2][0];
		tri[2].modulate[1]  = pPolyBuffer->color[i2][1];
		tri[2].modulate[2]  = pPolyBuffer->color[i2][2];
		tri[2].modulate[3]  = pPolyBuffer->color[i2][3];

		DX12_AddScenePoly(pPolyBuffer->shader, 3, tri);
	}
}

static void RE_DX12_SetGlobalFog(qboolean restore, int duration, float r, float g, float b, float depthForOpaque)
{
	(void)restore; (void)duration; (void)r; (void)g; (void)b; (void)depthForOpaque;
}

static qboolean RE_DX12_inPVS(const vec3_t p1, const vec3_t p2)
{
	return DX12_inPVS(p1, p2);
}

static void RE_DX12_purgeCache(void)
{
}

static qboolean RE_DX12_LoadDynamicShader(const char *shadername, const char *shadertext)
{
	(void)shadername; (void)shadertext; return qfalse;
}

static void RE_DX12_RenderToTexture(int textureid, int x, int y, int w, int h)
{
	(void)textureid; (void)x; (void)y; (void)w; (void)h;
}

static int RE_DX12_GetTextureId(const char *imagename)
{
	int i;

	if (!imagename || !imagename[0])
	{
		return -1;
	}

	for (i = 0; i < dx12NumShaders; i++)
	{
		if (dx12Shaders[i].valid && !Q_stricmp(dx12Shaders[i].name, imagename))
		{
			return i;
		}
	}

	return -1;
}

static void RE_DX12_Finish(void)
{
	if (dx12.initialized && dx12.commandQueue)
	{
		DX12_WaitForUpload(dx12.commandQueue);
	}
}

static void RE_DX12_TakeVideoFrame(int h, int w, byte *captureBuffer, byte *encodeBuffer, qboolean motionJpeg)
{
	(void)h; (void)w; (void)captureBuffer; (void)encodeBuffer; (void)motionJpeg;
}

static void RE_DX12_InitOpenGL(void)
{
	R_DX12_Init( );
}

static int RE_DX12_InitOpenGLSubSystem(void)
{
	DX12_InitSwapchain( );
	return qtrue;
}

// ---------------------------------------------------------------------------
// GetRefAPI – DLL entry point
// ---------------------------------------------------------------------------

extern "C"
{

#ifdef USE_RENDERER_DLOPEN
/**
 * @brief GetRefAPI
 * @param[in] apiVersion
 * @param[in] rimp
 * @return Pointer to populated refexport_t, or NULL on version mismatch
 */
Q_EXPORT refexport_t *QDECL GetRefAPI(int apiVersion, refimport_t *rimp)
#else
refexport_t *GetRefAPI(int apiVersion, refimport_t *rimp)
#endif
{
	static refexport_t re;

	dx12.ri = *rimp;

	Com_Memset(&re, 0, sizeof(re));

	if (apiVersion != REF_API_VERSION)
	{
		dx12.ri.Printf(PRINT_ALL, "Mismatched REF_API_VERSION: expected %i, got %i\n",
		               REF_API_VERSION, apiVersion);
		return NULL;
	}

	re.Shutdown = R_DX12_Shutdown;

	re.BeginRegistration = RE_DX12_BeginRegistration;
	re.RegisterModel     = RE_DX12_RegisterModel;
	re.RegisterModelAllLODs = RE_DX12_RegisterModelAllLODs;
	re.RegisterSkin      = RE_DX12_RegisterSkin;
	re.RegisterShader    = RE_DX12_RegisterShader;
	re.RegisterShaderNoMip = RE_DX12_RegisterShaderNoMip;
	re.RegisterFont      = RE_DX12_RegisterFont;
	re.LoadWorld         = RE_DX12_LoadWorld;
	re.GetSkinModel      = RE_DX12_GetSkinModel;
	re.GetShaderFromModel = RE_DX12_GetShaderFromModel;
	re.SetWorldVisData   = RE_DX12_SetWorldVisData;
	re.EndRegistration   = RE_DX12_EndRegistration;

	re.ClearScene          = RE_DX12_ClearScene;
	re.AddRefEntityToScene = RE_DX12_AddRefEntityToScene;
	re.LightForPoint       = RE_DX12_LightForPoint;
	re.AddPolyToScene      = RE_DX12_AddPolyToScene;
	re.AddPolysToScene     = RE_DX12_AddPolysToScene;
	re.AddLightToScene     = RE_DX12_AddLightToScene;
	re.AddCoronaToScene    = RE_DX12_AddCoronaToScene;
	re.SetFog              = RE_DX12_SetFog;
	re.RenderScene         = RE_DX12_RenderScene;

	re.SetColor               = RE_DX12_SetColor;
	re.DrawStretchPic         = RE_DX12_DrawStretchPic;
	re.DrawRotatedPic         = RE_DX12_DrawRotatedPic;
	re.DrawStretchPicGradient = RE_DX12_DrawStretchPicGradient;
	re.Add2dPolys             = RE_DX12_Add2dPolys;
	re.DrawStretchRaw         = RE_DX12_DrawStretchRaw;
	re.UploadCinematic        = RE_DX12_UploadCinematic;

	re.BeginFrame = RE_DX12_BeginFrame;
	re.EndFrame   = RE_DX12_EndFrame;

	re.MarkFragments = RE_DX12_MarkFragments;
	re.ProjectDecal  = RE_DX12_ProjectDecal;
	re.ClearDecals   = RE_DX12_ClearDecals;

	re.LerpTag    = RE_DX12_LerpTag;
	re.ModelBounds = RE_DX12_ModelBounds;

	re.RemapShader     = RE_DX12_RemapShader;
	re.DrawDebugPolygon = RE_DX12_DrawDebugPolygon;
	re.DrawDebugText   = RE_DX12_DrawDebugText;
	re.GetEntityToken  = RE_DX12_GetEntityToken;

	re.AddPolyBufferToScene = RE_DX12_AddPolyBufferToScene;
	re.SetGlobalFog         = RE_DX12_SetGlobalFog;
	re.inPVS                = RE_DX12_inPVS;
	re.purgeCache           = RE_DX12_purgeCache;
	re.LoadDynamicShader    = RE_DX12_LoadDynamicShader;
	re.RenderToTexture      = RE_DX12_RenderToTexture;
	re.GetTextureId         = RE_DX12_GetTextureId;
	re.Finish               = RE_DX12_Finish;
	re.TakeVideoFrame       = RE_DX12_TakeVideoFrame;
	re.InitOpenGL           = RE_DX12_InitOpenGL;
	re.InitOpenGLSubSystem  = RE_DX12_InitOpenGLSubSystem;

	R_DX12_Init( );

	return &re;
}

} // extern "C"

#endif // _WIN32
