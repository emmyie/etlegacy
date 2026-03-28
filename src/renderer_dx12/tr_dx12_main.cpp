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
#include "dx12_skeletal.h"  // DX12_LoadMDS, DX12_LoadMDX, DX12_LoadMDM

#ifdef _WIN32

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

#include <string.h>  // memcpy
#include "tr_common.h"

dx12Globals_t dx12;

// Global refimport_t required by shared renderercommon files (tr_image_jpg.c, etc.)
refimport_t ri;

/* Minimal image buffer helpers (copied from the GL renderer). */
#define R_IMAGE_BUFFER_SIZE (512 * 512 * 4)

int  imageBufferSize[BUFFER_MAX_TYPES] = { 0, 0, 0 };
void *imageBufferPtr[BUFFER_MAX_TYPES] = { NULL, NULL, NULL };

extern "C" void *R_GetImageBuffer(int size, bufferMemType_t bufferType, const char *filename)
{
	if (imageBufferSize[bufferType] < R_IMAGE_BUFFER_SIZE)
	{
		imageBufferSize[bufferType] = R_IMAGE_BUFFER_SIZE;
		imageBufferPtr[bufferType]  = Com_Allocate(imageBufferSize[bufferType]);
	}
	if (size > imageBufferSize[bufferType])
	{
		if (imageBufferPtr[bufferType])
		{
			Com_Dealloc(imageBufferPtr[bufferType]);
		}

		imageBufferSize[bufferType] = size;
		imageBufferPtr[bufferType]  = Com_Allocate(imageBufferSize[bufferType]);
	}

	if (!imageBufferPtr[bufferType])
	{
		Ren_Drop("R_GetImageBuffer: unable to allocate buffer for image %s with size: %i\n", filename, size);
	}

	return imageBufferPtr[bufferType];
}

extern "C" void R_FreeImageBuffer(void)
{
	int bufferType;

	for (bufferType = 0; bufferType < BUFFER_MAX_TYPES; bufferType++)
	{
		if (!imageBufferPtr[bufferType])
		{
			continue;
		}
		Com_Dealloc(imageBufferPtr[bufferType]);

		imageBufferSize[bufferType] = 0;
		imageBufferPtr[bufferType]  = NULL;
	}
}

// ---------------------------------------------------------------------------
// Dynamic shader list  (populated by RE_DX12_LoadDynamicShader)
// ---------------------------------------------------------------------------

/**
 * @struct dx12DynShader_t
 * @brief Linked-list node for a runtime-loaded shader text block.
 *        Mirrors GL's dynamicshader_t.
 */
typedef struct dx12DynShader_s
{
	char name[MAX_QPATH];                   ///< Shader name (cache key)
	char *shadertext;                       ///< Heap-allocated shader text
	struct dx12DynShader_s *next;
} dx12DynShader_t;

/** Head of the dynamic shader list.  NULL when the list is empty. */
static dx12DynShader_t *s_dynShaderHead = NULL;

/** Purge all entries from the dynamic shader list and free their memory. */
static void DX12_PurgeDynamicShaders(void)
{
	dx12DynShader_t *cur = s_dynShaderHead;

	while (cur)
	{
		dx12DynShader_t *next = cur->next;
		dx12.ri.Free(cur->shadertext);
		dx12.ri.Free(cur);
		cur = next;
	}
	s_dynShaderHead = NULL;
}

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
	ID3D12Resource *resource;               ///< GPU texture resource (or NULL)
	D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;  ///< SRV CPU handle in the SRV heap
	D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle;  ///< SRV GPU handle for binding
	int width;                              ///< Current texture width in pixels
	int height;                             ///< Current texture height in pixels
	qboolean valid;                         ///< qtrue once successfully created
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
	if (!dx12.initialized)
	{
		// First start or after a hard shutdown (vid_restart / quit): create the
		// D3D12 device, descriptor heaps, PSO, swap chain and 2D VB from scratch.
		// R_DX12_Init also calls DX12_InitTextures() and DX12_SceneInit().
		R_DX12_Init();
	}
	else
	{
		// Soft-reset path: R_DX12_Shutdown(qfalse) was called by CL_ShutdownAll,
		// which released all per-map content (textures, models, world geometry,
		// 3D scene pipeline) but kept the D3D12 device and infrastructure alive.
		// re.purgeCache() (RE_DX12_purgeCache) was also called by CL_ShutdownAll
		// and has already reinitialised the white/black fallback textures and
		// zeroed all CPU-side registries.
		//
		// All we need to do here is recreate the 3D scene pipeline (PSO, root
		// signature, constant buffer) which was released by DX12_SceneShutdown().
		// DX12_InitTextures() does NOT need to be called again – the fallback
		// textures are already in place from the purgeCache call above.
		//
		// Guard against double-init: R_DX12_Init() already calls DX12_SceneInit()
		// at startup.  If BeginRegistration is called immediately after Init
		// (first map load), the scene is already initialized and we must not call
		// DX12_SceneInit() again or the existing resources will be leaked by the
		// Com_Memset at the top of that function.
		if (!dx12Scene.initialized)
		{
			DX12_SceneInit();
		}
	}

	r_mode         = dx12.ri.Cvar_Get("r_mode", "4", 0);
	r_customwidth  = dx12.ri.Cvar_Get("r_customwidth", "1280", 0);
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

	config->colorBits     = 32;
	config->depthBits     = 24;
	config->stencilBits   = 8;
	config->isFullscreen  = qfalse;
	config->windowAspect  = (float)config->vidWidth / (float)config->vidHeight;
	config->displayAspect = config->windowAspect;

	dx12.vidWidth  = config->vidWidth;
	dx12.vidHeight = config->vidHeight;
}

/**
 * @brief RE_DX12_EndRegistration
 *
 * Mirrors GL_EndRegistration semantics.  The GL implementation flushes
 * pending render commands (R_IssuePendingRenderCommands) and optionally
 * calls RB_ShowImages.  Neither concept applies to the DX12 renderer:
 *
 *  - DX12 does not use a registration-sequence counter, so there are no
 *    "stale" skins, models, or materials to invalidate.
 *  - DX12 has no deferred render-command queue to flush.
 *  - World geometry, scene data, and GPU buffers must not be touched here
 *    (per the function contract).
 *
 * Therefore this function is intentionally a no-op.  The hook is retained
 * so that future DX12 resource-lifetime tracking can be wired in here
 * without changing the call-site.
 */
static void RE_DX12_EndRegistration(void)
{
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
	int              srvSlot;
	dx12Texture_t    tex;

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
	float            nx1, ny1, nx2, ny2;
	float            r, g, b, a;
	dx12QuadVertex_t corners[4];

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
	corners[0].pos[0]   = nx1; corners[0].pos[1] = ny1;
	corners[0].uv[0]    = 0.0f; corners[0].uv[1] = 0.0f;
	corners[0].color[0] = r; corners[0].color[1] = g;
	corners[0].color[2] = b; corners[0].color[3] = a;

	// TR (top-right)
	corners[1].pos[0]   = nx2; corners[1].pos[1] = ny1;
	corners[1].uv[0]    = 1.0f; corners[1].uv[1] = 0.0f;
	corners[1].color[0] = r; corners[1].color[1] = g;
	corners[1].color[2] = b; corners[1].color[3] = a;

	// BL (bottom-left)
	corners[2].pos[0]   = nx1; corners[2].pos[1] = ny2;
	corners[2].uv[0]    = 0.0f; corners[2].uv[1] = 1.0f;
	corners[2].color[0] = r; corners[2].color[1] = g;
	corners[2].color[2] = b; corners[2].color[3] = a;

	// BR (bottom-right)
	corners[3].pos[0]   = nx2; corners[3].pos[1] = ny2;
	corners[3].uv[0]    = 1.0f; corners[3].uv[1] = 1.0f;
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
// Skin registry – mirrors GL's RE_RegisterSkin / RE_GetSkinModel
// ---------------------------------------------------------------------------

/** Maximum surfaces per skin (matches the GL renderer's MAX_SKIN_SURFACES). */
#define DX12_MAX_SKIN_SURFACES  256
/** Maximum model-part entries per skin (matches GL's MAX_PART_MODELS). */
#define DX12_MAX_PART_MODELS    7
/** Maximum number of concurrently loaded skins. */
#define DX12_MAX_SKINS          1024

/**
 * @struct dx12SkinSurface_t
 * @brief One surface-to-material mapping parsed from a .skin file.
 */
typedef struct
{
	char name[MAX_QPATH];       ///< Surface name (lower-cased)
	int hash;                   ///< Pre-computed hash of name
	qhandle_t matHandle;        ///< DX12 material / texture handle
} dx12SkinSurface_t;

/**
 * @struct dx12SkinModel_t
 * @brief One "md3_XXX,model.md3" entry parsed from a .skin file.
 */
typedef struct
{
	char type[MAX_QPATH];   ///< e.g. "md3_lower"
	char model[MAX_QPATH];  ///< e.g. "models/players/soldier/lower.md3"
	int hash;
} dx12SkinModel_t;

/**
 * @struct dx12Skin_t
 * @brief A parsed skin file stored in the DX12 skin table.
 */
typedef struct
{
	char name[MAX_QPATH];
	int numSurfaces;
	int numModels;
	dx12SkinSurface_t surfaces[DX12_MAX_SKIN_SURFACES];
	dx12SkinModel_t models[DX12_MAX_PART_MODELS];
} dx12Skin_t;

static dx12Skin_t dx12Skins[DX12_MAX_SKINS];
static int        dx12NumSkins = 0;

/**
 * @brief Compute a name-based hash (same algorithm as Com_HashKey in q_shared.c).
 *
 * @param[in] string  Input string (NUL-terminated).
 * @param[in] maxlen  Maximum number of characters to hash.
 * @return            Hash value.
 */
static int DX12_HashKey(char *string, int maxlen)
{
	int hash = 0;
	int i;

	for (i = 0; i < maxlen && string[i] != '\0'; i++)
	{
		hash += string[i] * (119 + i);
	}
	hash = (hash ^ (hash >> 10) ^ (hash >> 20));
	return hash;
}

/**
 * @brief Skin-file token parser: reads tokens separated by commas (or EOL).
 *
 * Mirrors the CommaParse() static function in the GL renderer's tr_image.c.
 * The caller advances @p data_p through the text buffer; returns a pointer
 * to a static token buffer that is overwritten on each call.
 *
 * @param[in,out] data_p  Pointer-to-pointer into the text buffer.
 * @return Pointer to a NUL-terminated token, or "" on end-of-input.
 */
static char *DX12_CommaParse(char **data_p)
{
	int         c     = 0;
	int         len   = 0;
	char        *data = *data_p;
	static char com_token[MAX_TOKEN_CHARS];

	com_token[0] = '\0';

	if (!data)
	{
		*data_p = NULL;
		return com_token;
	}

	// Skip whitespace and comments
	while (1)
	{
		// skip whitespace
		while ((c = *data) <= ' ')
		{
			if (!c)
			{
				break;
			}
			data++;
		}
		c = *data;

		if (c == '/' && data[1] == '/')
		{
			while (*data && *data != '\n')
			{
				data++;
			}
		}
		else if (c == '/' && data[1] == '*')
		{
			while (*data && (*data != '*' || data[1] != '/'))
			{
				data++;
			}
			if (*data)
			{
				data += 2;
			}
		}
		else
		{
			break;
		}
	}

	if (c == 0)
	{
		return (char *)"";
	}

	// Quoted string
	if (c == '\"')
	{
		data++;
		while (1)
		{
			c = *data++;
			if (c == '\"' || !c)
			{
				com_token[len] = '\0';
				*data_p        = data;
				return com_token;
			}
			if (len < MAX_TOKEN_CHARS - 1)
			{
				com_token[len++] = (char)c;
			}
		}
	}

	// Regular word – stop at whitespace or comma
	do
	{
		if (len < MAX_TOKEN_CHARS - 1)
		{
			com_token[len++] = (char)c;
		}
		data++;
		c = *data;
	}
	while (c > 32 && c != ',');

	if (len == MAX_TOKEN_CHARS)
	{
		len = 0;
	}
	com_token[len] = '\0';

	*data_p = data;
	return com_token;
}

// ---------------------------------------------------------------------------
// Minimal model registry
// ---------------------------------------------------------------------------

/** Maximum number of simultaneously registered models (matches classic renderer). */
#define DX12_MAX_MOD_KNOWN 2048

static char dx12ModelNames[DX12_MAX_MOD_KNOWN][MAX_QPATH];
static int  dx12NumModels = 0;

// ---------------------------------------------------------------------------
// DX12_ClearPerSessionState
// ---------------------------------------------------------------------------

/**
 * @brief DX12_ClearPerSessionState
 *
 * Resets all CPU-side per-session registries: the model-name lookup table
 * and the skin table.  GPU model resources must have been released already
 * by DX12_ShutdownModels() before this is called.
 *
 * Called by R_DX12_Shutdown(qfalse) (soft reset, between map loads) so that
 * the next map load starts with completely clean registries.  Also called by
 * RE_DX12_purgeCache() for consistency.
 */
void DX12_ClearPerSessionState(void)
{
	Com_Memset(dx12ModelNames, 0, sizeof(dx12ModelNames));
	dx12NumModels = 0;

	Com_Memset(dx12Skins, 0, sizeof(dx12Skins));
	dx12NumSkins = 0;
}

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

	// Zero the model data entry before the loader populates it.  DX12_LoadMD3
	// calls Com_Memset itself, but the MDS/MDX/MDM paths only set specific
	// fields – leaving stale data (lodSlots, surfaces[*] indices, numSurfaces)
	// from a previous map's model at the same slot.
	Com_Memset(&dx12ModelData[slot], 0, sizeof(dx12ModelData[slot]));

	// Choose loader based on file extension.
	{
		const char *ext = strrchr(name, '.');
		if (ext)
		{
			if (!Q_stricmp(ext, ".mds"))
			{
				// MDS: mesh + embedded animation (player bodies).
				// Store raw data for bone-based tag lookup.
				void *rawData = NULL;
				int  rawSize  = 0;
				if (DX12_LoadMDS(name, &rawData, &rawSize))
				{
					dx12ModelData[slot].rawData     = rawData;
					dx12ModelData[slot].rawDataSize = rawSize;
					dx12ModelData[slot].modelType   = DX12_MOD_MDS;
					dx12ModelData[slot].valid       = qtrue;

					// Use frame-0 bounds from the MDS header
					mdsHeader_t *mds    = (mdsHeader_t *)rawData;
					mdsFrame_t  *frame0 = (mdsFrame_t *)((byte *)mds + mds->ofsFrames);
					VectorCopy(frame0->bounds[0], dx12ModelData[slot].mins);
					VectorCopy(frame0->bounds[1], dx12ModelData[slot].maxs);
				}
			}
			else if (!Q_stricmp(ext, ".mdx"))
			{
				// MDX: pure animation companion for MDM models.
				void *rawData = NULL;
				int  rawSize  = 0;
				if (DX12_LoadMDX(name, &rawData, &rawSize))
				{
					dx12ModelData[slot].rawData     = rawData;
					dx12ModelData[slot].rawDataSize = rawSize;
					dx12ModelData[slot].modelType   = DX12_MOD_MDX;
					dx12ModelData[slot].valid       = qtrue;
				}
			}
			else if (!Q_stricmp(ext, ".mdm"))
			{
				// MDM: skeletal mesh without embedded animation.
				void *rawData = NULL;
				int  rawSize  = 0;
				if (DX12_LoadMDM(name, &rawData, &rawSize))
				{
					dx12ModelData[slot].rawData     = rawData;
					dx12ModelData[slot].rawDataSize = rawSize;
					dx12ModelData[slot].modelType   = DX12_MOD_MDM;
					dx12ModelData[slot].valid       = qtrue;
				}
			}
			else
			{
				// Attempt MD3 (includes .md3 and any unrecognised extension).
				DX12_LoadMD3(slot, name);
			}
		}
		else
		{
			DX12_LoadMD3(slot, name);
		}
	}

	return (qhandle_t)(slot + 1);
}

/**
 * @brief RE_DX12_RegisterModelAllLODs
 *
 * Loads the base model and all available LOD variants (_1.md3 .. _3.md3),
 * mirroring the GL renderer's RE_RegisterModel LOD-loading loop.  Each LOD
 * variant is registered as an independent model slot so its GPU buffers are
 * uploaded independently; the per-LOD slot indices are then stored in the
 * base model's dx12ModelEntry_t.lodSlots[] array for draw-time LOD selection.
 *
 * Missing LOD slots are filled by duplicating from the nearest available
 * level, matching the GL renderer's "duplicate into higher lod spots" fixup.
 *
 * Skeletal formats (.mds / .mdx / .mdm) have no per-LOD variants; this
 * function delegates to RE_DX12_RegisterModel for those extensions.
 *
 * @param name  Game-path of the base model file.
 * @return      Handle for the base model slot, or 0 on failure.
 */
static qhandle_t RE_DX12_RegisterModelAllLODs(const char *name)
{
	int       lod;
	int       baseSlot;
	int       numDistinct;
	qhandle_t lodHandle;
	char      lodName[MAX_QPATH];
	char      suffix[32];
	char      *dot;
	int       workSlots[MD3_MAX_LODS];

	if (!name || !name[0])
	{
		return 0;
	}

	// Skeletal formats (.mds / .mdx / .mdm) carry their own multi-frame data
	// and have no per-LOD file variants – fall back to the single-model path.
	{
		const char *ext = strrchr(name, '.');

		if (ext && (!Q_stricmp(ext, ".mds") || !Q_stricmp(ext, ".mdx") || !Q_stricmp(ext, ".mdm")))
		{
			return RE_DX12_RegisterModel(name);
		}
	}

	// Initialise all working LOD slots to "not found".
	for (lod = 0; lod < MD3_MAX_LODS; lod++)
	{
		workSlots[lod] = -1;
	}

	// Iterate from the lowest-quality LOD (highest index) down to LOD 0,
	// matching the GL RE_RegisterModel loop direction.
	// LOD 0 = base file (name unchanged); LOD N > 0 = name_N.md3 variant.
	for (lod = MD3_MAX_LODS - 1; lod >= 0; lod--)
	{
		Q_strncpyz(lodName, name, sizeof(lodName));

		if (lod != 0)
		{
			// Strip extension then append _N.md3 suffix.
			dot = strrchr(lodName, '.');
			if (dot)
			{
				*dot = '\0';
			}
			Com_sprintf(suffix, sizeof(suffix), "_%d.md3", lod);
			Q_strcat(lodName, sizeof(lodName), suffix);
		}

		lodHandle = RE_DX12_RegisterModel(lodName);
		if (lodHandle)
		{
			workSlots[lod] = (int)(lodHandle - 1);
		}
	}

	// LOD 0 (the base model) is mandatory.
	if (workSlots[0] < 0)
	{
		return 0;
	}

	baseSlot = workSlots[0];

	// Fill missing higher-LOD slots from the next better-quality (lower-index)
	// one, then fill any remaining gaps from the next lower-quality level.
	// This mirrors GL's "duplicate into higher lod spots that weren't loaded".
	for (lod = 1; lod < MD3_MAX_LODS; lod++)
	{
		if (workSlots[lod] < 0)
		{
			workSlots[lod] = workSlots[lod - 1];
		}
	}
	for (lod = MD3_MAX_LODS - 2; lod >= 0; lod--)
	{
		if (workSlots[lod] < 0)
		{
			workSlots[lod] = workSlots[lod + 1];
		}
	}

	// Count how many slots are genuinely distinct (different GPU data).
	numDistinct = 1;
	for (lod = 1; lod < MD3_MAX_LODS; lod++)
	{
		if (workSlots[lod] != workSlots[lod - 1])
		{
			numDistinct++;
		}
	}

	// Store the LOD map and count in the base slot for draw-time use.
	Com_Memcpy(dx12ModelData[baseSlot].lodSlots, workSlots, MD3_MAX_LODS * sizeof(int));
	dx12ModelData[baseSlot].numLods = numDistinct;

	return (qhandle_t)(baseSlot + 1);
}

static qhandle_t RE_DX12_RegisterSkin(const char *name)
{
	int        i;
	int        slot;
	dx12Skin_t *skin;
	void       *text_v = NULL;
	char       *text_p;
	char       *token;
	char       surfName[MAX_QPATH];
	int        totalSurfaces = 0;

	if (!name || !name[0])
	{
		dx12.ri.Printf(PRINT_DEVELOPER, "RE_DX12_RegisterSkin: empty name\n");
		return 0;
	}

	if (strlen(name) >= MAX_QPATH)
	{
		dx12.ri.Printf(PRINT_DEVELOPER, "RE_DX12_RegisterSkin: name exceeds MAX_QPATH\n");
		return 0;
	}

	// Deduplicate: return existing handle (1-based)
	for (i = 0; i < dx12NumSkins; i++)
	{
		if (!DX12_Stricmp(dx12Skins[i].name, name))
		{
			return (dx12Skins[i].numSurfaces > 0) ? (qhandle_t)(i + 1) : 0;
		}
	}

	// Allocate a new slot
	if (dx12NumSkins >= DX12_MAX_SKINS)
	{
		dx12.ri.Printf(PRINT_WARNING, "RE_DX12_RegisterSkin: skin table full, dropping '%s'\n", name);
		return 0;
	}

	slot = dx12NumSkins;
	skin = &dx12Skins[slot];
	Com_Memset(skin, 0, sizeof(*skin));
	Q_strncpyz(skin->name, name, sizeof(skin->name));
	dx12NumSkins++;

	// Load the .skin file
	dx12.ri.FS_ReadFile(name, &text_v);
	if (!text_v)
	{
		dx12.ri.Printf(PRINT_DEVELOPER, "RE_DX12_RegisterSkin: '%s' not found\n", name);
		return 0;
	}

	text_p = (char *)text_v;

	while (text_p && *text_p)
	{
		// Get the surface name (left side of comma)
		token = DX12_CommaParse(&text_p);
		Q_strncpyz(surfName, token, sizeof(surfName));

		if (!token[0])
		{
			break;
		}

		// Advance past comma
		if (*text_p == ',')
		{
			text_p++;
		}

		// Skip tag entries
		if (strstr(surfName, "tag_"))
		{
			continue;
		}

		// Model-part entry: "md3_lower,models/…/lower.md3"
		if (strstr(surfName, "md3_") || strstr(surfName, "mdc_"))
		{
			if (skin->numModels < DX12_MAX_PART_MODELS)
			{
				dx12SkinModel_t *mdl = &skin->models[skin->numModels];
				Q_strncpyz(mdl->type, surfName, sizeof(mdl->type));
				mdl->hash = DX12_HashKey(mdl->type, sizeof(mdl->type));

				token = DX12_CommaParse(&text_p);
				Q_strncpyz(mdl->model, token, sizeof(mdl->model));
				skin->numModels++;
			}
			continue;
		}

		// Surface-to-shader mapping: "surfacename,shadername"
		token = DX12_CommaParse(&text_p);

		if (totalSurfaces < DX12_MAX_SKIN_SURFACES && skin->numSurfaces < DX12_MAX_SKIN_SURFACES)
		{
			dx12SkinSurface_t *ss = &skin->surfaces[skin->numSurfaces];
			Q_strncpyz(ss->name, surfName, sizeof(ss->name));
			Q_strlwr(ss->name);
			ss->hash      = DX12_HashKey(ss->name, sizeof(ss->name));
			ss->matHandle = DX12_RegisterMaterial(token);
			skin->numSurfaces++;
		}

		totalSurfaces++;
	}

	dx12.ri.FS_FreeFile(text_v);

	if (totalSurfaces > DX12_MAX_SKIN_SURFACES)
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "RE_DX12_RegisterSkin: '%s' has %d surfaces, max is %d\n",
		               name, totalSurfaces, DX12_MAX_SKIN_SURFACES);
	}

	if (skin->numSurfaces == 0)
	{
		dx12.ri.Printf(PRINT_DEVELOPER, "RE_DX12_RegisterSkin: '%s' has no usable surfaces\n", name);
		return 0;
	}

	return (qhandle_t)(slot + 1);
}

static qhandle_t RE_DX12_RegisterShader(const char *name)
{
	const char *resolvedName;
	qhandle_t  h;

	if (!name || !name[0])
	{
		return 0;
	}

	// Apply remap table (set by RE_DX12_RemapShader / BSP loading).
	resolvedName = DX12_GetRemappedShader(name);
	if (!resolvedName)
	{
		resolvedName = name;
	}

	h = DX12_RegisterMaterial(resolvedName);
	if (!h)
	{
		// Fallback to "noshader" sentinel, mirroring the GL renderer.
		h = DX12_RegisterMaterial("noshader");
	}
	return h;
}

static qhandle_t RE_DX12_RegisterShaderNoMip(const char *name)
{
	const char     *resolvedName;
	qhandle_t      h;
	dx12Material_t *mat;

	if (!name || !name[0])
	{
		return 0;
	}

	// Apply remap table.
	resolvedName = DX12_GetRemappedShader(name);
	if (!resolvedName)
	{
		resolvedName = name;
	}

	h = DX12_RegisterMaterial(resolvedName);
	if (!h)
	{
		h = DX12_RegisterMaterial("noshader");
	}

	// Mark the material so the upload path can skip mipmap generation.
	mat = DX12_GetMaterial(h);
	if (mat)
	{
		mat->noMip = qtrue;
	}
	return h;
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
	fontInfo_t          *fi = (fontInfo_t *)font;
	char                datName[MAX_QPATH];
	void                *faceData = NULL;
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
	int        i;
	int        hash;
	dx12Skin_t *skin;

	if (skinid < 1 || skinid > dx12NumSkins)
	{
		return qfalse;
	}

	skin = &dx12Skins[skinid - 1];
	hash = DX12_HashKey((char *)type, (int)strlen(type));

	for (i = 0; i < skin->numModels; i++)
	{
		if (hash != skin->models[i].hash)
		{
			continue;
		}
		if (!Q_stricmp(skin->models[i].type, type))
		{
			Q_strncpyz(name, skin->models[i].model, sizeof(skin->models[i].model));
			return qtrue;
		}
	}

	return qfalse;
}

static qhandle_t RE_DX12_GetShaderFromModel(qhandle_t modelid, int surfnum, int withlightmap)
{
	int              slot;
	dx12ModelEntry_t *mdl;

	(void)withlightmap; // DX12 renderer does not distinguish lightmapped variants

	if (modelid < 1 || modelid > dx12NumModels)
	{
		return 0;
	}

	slot = (int)modelid - 1;
	mdl  = &dx12ModelData[slot];

	if (!mdl->valid || mdl->numSurfaces <= 0)
	{
		return 0;
	}

	if (surfnum < 0)
	{
		surfnum = 0;
	}

	if (surfnum >= mdl->numSurfaces)
	{
		surfnum = 0;
	}

	return mdl->surfaces[surfnum].texHandle;
}

static void RE_DX12_SetWorldVisData(const byte *vis)
{
	// Store the external PVS pointer.  DX12_ClusterPVS in dx12_world.cpp
	// will prefer this over the BSP-loaded vis data when non-NULL, matching
	// the GL renderer's tr.externalVisData / s_worldData.vis semantics.
	dx12World.externalVisData = vis;
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
	vec3_t lightOrigin;
	int    pos[3];
	int    i, j;
	byte   *gridData;
	float  frac[3];
	int    gridStep[3];
	vec3_t direction;
	float  totalFactor;
	float  factor;
	byte   *data;
	float  v;
	cvar_t *r_ambientScale;
	cvar_t *r_directedScale;

	if (!dx12World.lightGridData || !dx12World.loaded)
	{
		VectorSet(ambientLight, 0.0f, 0.0f, 0.0f);
		VectorSet(directedLight, 0.0f, 0.0f, 0.0f);
		VectorSet(lightDir, 0.0f, 0.0f, 1.0f);
		return qfalse;
	}

	VectorCopy(point, lightOrigin);
	VectorSubtract(lightOrigin, dx12World.lightGridOrigin, lightOrigin);

	for (i = 0; i < 3; i++)
	{
		v       = lightOrigin[i] * dx12World.lightGridInverseSize[i];
		pos[i]  = (int)floorf(v);
		frac[i] = v - (float)pos[i];
		if (pos[i] < 0)
		{
			pos[i] = 0;
		}
		else if (pos[i] > dx12World.lightGridBounds[i] - 1)
		{
			pos[i] = dx12World.lightGridBounds[i] - 1;
		}
	}

	VectorClear(ambientLight);
	VectorClear(directedLight);
	VectorClear(direction);

	// trilerp the light value
	gridStep[0] = 8;
	gridStep[1] = 8 * dx12World.lightGridBounds[0];
	gridStep[2] = 8 * dx12World.lightGridBounds[0] * dx12World.lightGridBounds[1];
	gridData    = dx12World.lightGridData
	              + pos[0] * gridStep[0]
	              + pos[1] * gridStep[1]
	              + pos[2] * gridStep[2];

	totalFactor = 0.0f;
	for (i = 0; i < 8; i++)
	{
		float  lat_rad, lng_rad;
		vec3_t normal;
		int    lat_byte, lng_byte;

		factor = 1.0f;
		data   = gridData;
		for (j = 0; j < 3; j++)
		{
			if (i & (1 << j))
			{
				factor *= frac[j];
				data   += gridStep[j];
			}
			else
			{
				factor *= (1.0f - frac[j]);
			}
		}

		if (!(data[0] + data[1] + data[2]))
		{
			continue;   // ignore samples in walls
		}
		totalFactor += factor;

		ambientLight[0] += factor * data[0];
		ambientLight[1] += factor * data[1];
		ambientLight[2] += factor * data[2];

		directedLight[0] += factor * data[3];
		directedLight[1] += factor * data[4];
		directedLight[2] += factor * data[5];

		// Decode the light direction encoded as latitude/longitude bytes.
		// data[6] = lng byte, data[7] = lat byte (same layout as GL renderer).
		// The byte encodes a fraction of a full circle (divide by 256, not 255),
		// matching sinTable[(byte * FUNCTABLE_SIZE/256)] in the GL renderer.
		// normal[0] = cos(lat) * sin(lng)
		// normal[1] = sin(lat) * sin(lng)
		// normal[2] = cos(lng)
		lng_byte = data[6];
		lat_byte = data[7];
		lat_rad  = ((float)lat_byte * (2.0f * M_PI)) / 256.0f;
		lng_rad  = ((float)lng_byte * (2.0f * M_PI)) / 256.0f;

		normal[0] = cosf(lat_rad) * sinf(lng_rad);
		normal[1] = sinf(lat_rad) * sinf(lng_rad);
		normal[2] = cosf(lng_rad);

		VectorMA(direction, factor, normal, direction);
	}

	if (totalFactor > 0.0f && totalFactor < 0.99f)
	{
		totalFactor = 1.0f / totalFactor;
		VectorScale(ambientLight, totalFactor, ambientLight);
		VectorScale(directedLight, totalFactor, directedLight);
	}

	r_ambientScale  = dx12.ri.Cvar_Get("r_ambientScale", "0.5", CVAR_CHEAT);
	r_directedScale = dx12.ri.Cvar_Get("r_directedScale", "1", CVAR_CHEAT);

	if (r_ambientScale)
	{
		VectorScale(ambientLight, r_ambientScale->value, ambientLight);
	}
	if (r_directedScale)
	{
		VectorScale(directedLight, r_directedScale->value, directedLight);
	}

	VectorNormalize2(direction, lightDir);

	return qtrue;
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
	dx12DLight_t *dl;
	cvar_t       *r_dynamicLight;

	(void)hShader; // No material lookup for dlights in the DX12 renderer yet

	// Mirror GL's tr.registered guard – skip if the renderer is not initialised
	if (!dx12.initialized)
	{
		return;
	}

	if (radius <= 0.0f || intensity <= 0.0f)
	{
		return;
	}

	// Honor r_dynamiclight cvar: non-forced dlights are dropped when it is 0,
	// exactly as GL's RE_AddLightToScene does.
	if (!(flags & REF_FORCE_DLIGHT))
	{
		r_dynamicLight = dx12.ri.Cvar_Get("r_dynamiclight", "1", CVAR_ARCHIVE);
		if (r_dynamicLight && r_dynamicLight->integer == 0)
		{
			return;
		}
	}

	if (dx12Scene.numDLights >= DX12_MAX_DLIGHTS)
	{
		dx12.ri.Printf(PRINT_DEVELOPER,
		               "RE_DX12_AddLightToScene: dropping dlight, reached MAX_DLIGHTS (%d)\n",
		               DX12_MAX_DLIGHTS);
		return;
	}

	dl = &dx12Scene.dlights[dx12Scene.numDLights++];
	VectorCopy(org, dl->origin);
	VectorCopy(org, dl->transformed);
	dl->radius             = radius;
	dl->radiusInverseCubed = 1.0f / radius;
	dl->radiusInverseCubed = dl->radiusInverseCubed * dl->radiusInverseCubed * dl->radiusInverseCubed;
	dl->intensity          = intensity;
	dl->color[0]           = r;
	dl->color[1]           = g;
	dl->color[2]           = b;
	dl->flags              = flags;
}

static void RE_DX12_AddCoronaToScene(const vec3_t org, float r, float g, float b,
                                     float scale, int id, qboolean visible)
{
	dx12Corona_t *cor;

	if (!visible)
	{
		return;
	}

	if (dx12Scene.numCoronas >= DX12_MAX_CORONAS)
	{
		dx12.ri.Printf(PRINT_DEVELOPER,
		               "RE_DX12_AddCoronaToScene: dropping corona, reached MAX_CORONAS (%d)\n",
		               DX12_MAX_CORONAS);
		return;
	}

	cor = &dx12Scene.coronas[dx12Scene.numCoronas++];
	VectorCopy(org, cor->origin);
	cor->color[0] = r;
	cor->color[1] = g;
	cor->color[2] = b;
	cor->scale    = scale;
	cor->id       = id;
	cor->visible  = visible;
}

// ---------------------------------------------------------------------------
// RE_DX12_SetFog – fog parameter table mirroring GL glfogsettings[]
// ---------------------------------------------------------------------------

// GL fog-mode / hint constants used only for the glfog_t.mode / .hint fields.
// Defined here to avoid pulling in OpenGL headers in the DX12 translation unit.
#ifndef GL_LINEAR
#define GL_LINEAR    0x2601
#endif
#ifndef GL_EXP
#define GL_EXP       0x0800
#endif
#ifndef GL_DONT_CARE
#define GL_DONT_CARE 0x1100
#endif

/// Per-fog-slot settings table, matching GL's glfogsettings[NUM_FOGS].
static glfog_t dx12FogSettings[NUM_FOGS];
/// Currently active fog slot (matches GL's glfogNum).
static glfogType_t dx12FogNum = FOG_NONE;

/**
 * @brief RE_DX12_SetFog
 * @details Direct port of R_SetFog (renderer/tr_main.c).
 *
 * Two modes:
 *  - fogvar != FOG_CMD_SWITCHFOG: set parameters for fog slot @c fogvar.
 *      var1/var2 = near/far (0,0 clears the slot).
 *      density >= 1  → GL_LINEAR mode (drawsky=false, clearscreen=true).
 *      density <  1  → GL_EXP   mode (drawsky=true,  clearscreen=false).
 *  - fogvar == FOG_CMD_SWITCHFOG: activate fog slot @c var1 over @c var2 ms.
 */
static void RE_DX12_SetFog(int fogvar, int var1, int var2, float r, float g, float b, float density)
{
	if (fogvar != FOG_CMD_SWITCHFOG)
	{
		if (var1 == 0 && var2 == 0)
		{
			dx12FogSettings[fogvar].registered = qfalse;
			return;
		}

		// identityLight is 1/(1<<overbrightBits).  DX12 has no overbright so use 1.0.
		dx12FogSettings[fogvar].color[0] = r;
		dx12FogSettings[fogvar].color[1] = g;
		dx12FogSettings[fogvar].color[2] = b;
		dx12FogSettings[fogvar].color[3] = 1.0f;
		dx12FogSettings[fogvar].start    = (float)var1;
		dx12FogSettings[fogvar].end      = (float)var2;

		if (density >= 1.0f)
		{
			dx12FogSettings[fogvar].mode        = GL_LINEAR;
			dx12FogSettings[fogvar].drawsky     = qfalse;
			dx12FogSettings[fogvar].clearscreen = qtrue;
			dx12FogSettings[fogvar].density     = 1.0f;
		}
		else
		{
			dx12FogSettings[fogvar].mode        = GL_EXP;
			dx12FogSettings[fogvar].drawsky     = qtrue;
			dx12FogSettings[fogvar].clearscreen = qfalse;
			dx12FogSettings[fogvar].density     = density;
		}

		dx12FogSettings[fogvar].hint       = GL_DONT_CARE;
		dx12FogSettings[fogvar].registered = qtrue;
		return;
	}

	// FOG_CMD_SWITCHFOG: var1 = target fog slot, var2 = transition ms
	if (dx12FogSettings[var1].registered != qtrue)
	{
		return;
	}

	dx12FogNum = (glfogType_t)var1;

	// Copy current → LAST, target → TARGET (for transition lerp)
	if (dx12FogSettings[FOG_CURRENT].registered)
	{
		Com_Memcpy(&dx12FogSettings[FOG_LAST], &dx12FogSettings[FOG_CURRENT], sizeof(glfog_t));
	}
	else
	{
		Com_Memcpy(&dx12FogSettings[FOG_LAST], &dx12FogSettings[FOG_MAP], sizeof(glfog_t));
	}

	Com_Memcpy(&dx12FogSettings[FOG_TARGET], &dx12FogSettings[dx12FogNum], sizeof(glfog_t));

	// Store transition window (use ri.Milliseconds as the renderer's time source)
	dx12FogSettings[FOG_TARGET].startTime  = dx12.ri.Milliseconds();
	dx12FogSettings[FOG_TARGET].finishTime = dx12.ri.Milliseconds() + var2;
}

// ---------------------------------------------------------------------------
// RE_DX12_MarkFragments helpers – ported from renderer/tr_marks.c
// ---------------------------------------------------------------------------

#define DX12_MAX_VERTS_ON_POLY  64
#define DX12_SIDE_FRONT  0
#define DX12_SIDE_BACK   1
#define DX12_SIDE_ON     2

/**
 * @brief DX12_ChopPolyBehindPlane
 * @details Port of R_ChopPolyBehindPlane from tr_marks.c.
 *   Clips the polygon defined by inPoints against the plane (normal, dist),
 *   writing surviving vertices to outPoints.
 */
static void DX12_ChopPolyBehindPlane(int numInPoints, vec3_t inPoints[DX12_MAX_VERTS_ON_POLY],
                                     int *numOutPoints, vec3_t outPoints[DX12_MAX_VERTS_ON_POLY],
                                     vec3_t normal, float dist, float epsilon)
{
	float dists[DX12_MAX_VERTS_ON_POLY + 4];
	int   sides[DX12_MAX_VERTS_ON_POLY + 4];
	int   counts[3];
	float dot;
	int   i, j;
	float *p1, *p2, *clip;
	float d;

	if (numInPoints >= DX12_MAX_VERTS_ON_POLY - 2)
	{
		*numOutPoints = 0;
		return;
	}

	counts[0] = counts[1] = counts[2] = 0;

	for (i = 0; i < numInPoints; i++)
	{
		dot      = DotProduct(inPoints[i], normal);
		dot     -= dist;
		dists[i] = dot;
		if (dot > epsilon)
		{
			sides[i] = DX12_SIDE_FRONT;
		}
		else if (dot < -epsilon)
		{
			sides[i] = DX12_SIDE_BACK;
		}
		else
		{
			sides[i] = DX12_SIDE_ON;
		}
		counts[sides[i]]++;
	}
	sides[i] = sides[0];
	dists[i] = dists[0];

	*numOutPoints = 0;

	if (!counts[0])
	{
		return;
	}
	if (!counts[1])
	{
		*numOutPoints = numInPoints;
		Com_Memcpy(outPoints, inPoints, numInPoints * sizeof(vec3_t));
		return;
	}

	for (i = 0; i < numInPoints; i++)
	{
		p1   = inPoints[i];
		clip = outPoints[*numOutPoints];

		if (sides[i] == DX12_SIDE_ON)
		{
			VectorCopy(p1, clip);
			(*numOutPoints)++;
			continue;
		}

		if (sides[i] == DX12_SIDE_FRONT)
		{
			VectorCopy(p1, clip);
			(*numOutPoints)++;
			clip = outPoints[*numOutPoints];
		}

		if (sides[i + 1] == DX12_SIDE_ON || sides[i + 1] == sides[i])
		{
			continue;
		}

		p2  = inPoints[(i + 1) % numInPoints];
		d   = dists[i] - dists[i + 1];
		dot = (d == 0.0f) ? 0.0f : dists[i] / d;

		for (j = 0; j < 3; j++)
		{
			clip[j] = p1[j] + dot * (p2[j] - p1[j]);
		}
		(*numOutPoints)++;
	}
}

/**
 * @brief DX12_AddMarkFragments
 * @details Port of R_AddMarkFragments from tr_marks.c.
 *   Clips a triangle against all numPlanes planes and, if any polygon remains,
 *   appends it to fragmentBuffer.
 */
static void DX12_AddMarkFragments(int numClipPoints, vec3_t clipPoints[2][DX12_MAX_VERTS_ON_POLY],
                                  int numPlanes, vec3_t *normals, float *dists,
                                  int maxPoints, vec3_t pointBuffer,
                                  int maxFragments, markFragment_t *fragmentBuffer,
                                  int *returnedPoints, int *returnedFragments)
{
	int            pingPong = 0, i;
	markFragment_t *mf;

	for (i = 0; i < numPlanes; i++)
	{
		DX12_ChopPolyBehindPlane(numClipPoints, clipPoints[pingPong],
		                         &numClipPoints, clipPoints[!pingPong],
		                         normals[i], dists[i], 0.5f);
		pingPong ^= 1;
		if (numClipPoints == 0)
		{
			break;
		}
	}

	if (numClipPoints == 0)
	{
		return;
	}

	if (numClipPoints + (*returnedPoints) > maxPoints)
	{
		return; // not enough space for this polygon
	}

	mf             = fragmentBuffer + (*returnedFragments);
	mf->firstPoint = (*returnedPoints);
	mf->numPoints  = numClipPoints;

	for (i = 0; i < numClipPoints; i++)
	{
		VectorCopy(clipPoints[pingPong][i], (float *)pointBuffer + 5 * (*returnedPoints + i));
	}

	(*returnedPoints) += numClipPoints;
	(*returnedFragments)++;
}

static int RE_DX12_MarkFragments(int numPoints, const vec3_t *points, const vec3_t projection,
                                 int maxPoints, vec3_t pointBuffer, int maxFragments,
                                 markFragment_t *fragmentBuffer)
{
	int      i, j, k;
	int      numPlanes;
	vec3_t   mins, maxs;
	int      returnedFragments;
	int      returnedPoints;
	vec3_t   normals[DX12_MAX_VERTS_ON_POLY + 2];
	float    dists[DX12_MAX_VERTS_ON_POLY + 2];
	vec3_t   clipPoints[2][DX12_MAX_VERTS_ON_POLY];
	vec3_t   projectionDir;
	vec3_t   v1, v2;
	float    radius;
	vec3_t   center;
	int      numberPoints = 4;  // caller always passes 4 corner points
	qboolean oldMapping   = qfalse;

	if (!dx12.initialized || !dx12World.loaded ||
	    !dx12World.cpuVerts || !dx12World.cpuIndexes || dx12World.numModels < 1)
	{
		return 0;
	}

	// Negative maxFragments signals the new-mapping (per-surface ST) path
	if (maxFragments < 0)
	{
		maxFragments = -maxFragments;
		oldMapping   = qtrue;
	}

	// Compute mark centre
	VectorClear(center);
	for (i = 0; i < numberPoints; i++)
	{
		VectorAdd(points[i], center, center);
	}
	VectorScale(center, 1.0f / (float)numberPoints, center);

	radius = VectorNormalize2(projection, projectionDir) / 2.0f;

	// AABB enclosing the projection volume (matches GL R_MarkFragments)
	ClearBounds(mins, maxs);
	for (i = 0; i < numberPoints; i++)
	{
		vec3_t temp;

		AddPointToBounds(points[i], mins, maxs);
		VectorMA(points[i], 1 * (1 + (int)oldMapping * radius * 4), projection, temp);
		AddPointToBounds(temp, mins, maxs);
		VectorMA(points[i], -20.0f * (1.0f + (float)oldMapping * (radius / 20.0f) * 4.0f), projectionDir, temp);
		AddPointToBounds(temp, mins, maxs);
	}

	// Build bounding planes: 4 side planes + near + far
	for (i = 0; i < numberPoints; i++)
	{
		VectorSubtract(points[(i + 1) % numberPoints], points[i], v1);
		VectorAdd(points[i], projection, v2);
		VectorSubtract(points[i], v2, v2);
		CrossProduct(v1, v2, normals[i]);
		VectorNormalize(normals[i]);
		dists[i] = DotProduct(normals[i], points[i]);
	}
	VectorCopy(projectionDir, normals[numberPoints]);
	dists[numberPoints] = DotProduct(normals[numberPoints], points[0]) - radius * (1 + (int)oldMapping * 10);
	VectorCopy(projectionDir, normals[numberPoints + 1]);
	VectorInverse(normals[numberPoints + 1]);
	dists[numberPoints + 1] = DotProduct(normals[numberPoints + 1], points[0]) - radius * (1 + (int)oldMapping * 10);
	numPlanes               = numberPoints + 2;

	returnedPoints    = 0;
	returnedFragments = 0;

	// Iterate world model[0] draw surfaces (all types are triangle lists in DX12)
	{
		const dx12WorldModel_t *wm = &dx12World.models[0];

		for (i = 0; i < wm->numSurfaces; i++)
		{
			const dx12DrawSurf_t *ds = &dx12World.drawSurfs[wm->firstSurface + i];

			// Skip non-triangle surfaces (sky, flares) and empty surfaces
			if (ds->isSky || ds->surfaceType == MST_FLARE || ds->numIndexes < 3)
			{
				continue;
			}

			if (!oldMapping && ds->surfaceType == MST_PLANAR)
			{
				// New-mapping path: per-surface axis-based texture coordinate
				// generation, matching GL's SF_FACE non-oldMapping branch.
				vec3_t axis[3];
				vec3_t originalPoints[4];
				vec3_t newCenter;
				vec3_t lnormals[DX12_MAX_VERTS_ON_POLY + 2];
				float  ldists[DX12_MAX_VERTS_ON_POLY + 2];
				vec3_t lmins, lmaxs;
				vec3_t surfnormal;
				float  texCoordScale;
				float  epsilon = 0.5f;
				float  dot;
				int    lnumPlanes;

				// Derive surface normal from the first triangle
				{
					const dx12WorldVertex_t *va = &dx12World.cpuVerts[dx12World.cpuIndexes[ds->firstIndex]];
					const dx12WorldVertex_t *vb = &dx12World.cpuVerts[dx12World.cpuIndexes[ds->firstIndex + 1]];
					const dx12WorldVertex_t *vc = &dx12World.cpuVerts[dx12World.cpuIndexes[ds->firstIndex + 2]];
					vec3_t                  ea, eb;

					VectorSubtract(vb->xyz, va->xyz, ea);
					VectorSubtract(vc->xyz, va->xyz, eb);
					CrossProduct(ea, eb, surfnormal);
					if (VectorNormalize(surfnormal) == 0.0f)
					{
						continue;
					}
				}

				{
					float planeDist = DotProduct(surfnormal,
					                             dx12World.cpuVerts[dx12World.cpuIndexes[ds->firstIndex]].xyz);

					dot = DotProduct(center, surfnormal) - planeDist;
					if (dot < -epsilon && DotProduct(surfnormal, projectionDir) >= 0.01f)
					{
						continue;
					}
					if (Q_fabs(dot) > radius)
					{
						continue;
					}

					// Project mark centre onto the surface plane
					VectorMA(center, -dot, surfnormal, newCenter);
				}

				// Build local texture axis (matching GL SF_FACE path)
				VectorNormalize2(surfnormal, axis[0]);
				PerpendicularVector(axis[1], axis[0]);
				RotatePointAroundVector(axis[2], axis[0], axis[1], (float)numPoints);
				CrossProduct(axis[0], axis[2], axis[1]);

				texCoordScale = 0.5f * 1.0f / radius;

				// 4 corners of the oriented decal quad
				for (j = 0; j < 3; j++)
				{
					originalPoints[0][j] = newCenter[j] - radius * axis[1][j] - radius * axis[2][j];
					originalPoints[1][j] = newCenter[j] + radius * axis[1][j] - radius * axis[2][j];
					originalPoints[2][j] = newCenter[j] + radius * axis[1][j] + radius * axis[2][j];
					originalPoints[3][j] = newCenter[j] - radius * axis[1][j] + radius * axis[2][j];
				}

				ClearBounds(lmins, lmaxs);
				for (j = 0; j < 4; j++)
				{
					AddPointToBounds(originalPoints[j], lmins, lmaxs);
					VectorSubtract(originalPoints[(j + 1) % numberPoints], originalPoints[j], v1);
					VectorSubtract(originalPoints[j], surfnormal, v2);
					VectorSubtract(originalPoints[j], v2, v2);
					CrossProduct(v1, v2, lnormals[j]);
					VectorNormalize(lnormals[j]);
					ldists[j] = DotProduct(lnormals[j], originalPoints[j]);
				}
				lnumPlanes = numberPoints;

				for (k = 0; k < ds->numIndexes; k += 3)
				{
					int oldNumPoints = returnedPoints;

					for (j = 0; j < 3; j++)
					{
						const dx12WorldVertex_t *vtx =
							&dx12World.cpuVerts[dx12World.cpuIndexes[ds->firstIndex + k + j]];
						VectorCopy(vtx->xyz, clipPoints[0][j]);
					}

					DX12_AddMarkFragments(3, clipPoints,
					                      lnumPlanes, lnormals, ldists,
					                      maxPoints, pointBuffer,
					                      maxFragments, fragmentBuffer,
					                      &returnedPoints, &returnedFragments);

					if (oldNumPoints != returnedPoints)
					{
						vec3_t delta;

						// Flag this fragment as having pre-computed ST coordinates
						fragmentBuffer[returnedFragments - 1].numPoints *= -1;

						for (j = 0; j < (returnedPoints - oldNumPoints); j++)
						{
							VectorSubtract((float *)pointBuffer + 5 * (oldNumPoints + j),
							               newCenter, delta);
							*((float *)pointBuffer + 5 * (oldNumPoints + j) + 3) =
								0.5f + DotProduct(delta, axis[1]) * texCoordScale;
							*((float *)pointBuffer + 5 * (oldNumPoints + j) + 4) =
								0.5f + DotProduct(delta, axis[2]) * texCoordScale;
						}
					}

					if (returnedFragments == maxFragments)
					{
						return returnedFragments;
					}
				}
			}
			else
			{
				// Old-mapping path, or non-planar triangle soup: clip each triangle
				// directly against the original projection planes.
				for (k = 0; k < ds->numIndexes; k += 3)
				{
					for (j = 0; j < 3; j++)
					{
						const dx12WorldVertex_t *vtx =
							&dx12World.cpuVerts[dx12World.cpuIndexes[ds->firstIndex + k + j]];
						VectorCopy(vtx->xyz, clipPoints[0][j]);
					}

					DX12_AddMarkFragments(3, clipPoints,
					                      numPlanes, normals, dists,
					                      maxPoints, pointBuffer,
					                      maxFragments, fragmentBuffer,
					                      &returnedPoints, &returnedFragments);

					if (returnedFragments == maxFragments)
					{
						return returnedFragments;
					}
				}
			}
		}
	}

	return returnedFragments;
}

static void RE_DX12_ProjectDecal(qhandle_t hShader, int numPoints, vec3_t *points,
                                 vec4_t projection, vec4_t color, int lifeTime, int fadeTime)
{
	// ---------------------------------------------------------------------------
	// Port of GL RE_ProjectDecal (renderer/tr_decals.c) adapted to use
	// RE_DX12_MarkFragments for world-geometry clipping instead of walking
	// BSP surfaces directly.
	// ---------------------------------------------------------------------------

	// Decal vertices built from the source polygon (xyz + standard ST layout)
	typedef struct
	{
		vec3_t xyz;
		float st[2];
	} dx12DeclVert_t;

	dx12DeclVert_t dv[4];
	vec4_t         texMat[2];
	vec4_t         proj;              // working copy of projection
	int            numDvPoints;
	int            now, fadeStartTime, fadeEndTime;

	// Buffers for RE_DX12_MarkFragments output
#define DX12_PROJ_MAX_POINTS    512
#define DX12_PROJ_MAX_FRAGS     128
	vec3_t         markPoints[DX12_PROJ_MAX_POINTS];
	markFragment_t markFrags[DX12_PROJ_MAX_FRAGS];
	vec3_t         scaledProj;
	int            numFrags;
	int            f, p;

	// ---- Input validation (matches GL RE_ProjectDecal) ----------------------
	if (numPoints != 1 && numPoints != 3 && numPoints != 4)
	{
		return;
	}
	if (lifeTime == 0)
	{
		return;
	}
	if (projection[3] <= 0.0f)
	{
		return;
	}

	// ---- Timing (matches GL) ------------------------------------------------
	if (lifeTime < 0 || fadeTime < 0)
	{
		lifeTime = 0;
		fadeTime = 0;
	}
	now           = dx12.ri.Milliseconds();
	fadeStartTime = now + lifeTime - fadeTime;
	fadeEndTime   = fadeStartTime + fadeTime;

	// ---- Byte colour components ---------------------------------------------
	byte colR = (byte)(color[0] * 255.0f);
	byte colG = (byte)(color[1] * 255.0f);
	byte colB = (byte)(color[2] * 255.0f);
	byte colA = (byte)(color[3] * 255.0f);

	// ---- Standard UV layout (matches GL decalVert_t dv[0..3]) --------------
	dv[0].st[0] = 0.0f; dv[0].st[1] = 0.0f;
	dv[1].st[0] = 0.0f; dv[1].st[1] = 1.0f;
	dv[2].st[0] = 1.0f; dv[2].st[1] = 1.0f;
	dv[3].st[0] = 1.0f; dv[3].st[1] = 0.0f;

	// Work on a local copy of projection so we can modify it for omnidirectional
	proj[0] = projection[0];
	proj[1] = projection[1];
	proj[2] = projection[2];
	proj[3] = projection[3];

	// ---- Omnidirectional vs directional (matches GL) ------------------------
	if (numPoints == 1)
	{
		float  radius, iDist;
		vec3_t corner;

		radius = projection[3];
		iDist  = 1.0f / (radius * 2.0f);

		// Reconfigure as a downward-facing quad (GL omnidirectional path)
		proj[0] = 0.0f; proj[1] = 0.0f; proj[2] = -1.0f; proj[3] = radius * 2.0f;

		VectorSet(dv[0].xyz, points[0][0] - radius, points[0][1] - radius, points[0][2] + radius);
		VectorSet(dv[1].xyz, points[0][0] - radius, points[0][1] + radius, points[0][2] + radius);
		VectorSet(dv[2].xyz, points[0][0] + radius, points[0][1] + radius, points[0][2] + radius);
		VectorSet(dv[3].xyz, points[0][0] + radius, points[0][1] - radius, points[0][2] + radius);
		numDvPoints = 4;

		// Build texMat for z-axis plane (xy coordinates)
		VectorCopy(dv[0].xyz, corner);
		texMat[0][0] = iDist; texMat[0][1] = 0.0f; texMat[0][2] = 0.0f;
		texMat[0][3] = -DotProduct(texMat[0], corner);
		texMat[1][0] = 0.0f;  texMat[1][1] = iDist; texMat[1][2] = 0.0f;
		texMat[1][3] = -DotProduct(texMat[1], corner);
	}
	else
	{
		int    i, j;
		float  bb, s, t, d;
		vec3_t pa, pb, pc;
		vec3_t bary, origin, xyz;
		vec3_t vecs[3], axis[3], lengths;

		VectorCopy(points[0], dv[0].xyz);
		VectorCopy(points[1], dv[1].xyz);
		VectorCopy(points[2], dv[2].xyz);
		if (numPoints >= 4)
		{
			VectorCopy(points[3], dv[3].xyz);
		}
		else
		{
			VectorCopy(points[2], dv[3].xyz); // pad degenerate quad
		}
		numDvPoints = 4;

		// ---- MakeTextureMatrix (port of GL static MakeTextureMatrix) --------
		// Project footprint triangle onto the projection plane
		d = DotProduct(dv[0].xyz, proj) - proj[3];
		VectorMA(dv[0].xyz, -d, proj, pa);
		d = DotProduct(dv[1].xyz, proj) - proj[3];
		VectorMA(dv[1].xyz, -d, proj, pb);
		d = DotProduct(dv[2].xyz, proj) - proj[3];
		VectorMA(dv[2].xyz, -d, proj, pc);

		// Barycentric basis
		bb = (dv[1].st[0] - dv[0].st[0]) * (dv[2].st[1] - dv[0].st[1])
		     - (dv[2].st[0] - dv[0].st[0]) * (dv[1].st[1] - dv[0].st[1]);
		if (Q_fabs(bb) < 0.00000001f)
		{
			return; // degenerate (matches GL "MakeTextureMatrix returns NULL")
		}

		// Texture origin (s=0, t=0)
		s         = 0.0f; t = 0.0f;
		bary[0]   = ((dv[1].st[0] - s) * (dv[2].st[1] - t) - (dv[2].st[0] - s) * (dv[1].st[1] - t)) / bb;
		bary[1]   = ((dv[2].st[0] - s) * (dv[0].st[1] - t) - (dv[0].st[0] - s) * (dv[2].st[1] - t)) / bb;
		bary[2]   = ((dv[0].st[0] - s) * (dv[1].st[1] - t) - (dv[1].st[0] - s) * (dv[0].st[1] - t)) / bb;
		origin[0] = bary[0] * pa[0] + bary[1] * pb[0] + bary[2] * pc[0];
		origin[1] = bary[0] * pa[1] + bary[1] * pb[1] + bary[2] * pc[1];
		origin[2] = bary[0] * pa[2] + bary[1] * pb[2] + bary[2] * pc[2];

		// S direction (s=1, t=0)
		s       = 1.0f; t = 0.0f;
		bary[0] = ((dv[1].st[0] - s) * (dv[2].st[1] - t) - (dv[2].st[0] - s) * (dv[1].st[1] - t)) / bb;
		bary[1] = ((dv[2].st[0] - s) * (dv[0].st[1] - t) - (dv[0].st[0] - s) * (dv[2].st[1] - t)) / bb;
		bary[2] = ((dv[0].st[0] - s) * (dv[1].st[1] - t) - (dv[1].st[0] - s) * (dv[0].st[1] - t)) / bb;
		xyz[0]  = bary[0] * pa[0] + bary[1] * pb[0] + bary[2] * pc[0];
		xyz[1]  = bary[0] * pa[1] + bary[1] * pb[1] + bary[2] * pc[1];
		xyz[2]  = bary[0] * pa[2] + bary[1] * pb[2] + bary[2] * pc[2];
		VectorSubtract(xyz, origin, vecs[0]);

		// T direction (s=0, t=1)
		s       = 0.0f; t = 1.0f;
		bary[0] = ((dv[1].st[0] - s) * (dv[2].st[1] - t) - (dv[2].st[0] - s) * (dv[1].st[1] - t)) / bb;
		bary[1] = ((dv[2].st[0] - s) * (dv[0].st[1] - t) - (dv[0].st[0] - s) * (dv[2].st[1] - t)) / bb;
		bary[2] = ((dv[0].st[0] - s) * (dv[1].st[1] - t) - (dv[1].st[0] - s) * (dv[0].st[1] - t)) / bb;
		xyz[0]  = bary[0] * pa[0] + bary[1] * pb[0] + bary[2] * pc[0];
		xyz[1]  = bary[0] * pa[1] + bary[1] * pb[1] + bary[2] * pc[1];
		xyz[2]  = bary[0] * pa[2] + bary[1] * pb[2] + bary[2] * pc[2];
		VectorSubtract(xyz, origin, vecs[1]);

		// Projection (R) direction
		VectorScale(proj, -1.0f, vecs[2]);

		// Normalise to build texMat rows
		for (i = 0; i < 3; i++)
		{
			lengths[i] = VectorNormalize2(vecs[i], axis[i]);
		}
		for (i = 0; i < 2; i++)
		{
			for (j = 0; j < 3; j++)
			{
				texMat[i][j] = lengths[i] > 0.0f ? (axis[i][j] / lengths[i]) : 0.0f;
			}
		}
		texMat[0][3] = dv[0].st[0] - DotProduct(pa, texMat[0]);
		texMat[1][3] = dv[0].st[1] - DotProduct(pa, texMat[1]);
	}

	// ---- Call MarkFragments to get clipped world geometry -------------------
	// MarkFragments expects projection as a vec3_t scaled by depth.
	// proj[0..2] * proj[3] gives the directional extent vector.
	scaledProj[0] = proj[0] * proj[3];
	scaledProj[1] = proj[1] * proj[3];
	scaledProj[2] = proj[2] * proj[3];

	{
		vec3_t dvXyz[4];

		dvXyz[0][0] = dv[0].xyz[0]; dvXyz[0][1] = dv[0].xyz[1]; dvXyz[0][2] = dv[0].xyz[2];
		dvXyz[1][0] = dv[1].xyz[0]; dvXyz[1][1] = dv[1].xyz[1]; dvXyz[1][2] = dv[1].xyz[2];
		dvXyz[2][0] = dv[2].xyz[0]; dvXyz[2][1] = dv[2].xyz[1]; dvXyz[2][2] = dv[2].xyz[2];
		dvXyz[3][0] = dv[3].xyz[0]; dvXyz[3][1] = dv[3].xyz[1]; dvXyz[3][2] = dv[3].xyz[2];

		numFrags = RE_DX12_MarkFragments(numDvPoints, (const vec3_t *)dvXyz,
		                                 scaledProj,
		                                 DX12_PROJ_MAX_POINTS, markPoints[0],
		                                 DX12_PROJ_MAX_FRAGS, markFrags);
	}

	if (numFrags == 0)
	{
		return;
	}

	// ---- For each fragment: apply texMat ST, store as persistent decal ------
	for (f = 0; f < numFrags; f++)
	{
		markFragment_t *mf = &markFrags[f];
		polyVert_t     verts[DX12_MAX_DECAL_VERTS];
		int            numV;

		if (mf->numPoints < 3)
		{
			continue;
		}

		numV = mf->numPoints < DX12_MAX_DECAL_VERTS ? mf->numPoints : DX12_MAX_DECAL_VERTS;

		for (p = 0; p < numV; p++)
		{
			float *xyz = markPoints[mf->firstPoint + p];

			verts[p].xyz[0] = xyz[0];
			verts[p].xyz[1] = xyz[1];
			verts[p].xyz[2] = xyz[2];

			// ST from texMat (matches GL ProjectDecalOntoWinding)
			verts[p].st[0] = DotProduct(xyz, texMat[0]) + texMat[0][3];
			verts[p].st[1] = DotProduct(xyz, texMat[1]) + texMat[1][3];

			// Base colour (time-based alpha applied per-frame in DX12_RenderScene)
			verts[p].modulate[0] = colR;
			verts[p].modulate[1] = colG;
			verts[p].modulate[2] = colB;
			verts[p].modulate[3] = colA;
		}

		DX12_AddDecalToScene(hShader, numV, verts, fadeStartTime, fadeEndTime);
	}

#undef DX12_PROJ_MAX_POINTS
#undef DX12_PROJ_MAX_FRAGS
}

static void RE_DX12_ClearDecals(void)
{
	DX12_ClearDecals();
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
	float timeOffset = offsetTime ? (float)atof(offsetTime) : 0.0f;

	DX12_AddShaderRemap(oldShader, newShader, timeOffset);
}

static void RE_DX12_DrawDebugPolygon(int color, int numpoints, float *points)
{
	polyVert_t verts[64];
	int        i;
	byte       r, g, b;

	if (!points || numpoints < 3 || numpoints > 64)
	{
		return;
	}

	// Decode packed 3-bit colour (bit 0=R, bit 1=G, bit 2=B) matching GL renderer.
	r = (color & 1)       ? 255 : 0;
	g = ((color >> 1) & 1) ? 255 : 0;
	b = ((color >> 2) & 1) ? 255 : 0;

	for (i = 0; i < numpoints; i++)
	{
		verts[i].xyz[0]      = points[i * 3 + 0];
		verts[i].xyz[1]      = points[i * 3 + 1];
		verts[i].xyz[2]      = points[i * 3 + 2];
		verts[i].st[0]       = 0.0f;
		verts[i].st[1]       = 0.0f;
		verts[i].modulate[0] = r;
		verts[i].modulate[1] = g;
		verts[i].modulate[2] = b;
		verts[i].modulate[3] = 255;
	}

	// Route through the poly system using the __white__ texture (handle 0).
	DX12_AddScenePoly(0, numpoints, verts);
}

static void RE_DX12_DrawDebugText(const vec3_t org, float r, float g, float b,
                                  const char *text, qboolean neverOcclude)
{
	// Mirrors the GL renderer which is also unimplemented (prints a TODO).
	(void)org; (void)r; (void)g; (void)b; (void)neverOcclude;
	dx12.ri.Printf(PRINT_DEVELOPER, "TODO: RE_DX12_DrawDebugText – text: %s\n",
	               text ? text : "(null)");
}

static qboolean RE_DX12_GetEntityToken(char *buffer, size_t size)
{
	auto s = COM_Parse((char ** ) &dx12World.entityParsePoint);
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
	int        i;
	polyVert_t tri[3];

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

		tri[0].xyz[0]      = pPolyBuffer->xyz[i0][0];
		tri[0].xyz[1]      = pPolyBuffer->xyz[i0][1];
		tri[0].xyz[2]      = pPolyBuffer->xyz[i0][2];
		tri[0].st[0]       = pPolyBuffer->st[i0][0];
		tri[0].st[1]       = pPolyBuffer->st[i0][1];
		tri[0].modulate[0] = pPolyBuffer->color[i0][0];
		tri[0].modulate[1] = pPolyBuffer->color[i0][1];
		tri[0].modulate[2] = pPolyBuffer->color[i0][2];
		tri[0].modulate[3] = pPolyBuffer->color[i0][3];

		tri[1].xyz[0]      = pPolyBuffer->xyz[i1][0];
		tri[1].xyz[1]      = pPolyBuffer->xyz[i1][1];
		tri[1].xyz[2]      = pPolyBuffer->xyz[i1][2];
		tri[1].st[0]       = pPolyBuffer->st[i1][0];
		tri[1].st[1]       = pPolyBuffer->st[i1][1];
		tri[1].modulate[0] = pPolyBuffer->color[i1][0];
		tri[1].modulate[1] = pPolyBuffer->color[i1][1];
		tri[1].modulate[2] = pPolyBuffer->color[i1][2];
		tri[1].modulate[3] = pPolyBuffer->color[i1][3];

		tri[2].xyz[0]      = pPolyBuffer->xyz[i2][0];
		tri[2].xyz[1]      = pPolyBuffer->xyz[i2][1];
		tri[2].xyz[2]      = pPolyBuffer->xyz[i2][2];
		tri[2].st[0]       = pPolyBuffer->st[i2][0];
		tri[2].st[1]       = pPolyBuffer->st[i2][1];
		tri[2].modulate[0] = pPolyBuffer->color[i2][0];
		tri[2].modulate[1] = pPolyBuffer->color[i2][1];
		tri[2].modulate[2] = pPolyBuffer->color[i2][2];
		tri[2].modulate[3] = pPolyBuffer->color[i2][3];

		DX12_AddScenePoly(pPolyBuffer->shader, 3, tri);
	}
}

static void RE_DX12_SetGlobalFog(qboolean restore, int duration, float r, float g, float b, float depthForOpaque)
{
	// Full port of GL RE_SetGlobalFog (renderer/tr_cmds.c).
	// Saves original fog on first activation; drives timed transitions via
	// dx12World.globalFogTrans* fields ticked in DX12_RenderScene.
	if (restore)
	{
		if (duration > 0)
		{
			// Transition from current back to original
			dx12World.globalFogTransStartFog[0] = dx12World.globalFogColor[0];
			dx12World.globalFogTransStartFog[1] = dx12World.globalFogColor[1];
			dx12World.globalFogTransStartFog[2] = dx12World.globalFogColor[2];
			dx12World.globalFogTransStartFog[3] = dx12World.globalFogDepth;

			dx12World.globalFogTransEndFog[0] = dx12World.globalFogOrigColor[0];
			dx12World.globalFogTransEndFog[1] = dx12World.globalFogOrigColor[1];
			dx12World.globalFogTransEndFog[2] = dx12World.globalFogOrigColor[2];
			dx12World.globalFogTransEndFog[3] = dx12World.globalFogOrigColor[3];

			dx12World.globalFogTransStartTime = dx12.ri.Milliseconds();
			dx12World.globalFogTransEndTime   = dx12World.globalFogTransStartTime + duration;
		}
		else
		{
			// Instant restore
			dx12World.globalFogColor[0]       = dx12World.globalFogOrigColor[0];
			dx12World.globalFogColor[1]       = dx12World.globalFogOrigColor[1];
			dx12World.globalFogColor[2]       = dx12World.globalFogOrigColor[2];
			dx12World.globalFogDepth          = dx12World.globalFogOrigColor[3];
			dx12World.globalFogActive         = (dx12World.globalFogDepth > 0.0f) ? qtrue : qfalse;
			dx12World.globalFogTransEndTime   = 0;
			dx12World.globalFogTransStartTime = 0;
		}
	}
	else
	{
		// First call: save original state so restore can return to it
		if (!dx12World.globalFogActive && dx12World.globalFogTransEndTime == 0)
		{
			dx12World.globalFogOrigColor[0] = dx12World.globalFogColor[0];
			dx12World.globalFogOrigColor[1] = dx12World.globalFogColor[1];
			dx12World.globalFogOrigColor[2] = dx12World.globalFogColor[2];
			dx12World.globalFogOrigColor[3] = dx12World.globalFogDepth;
		}

		if (duration > 0)
		{
			// Transition from current to new
			dx12World.globalFogTransStartFog[0] = dx12World.globalFogColor[0];
			dx12World.globalFogTransStartFog[1] = dx12World.globalFogColor[1];
			dx12World.globalFogTransStartFog[2] = dx12World.globalFogColor[2];
			dx12World.globalFogTransStartFog[3] = dx12World.globalFogDepth;

			dx12World.globalFogTransEndFog[0] = r;
			dx12World.globalFogTransEndFog[1] = g;
			dx12World.globalFogTransEndFog[2] = b;
			dx12World.globalFogTransEndFog[3] = depthForOpaque < 1.0f ? 1.0f : depthForOpaque;

			dx12World.globalFogTransStartTime = dx12.ri.Milliseconds();
			dx12World.globalFogTransEndTime   = dx12World.globalFogTransStartTime + duration;
			dx12World.globalFogActive         = qtrue;
		}
		else
		{
			// Instant set
			dx12World.globalFogColor[0]       = r;
			dx12World.globalFogColor[1]       = g;
			dx12World.globalFogColor[2]       = b;
			dx12World.globalFogDepth          = depthForOpaque < 1.0f ? 1.0f : depthForOpaque;
			dx12World.globalFogActive         = qtrue;
			dx12World.globalFogTransEndTime   = 0;
			dx12World.globalFogTransStartTime = 0;
		}
	}
}

static qboolean RE_DX12_inPVS(const vec3_t p1, const vec3_t p2)
{
	return DX12_inPVS(p1, p2);
}

static void RE_DX12_purgeCache(void)
{
	if (!dx12.initialized)
	{
		return;
	}

	// ---- 1. Wait for the GPU to finish all outstanding work ----------------
	// This is required before releasing any D3D12 GPU-backed resources
	// (textures, model vertex/index buffers) to avoid device removal.
	DX12_FlushGpu();

	// ---- 2. Shut down the 3D scene pipeline --------------------------------
	// Release the 3D PSO, root signature, constant buffer and poly VB so that
	// they can be recreated cleanly by DX12_SceneInit() on the next map load.
	DX12_SceneShutdown();

	// ---- 3. Purge dynamic shader scripts (CPU-only) ------------------------
	DX12_PurgeDynamicShaders();

	// ---- 4. Release all texture GPU resources and clear material table -----
	// DX12_ShutdownTextures resets dx12NumShaders and dx12NumMaterials to 0.
	DX12_ShutdownTextures();

	// Rebuild the two mandatory fallback entries (white / black).
	DX12_InitTextures();

	// ---- 5. Release model GPU resources and reset all CPU lookup tables ----
	DX12_ShutdownModels();
	DX12_ClearPerSessionState();
}

static qboolean RE_DX12_LoadDynamicShader(const char *shadername, const char *shadertext)
{
	dx12DynShader_t *cur, *prev, *node;
	size_t          textLen;

	// NULL name + NULL text → purge all dynamic shaders (mirrors GL RE_LoadDynamicShader)
	if (!shadername && !shadertext)
	{
		DX12_PurgeDynamicShaders();
		return qtrue;
	}

	if (shadername && strlen(shadername) >= MAX_QPATH)
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "RE_DX12_LoadDynamicShader: shadername '%s' exceeds MAX_QPATH\n",
		               shadername);
		return qfalse;
	}

	// NULL text with valid name → remove this specific dynamic shader
	if (shadername && !shadertext)
	{
		prev = NULL;
		cur  = s_dynShaderHead;
		while (cur)
		{
			if (!Q_stricmp(cur->name, shadername))
			{
				if (prev)
				{
					prev->next = cur->next;
				}
				else
				{
					s_dynShaderHead = cur->next;
				}
				dx12.ri.Free(cur->shadertext);
				dx12.ri.Free(cur);
				return qtrue;
			}
			prev = cur;
			cur  = cur->next;
		}
		return qtrue; // not found; not an error
	}

	if (!shadername || !shadertext || !shadertext[0])
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "RE_DX12_LoadDynamicShader: missing shadername or shadertext\n");
		return qfalse;
	}

	// Reject duplicate names
	for (cur = s_dynShaderHead; cur; cur = cur->next)
	{
		if (!Q_stricmp(cur->name, shadername))
		{
			dx12.ri.Printf(PRINT_WARNING,
			               "RE_DX12_LoadDynamicShader: shader '%s' already exists\n",
			               shadername);
			return qfalse;
		}
	}

	// Allocate and store the new dynamic shader node.
	node = (dx12DynShader_t *)dx12.ri.Z_Malloc(sizeof(*node));
	Q_strncpyz(node->name, shadername, sizeof(node->name));

	textLen          = strlen(shadertext);
	node->shadertext = (char *)dx12.ri.Z_Malloc((int)textLen + 1);
	Q_strncpyz(node->shadertext, shadertext, (int)textLen + 1);
	node->next      = s_dynShaderHead;
	s_dynShaderHead = node;

	// Eagerly register the material so callers can immediately use it.
	return DX12_RegisterMaterialFromText(shadername, shadertext);
}

static void RE_DX12_RenderToTexture(int textureid, int x, int y, int w, int h)
{
	if (!dx12.initialized)
	{
		return;
	}

	if (textureid < 0 || textureid >= dx12NumShaders)
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "RE_DX12_RenderToTexture: textureid %d out of range (numShaders=%d)\n",
		               textureid, dx12NumShaders);
		return;
	}

	if (!dx12Shaders[textureid].valid)
	{
		dx12.ri.Printf(PRINT_DEVELOPER,
		               "RE_DX12_RenderToTexture: textureid %d is not a valid slot\n",
		               textureid);
		return;
	}

	if (!dx12.frameOpen)
	{
		dx12.ri.Printf(PRINT_DEVELOPER,
		               "RE_DX12_RenderToTexture: called outside of a frame\n");
		return;
	}

	DX12_CopyRenderTargetToTexture(&dx12Shaders[textureid], textureid, x, y, w, h);
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

// Forward declaration – defined in renderercommon/tr_image_jpg.c, linked with this module.
extern "C" size_t RE_SaveJPGToBuffer(byte *buffer, size_t bufSize, int quality,
                                     int image_width, int image_height,
                                     byte *image_buffer, int padding);

#ifndef AVI_LINE_PADDING
#define AVI_LINE_PADDING 4
#endif

/**
 * @brief RE_DX12_TakeVideoFrame
 * @param[in] h            Frame height (first argument per tr_public.h TakeVideoFrame signature)
 * @param[in] w            Frame width  (second argument)
 * @param[in] captureBuffer Caller-supplied readback scratch buffer
 * @param[in] encodeBuffer  Caller-supplied encode output buffer
 * @param[in] motionJpeg    If qtrue, encode as JPEG; otherwise produce raw BGR AVI rows
 */
static void RE_DX12_TakeVideoFrame(int h, int w, byte *captureBuffer, byte *encodeBuffer, qboolean motionJpeg)
{
	// tr_public.h prototype: TakeVideoFrame(int h, int w, ...) – h is height, w is width.
	int    width  = w;
	int    height = h;
	byte   *cBuf  = captureBuffer;
	size_t linelen;
	int    padwidth;
	int    avipadwidth;
	int    avipadlen;
	int    padlen;

	if (!dx12.initialized || !cBuf || !encodeBuffer || width <= 0 || height <= 0)
	{
		return;
	}

	// Read back the current back-buffer into cBuf as packed RGB with 4-byte row padding
	// (matches GL_PACK_ALIGNMENT=4 used by glReadPixels in the GL path).
	if (!DX12_ReadbackRenderTarget(cBuf, width, height))
	{
		return;
	}

	linelen     = (size_t)width * 3;
	padwidth    = ((int)linelen + 3) & ~3;  // PAD(linelen, 4)
	padlen      = padwidth - (int)linelen;
	avipadwidth = ((int)linelen + AVI_LINE_PADDING - 1) & ~(AVI_LINE_PADDING - 1);
	avipadlen   = avipadwidth - (int)linelen;

	if (motionJpeg)
	{
		size_t memcount = RE_SaveJPGToBuffer(encodeBuffer, linelen * (size_t)height,
		                                     dx12.ri.Cvar_Get("r_screenshotJpegQuality", "90", 0)->integer,
		                                     width, height, cBuf, padlen);
		dx12.ri.CL_WriteAVIVideoFrame(encodeBuffer, (int)memcount);
	}
	else
	{
		// Swap R↔B (GL reads RGB; AVI expects BGR) and remove row padding.
		byte       *srcptr  = cBuf;
		byte       *destptr = encodeBuffer;
		const byte *memend  = srcptr + (size_t)padwidth * (size_t)height;

		while (srcptr < memend)
		{
			const byte *lineend = srcptr + linelen;

			while (srcptr < lineend)
			{
				*destptr++ = srcptr[2];   // B
				*destptr++ = srcptr[1];   // G
				*destptr++ = srcptr[0];   // R
				srcptr    += 3;
			}

			// AVI row padding (zero bytes)
			Com_Memset(destptr, '\0', (size_t)avipadlen);
			destptr += avipadlen;

			srcptr += padlen;   // skip RGB row padding
		}

		dx12.ri.CL_WriteAVIVideoFrame(encodeBuffer, avipadwidth * height);
	}
}

static void RE_DX12_InitOpenGL(void)
{
	R_DX12_Init();
}

static int RE_DX12_InitOpenGLSubSystem(void)
{
	DX12_InitSwapchain();
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
	ri      = *rimp;

	Com_Memset(&re, 0, sizeof(re));

	if (apiVersion != REF_API_VERSION)
	{
		dx12.ri.Printf(PRINT_ALL, "Mismatched REF_API_VERSION: expected %i, got %i\n",
		               REF_API_VERSION, apiVersion);
		return NULL;
	}

	re.Shutdown = R_DX12_Shutdown;

	re.BeginRegistration    = RE_DX12_BeginRegistration;
	re.RegisterModel        = RE_DX12_RegisterModel;
	re.RegisterModelAllLODs = RE_DX12_RegisterModelAllLODs;
	re.RegisterSkin         = RE_DX12_RegisterSkin;
	re.RegisterShader       = RE_DX12_RegisterShader;
	re.RegisterShaderNoMip  = RE_DX12_RegisterShaderNoMip;
	re.RegisterFont         = RE_DX12_RegisterFont;
	re.LoadWorld            = RE_DX12_LoadWorld;
	re.GetSkinModel         = RE_DX12_GetSkinModel;
	re.GetShaderFromModel   = RE_DX12_GetShaderFromModel;
	re.SetWorldVisData      = RE_DX12_SetWorldVisData;
	re.EndRegistration      = RE_DX12_EndRegistration;

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

	re.LerpTag     = RE_DX12_LerpTag;
	re.ModelBounds = RE_DX12_ModelBounds;

	re.RemapShader      = RE_DX12_RemapShader;
	re.DrawDebugPolygon = RE_DX12_DrawDebugPolygon;
	re.DrawDebugText    = RE_DX12_DrawDebugText;
	re.GetEntityToken   = RE_DX12_GetEntityToken;

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

	R_DX12_Init();

	return &re;
}

} // extern "C"

#endif // _WIN32
