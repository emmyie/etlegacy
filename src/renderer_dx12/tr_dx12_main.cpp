/**
 * @file tr_dx12_main.cpp
 * @brief DirectX 12 renderer – main entry point and refexport_t implementations
 */

#include "tr_dx12_local.h"
#include "dx12_shader.h"
#include "dx12_poly.h"

#ifdef _WIN32

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

#include <string.h>  // memcpy

dx12Globals_t dx12;

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
	(void)fd;
	// World rendering not implemented; triangle is drawn in SwapBuffers
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
	(void)x; (void)y; (void)w; (void)h;
	(void)s1; (void)t1; (void)s2; (void)t2;
	(void)hShader; (void)angle;
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

static void RE_DX12_DrawStretchRaw(int x, int y, int w, int h, int cols, int rows,
                                    const byte *data, int client, qboolean dirty)
{
	(void)x; (void)y; (void)w; (void)h;
	(void)cols; (void)rows; (void)data; (void)client; (void)dirty;
}

static void RE_DX12_UploadCinematic(int w, int h, int cols, int rows,
                                     const byte *data, int client, qboolean dirty)
{
	(void)w; (void)h; (void)cols; (void)rows;
	(void)data; (void)client; (void)dirty;
}

static void RE_DX12_BeginFrame(void)
{
	DX12_BeginFrameRender();
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

	R_DX12_SwapBuffers();
}

static qhandle_t RE_DX12_RegisterModel(const char *name)
{
	(void)name; return 0;
}

static qhandle_t RE_DX12_RegisterModelAllLODs(const char *name)
{
	(void)name; return 0;
}

static qhandle_t RE_DX12_RegisterSkin(const char *name)
{
	(void)name; return 0;
}

static qhandle_t RE_DX12_RegisterShader(const char *name)
{
	return DX12_RegisterTexture(name);
}

static qhandle_t RE_DX12_RegisterShaderNoMip(const char *name)
{
	return DX12_RegisterTexture(name);
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
	(void)name;
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
}

static void RE_DX12_AddRefEntityToScene(const refEntity_t *re)
{
	(void)re;
}

static int RE_DX12_LightForPoint(vec3_t point, vec3_t ambientLight, vec3_t directedLight, vec3_t lightDir)
{
	(void)point; (void)ambientLight; (void)directedLight; (void)lightDir; return 0;
}

static void RE_DX12_AddPolyToScene(qhandle_t hShader, int numVerts, const polyVert_t *verts)
{
	(void)hShader; (void)numVerts; (void)verts;
}

static void RE_DX12_AddPolysToScene(qhandle_t hShader, int numVerts, const polyVert_t *verts, int numPolys)
{
	(void)hShader; (void)numVerts; (void)verts; (void)numPolys;
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
	(void)tag; (void)refent; (void)tagName; (void)startIndex; return 0;
}

static void RE_DX12_ModelBounds(qhandle_t model, vec3_t mins, vec3_t maxs)
{
	(void)model; (void)mins; (void)maxs;
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
	(void)buffer; (void)size; return qfalse;
}

static void RE_DX12_AddPolyBufferToScene(polyBuffer_t *pPolyBuffer)
{
	(void)pPolyBuffer;
}

static void RE_DX12_SetGlobalFog(qboolean restore, int duration, float r, float g, float b, float depthForOpaque)
{
	(void)restore; (void)duration; (void)r; (void)g; (void)b; (void)depthForOpaque;
}

static qboolean RE_DX12_inPVS(const vec3_t p1, const vec3_t p2)
{
	(void)p1; (void)p2; return qfalse;
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
	(void)imagename; return -1;
}

static void RE_DX12_Finish(void)
{
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
