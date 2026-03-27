/**
 * @file tr_dx12_local.h
 * @brief Private header for the DirectX 12 renderer backend
 */

#ifndef TR_DX12_LOCAL_H
#define TR_DX12_LOCAL_H

extern "C" {
#include "q_shared.h"
}

#include "../renderercommon/tr_public.h"

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>

#define DX12_FRAME_COUNT    2
#define DX12_MAX_TEXTURES   1024  ///< Max simultaneously loaded textures
/// Per-frame 2D vertex budget.  Strip quads need 4 verts each; fan-expanded
/// polygons need (numverts-2)*3 verts.  24576 comfortably fits ~4000 UI calls.
#define DX12_MAX_2D_VERTS   24576

/**
 * @struct dx12QuadVertex_t
 * @brief A single 2D vertex: NDC position, UV, and per-vertex RGBA color
 */
typedef struct
{
	float pos[2];    ///< NDC clip-space position (X, Y)
	float uv[2];     ///< Texture coordinates (S, T)
	float color[4];  ///< Modulate color (R, G, B, A) in [0, 1]
} dx12QuadVertex_t;

/**
 * @struct dx12Texture_t
 * @brief A DX12 2D texture with its SRV descriptor handles
 */
typedef struct
{
	ID3D12Resource              *resource;
	D3D12_CPU_DESCRIPTOR_HANDLE  cpuHandle;
	D3D12_GPU_DESCRIPTOR_HANDLE  gpuHandle;
} dx12Texture_t;

// ---------------------------------------------------------------------------
// Material system types
// ---------------------------------------------------------------------------

/** Maximum tcMod entries per material stage. */
#define DX12_MAX_TCMODS          4
/** Maximum animation frames per animMap stage. */
#define DX12_MAX_ANIM_FRAMES     16
/** Maximum rendering stages per material. */
#define DX12_MAX_MATERIAL_STAGES 8
/** Maximum simultaneously cached materials. */
#define DX12_MAX_MATERIALS       1024
/**
 * Material handles occupy [DX12_MATERIAL_HANDLE_BASE, …).
 * DX12_GetTexture() transparently maps a material handle to its
 * stage-0 texture so existing draw code continues to work unchanged.
 */
#define DX12_MATERIAL_HANDLE_BASE DX12_MAX_TEXTURES

/**
 * @enum dx12TcModType_t
 * @brief Texture-coordinate modification types for material stages.
 */
typedef enum
{
	DX12_TMOD_NONE    = 0,
	DX12_TMOD_SCROLL,
	DX12_TMOD_ROTATE,
	DX12_TMOD_STRETCH,
} dx12TcModType_t;

/**
 * @enum dx12WaveFunc_t
 * @brief Waveform function used by tcMod stretch.
 */
typedef enum
{
	DX12_WAVE_SIN               = 0,
	DX12_WAVE_SQUARE,
	DX12_WAVE_TRIANGLE,
	DX12_WAVE_SAWTOOTH,
	DX12_WAVE_INVERSE_SAWTOOTH,
} dx12WaveFunc_t;

/**
 * @struct dx12Wave_t
 * @brief Parameters for a periodic waveform (used by tcMod stretch).
 */
typedef struct
{
	dx12WaveFunc_t func;       ///< Wave function type
	float          base;       ///< DC offset
	float          amplitude;  ///< Peak amplitude
	float          phase;      ///< Phase shift in [0, 1]
	float          frequency;  ///< Cycles per second
} dx12Wave_t;

/**
 * @struct dx12TcMod_t
 * @brief One texture-coordinate modification entry for a material stage.
 */
typedef struct
{
	dx12TcModType_t type;
	float           scroll[2];   ///< DX12_TMOD_SCROLL: UV scroll rates (units/s)
	float           rotateSpeed; ///< DX12_TMOD_ROTATE: degrees per second (+ = CCW)
	dx12Wave_t      stretch;     ///< DX12_TMOD_STRETCH: wave envelope
} dx12TcMod_t;

/**
 * @struct dx12MaterialStage_t
 * @brief One rendering stage within a DX12 material.
 */
typedef struct
{
	qboolean    active;                            ///< qtrue when this slot is in use
	qhandle_t   texHandle;                         ///< Primary texture (from DX12_RegisterTexture)
	D3D12_BLEND srcBlend;                          ///< Source blend factor
	D3D12_BLEND dstBlend;                          ///< Destination blend factor
	dx12TcMod_t tcMods[DX12_MAX_TCMODS];           ///< Texture-coordinate modifiers
	int         numTcMods;                         ///< Number of active tcMod entries
	qhandle_t   animFrames[DX12_MAX_ANIM_FRAMES];  ///< Per-frame texture handles (animMap)
	int         animNumFrames;                     ///< Number of animation frames
	float       animFps;                           ///< Animation playback rate (frames/s)
} dx12MaterialStage_t;

/**
 * @struct dx12Material_t
 * @brief Parsed ET .shader script entry cached by DX12_RegisterMaterial().
 *
 * The handle returned by DX12_RegisterMaterial() equals
 * DX12_MATERIAL_HANDLE_BASE + (index into dx12Materials[]).
 */
typedef struct
{
	char                name[MAX_QPATH];                          ///< Shader name (cache key)
	dx12MaterialStage_t stages[DX12_MAX_MATERIAL_STAGES];        ///< Rendering stages
	int                 numStages;                               ///< Number of active stages
	qboolean            isSky;         ///< surfaceparm sky
	qboolean            isFog;         ///< surfaceparm fog
	qboolean            isTranslucent; ///< surfaceparm trans
	qboolean            isNodraw;      ///< surfaceparm nodraw
	qboolean            valid;         ///< qtrue once successfully built
} dx12Material_t;

/**
 * @struct dx12Globals_t
 * @brief All DirectX 12 state for the renderer
 */
typedef struct
{
	refimport_t ri;

	HWND hWnd;
	int  vidWidth;
	int  vidHeight;

	// Core DX12 objects
	ID3D12Device              *device;
	IDXGISwapChain3           *swapChain;
	ID3D12CommandQueue        *commandQueue;
	ID3D12CommandAllocator    *commandAllocators[DX12_FRAME_COUNT];
	ID3D12GraphicsCommandList *commandList;

	// Dedicated upload command objects – used exclusively by *UploadBuffer()
	// helpers so that resource uploads never touch the per-frame rendering
	// command allocator/list (which may already be open/recording).
	ID3D12CommandAllocator    *uploadCmdAllocator;
	ID3D12GraphicsCommandList *uploadCmdList;

	// Render target heap + targets
	ID3D12DescriptorHeap *rtvHeap;
	ID3D12Resource       *renderTargets[DX12_FRAME_COUNT];
	UINT                 rtvDescriptorSize;

	// Depth stencil heap + per-frame depth buffers (D32_FLOAT)
	ID3D12DescriptorHeap *dsvHeap;
	UINT                  dsvDescriptorSize;
	ID3D12Resource       *depthStencil[DX12_FRAME_COUNT];

	// SRV descriptor heap (shader-visible, for textures)
	ID3D12DescriptorHeap *srvHeap;
	UINT                  srvDescriptorSize;

	// Synchronization
	ID3D12Fence *fence;
	UINT64       fenceValues[DX12_FRAME_COUNT];
	HANDLE       fenceEvent;
	UINT64       nextFenceValue;                                            ///< Monotonically increasing fence counter

	// Root signature + PSO
	ID3D12RootSignature  *rootSignature;
	ID3D12PipelineState  *pipelineState;

	// 2D vertex ring-buffer (upload heap, persistently mapped)
	ID3D12Resource *quadVertexBuffer;   ///< DX12_MAX_2D_VERTS * sizeof(dx12QuadVertex_t) bytes
	UINT8          *quadVBMapped;       ///< Persistently-mapped CPU pointer
	UINT            quadVBOffset;       ///< Next free vertex index (reset each frame)

	// Per-frame state
	float    color2D[4];    ///< Current 2D modulate color set by RE_DX12_SetColor
	qboolean frameOpen;     ///< qtrue between DX12_BeginFrame and DX12_EndFrame

	// Active scissor rectangle for new draw calls (full-screen by default)
	D3D12_RECT currentScissor;

	// 2D draw batch – accumulates consecutive draws with same texture/topology/scissor
	D3D12_GPU_DESCRIPTOR_HANDLE batch2DTexHandle; ///< GPU texture handle for current batch
	D3D12_PRIMITIVE_TOPOLOGY    batch2DTopology;  ///< Primitive topology for current batch
	D3D12_RECT                  batch2DScissor;   ///< Scissor rect captured when batch started
	UINT                        batch2DStart;     ///< Ring-buffer vertex index where batch begins
	UINT                        batch2DCount;     ///< Number of vertices in current batch

	// Frame state
	UINT          frameIndex;

	// Viewport and scissor
	D3D12_VIEWPORT viewport;
	D3D12_RECT     scissorRect;

	qboolean initialized;
} dx12Globals_t;

extern dx12Globals_t dx12;

/**
 * @enum dx12RenderCommand_t
 * @brief Render command IDs matching the engine's renderCommand_t enum.
 *
 * These must stay in sync with the renderCommand_t enum defined in each
 * renderer's tr_local.h (all renderers share the same values).
 */
typedef enum
{
	RC_END_OF_LIST,
	RC_SET_COLOR,
	RC_STRETCH_PIC,
	RC_2DPOLYS,
	RC_ROTATED_PIC,
	RC_STRETCH_PIC_GRADIENT,
	RC_DRAW_SURFS,
	RC_DRAW_BUFFER,
	RC_SWAP_BUFFERS,
	RC_SCREENSHOT,
	RC_VIDEOFRAME,
	RC_RENDERTOTEXTURE,
	RC_FINISH
} dx12RenderCommand_t;

// Function declarations – backend (tr_dx12_backend.cpp)
void DX12_InitSwapchain( void );
qboolean      R_DX12_Init(void);
void          R_DX12_Shutdown(qboolean destroyWindow);
void          R_DX12_RenderCommandList(const void *data);
void          DX12_EndFrame(void);
void          DX12_BeginFrame(void);
dx12Texture_t DX12_CreateTextureFromRGBA(const byte *data, int width, int height, int srvSlot);
void          DX12_WaitForUpload(ID3D12CommandQueue *queue);

// Function declarations – texture registry (dx12_shader.cpp)
void      DX12_InitTextures(void);
void      DX12_ShutdownTextures(void);
qhandle_t DX12_RegisterTexture(const char *name);
dx12Texture_t *DX12_GetTexture(qhandle_t handle);

// Function declarations – 2D drawing (dx12_poly.cpp)
void DX12_DrawStretchPic(float x, float y, float w, float h,
                         float s1, float t1, float s2, float t2, qhandle_t hShader);
void DX12_DrawStretchPicGradient(float x, float y, float w, float h,
                                 float s1, float t1, float s2, float t2,
                                 qhandle_t hShader, const float *gradientColor, int gradientType);
void DX12_Add2dPolys(polyVert_t *polys, int numverts, qhandle_t hShader);
void DX12_Flush2D(void);
void DX12_SetScissor(int x, int y, int w, int h);
void DX12_DrawString(float x, float y, float scale, const char *text, const fontInfo_t *font);
void DX12_Begin2DBatch(D3D12_GPU_DESCRIPTOR_HANDLE texHandle, D3D12_PRIMITIVE_TOPOLOGY topology);
void DX12_AddQuadToBatch(const dx12QuadVertex_t corners[4]);
void DX12_Flush2DBatch(void);

// Function declarations – material cache (dx12_shader.cpp)
qhandle_t       DX12_RegisterMaterial(const char *name);
dx12Material_t *DX12_GetMaterial(qhandle_t handle);

// Function declarations – world geometry (dx12_world.cpp)
void DX12_LoadWorld(const char *name);
void DX12_ShutdownWorld(void);

// Function declarations – 3D scene rendering (dx12_scene.cpp)
qboolean DX12_SceneInit(void);
void     DX12_SceneShutdown(void);
void     DX12_ClearScene(void);
void     DX12_AddEntityToScene(const refEntity_t *re);
void     DX12_RenderScene(const refdef_t *fd);

// Function declarations – scratch textures for cinematics (tr_dx12_main.cpp)
void DX12_ShutdownScratchTextures(void);


void DX12_StripExtension( const char* in, char* out, int size );
int DX12_Stricmp( const char* s1, const char* s2 );


#endif // _WIN32

#endif // TR_DX12_LOCAL_H
