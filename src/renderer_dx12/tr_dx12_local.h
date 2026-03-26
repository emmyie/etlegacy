/**
 * @file tr_dx12_local.h
 * @brief Private header for the DirectX 12 renderer backend
 */

#ifndef TR_DX12_LOCAL_H
#define TR_DX12_LOCAL_H

#include "../qcommon/q_shared.h"
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

	// Render target heap + targets
	ID3D12DescriptorHeap *rtvHeap;
	ID3D12Resource       *renderTargets[DX12_FRAME_COUNT];
	UINT                 rtvDescriptorSize;

	// SRV descriptor heap (shader-visible, for textures)
	ID3D12DescriptorHeap *srvHeap;
	UINT                  srvDescriptorSize;

	// Synchronization
	ID3D12Fence *fence;
	UINT64       fenceValues[DX12_FRAME_COUNT];
	HANDLE       fenceEvent;

	// Root signature + PSO
	ID3D12RootSignature  *rootSignature;
	ID3D12PipelineState  *pipelineState;

	// 2D vertex ring-buffer (upload heap, persistently mapped)
	ID3D12Resource *quadVertexBuffer;   ///< DX12_MAX_2D_VERTS * sizeof(dx12QuadVertex_t) bytes
	UINT8          *quadVBMapped;       ///< Persistently-mapped CPU pointer
	UINT            quadVBOffset;       ///< Next free vertex index (reset each frame)

	// Per-frame state
	float    color2D[4];    ///< Current 2D modulate color set by RE_DX12_SetColor
	qboolean frameOpen;     ///< qtrue between DX12_BeginFrameRender and R_DX12_SwapBuffers

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
void          R_DX12_SwapBuffers(void);
void          DX12_BeginFrameRender(void);
dx12Texture_t DX12_CreateTextureFromRGBA(const byte *data, int width, int height, int srvSlot);

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


void DX12_StripExtension( const char* in, char* out, int size );
int DX12_Stricmp( const char* s1, const char* s2 );


#endif // _WIN32

#endif // TR_DX12_LOCAL_H
