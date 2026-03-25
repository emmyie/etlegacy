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

#define DX12_FRAME_COUNT 2

/**
 * @struct dx12QuadVertex_t
 * @brief A single vertex for the textured quad (2D clip-space position + UV)
 */
typedef struct
{
	float pos[2];
	float uv[2];
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

	// Quad vertex buffer (upload heap, updated each draw)
	ID3D12Resource           *quadVertexBuffer;
	D3D12_VERTEX_BUFFER_VIEW  quadVertexBufferView;

	// Test texture (checkerboard)
	dx12Texture_t testTexture;

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

// Function declarations
qboolean      R_DX12_Init(void);
void          R_DX12_Shutdown(qboolean destroyWindow);
void          R_DX12_RenderCommandList(const void *data);
void          R_DX12_SwapBuffers(void);
dx12Texture_t DX12_CreateTextureFromRGBA(const byte *data, int width, int height);
void          DX12_DrawTexturedQuad(float x, float y, float w, float h, dx12Texture_t *tex);

#endif // _WIN32

#endif // TR_DX12_LOCAL_H
