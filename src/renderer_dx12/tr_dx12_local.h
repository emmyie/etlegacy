/*
 * Wolfenstein: Enemy Territory GPL Source Code
 * Copyright (C) 1999-2010 id Software LLC, a ZeniMax Media company.
 *
 * ET: Legacy
 * Copyright (C) 2012-2024 ET:Legacy team <mail@etlegacy.com>
 *
 * This file is part of ET: Legacy - http://www.etlegacy.com
 *
 * ET: Legacy is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ET: Legacy is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ET: Legacy. If not, see <http://www.gnu.org/licenses/>.
 *
 * In addition, Wolfenstein: Enemy Territory GPL Source Code is also
 * subject to certain additional terms. You should have received a copy
 * of these additional terms immediately following the terms and conditions
 * of the GNU General Public License which accompanied the source code.
 * If not, please request a copy in writing from id Software at the address below.
 *
 * id Software LLC, c/o ZeniMax Media Inc., Suite 120, Rockville, Maryland 20850 USA.
 */
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
 * @struct dx12Vertex_t
 * @brief A single colored vertex for the triangle demo
 */
typedef struct
{
	float pos[3];
	float color[4];
} dx12Vertex_t;

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

	// Synchronization
	ID3D12Fence *fence;
	UINT64       fenceValues[DX12_FRAME_COUNT];
	HANDLE       fenceEvent;

	// Root signature + PSO
	ID3D12RootSignature  *rootSignature;
	ID3D12PipelineState  *pipelineState;

	// Vertex buffer
	ID3D12Resource    *vertexBuffer;
	D3D12_VERTEX_BUFFER_VIEW vertexBufferView;

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
void R_DX12_Init(void);
void R_DX12_Shutdown(qboolean destroyWindow);
void R_DX12_RenderCommandList(const void *data);
void R_DX12_SwapBuffers(void);

#endif // _WIN32

#endif // TR_DX12_LOCAL_H
