/**
 * @file tr_dx12_backend.cpp
 * @brief DirectX 12 renderer backend – initialization, rendering and shutdown
 */

#include "tr_dx12_local.h"

#ifdef _WIN32

// ---------------------------------------------------------------------------
// HLSL shader source (inlined to avoid file-system dependency at startup)
// ---------------------------------------------------------------------------

static const char *g_shaderSource =
	"Texture2D    g_texture : register(t0);\n"
	"SamplerState g_sampler : register(s0);\n"
	"\n"
	"struct VSInput { float2 pos : POSITION; float2 uv : TEXCOORD; };\n"
	"struct PSInput { float4 pos : SV_POSITION; float2 uv : TEXCOORD; };\n"
	"\n"
	"PSInput VSMain(VSInput input)\n"
	"{\n"
	"    PSInput o;\n"
	"    o.pos = float4(input.pos, 0.0, 1.0);\n"
	"    o.uv  = input.uv;\n"
	"    return o;\n"
	"}\n"
	"\n"
	"float4 PSMain(PSInput input) : SV_TARGET\n"
	"{\n"
	"    return g_texture.Sample(g_sampler, input.uv);\n"
	"}\n";

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * @brief Wait for the GPU to finish processing the current fence value
 */
static void DX12_WaitForGpu(void)
{
	HRESULT hr;

	hr = dx12.commandQueue->Signal(dx12.fence, dx12.fenceValues[dx12.frameIndex]);
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING, "R_DX12: Signal failed (0x%08lx)\n", hr);
		return;
	}

	hr = dx12.fence->SetEventOnCompletion(dx12.fenceValues[dx12.frameIndex], dx12.fenceEvent);
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING, "R_DX12: SetEventOnCompletion failed (0x%08lx)\n", hr);
		return;
	}

	WaitForSingleObjectEx(dx12.fenceEvent, INFINITE, FALSE);
	dx12.fenceValues[dx12.frameIndex]++;
}

/**
 * @brief Advance to the next frame, waiting for the GPU if necessary
 */
static void DX12_MoveToNextFrame(void)
{
	HRESULT hr;
	UINT64  currentFenceValue;

	currentFenceValue = dx12.fenceValues[dx12.frameIndex];

	hr = dx12.commandQueue->Signal(dx12.fence, currentFenceValue);
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING, "R_DX12: MoveToNextFrame Signal failed (0x%08lx)\n", hr);
		return;
	}

	dx12.frameIndex = dx12.swapChain->GetCurrentBackBufferIndex();

	if (dx12.fence->GetCompletedValue() < dx12.fenceValues[dx12.frameIndex])
	{
		hr = dx12.fence->SetEventOnCompletion(dx12.fenceValues[dx12.frameIndex], dx12.fenceEvent);
		if (FAILED(hr))
		{
			dx12.ri.Printf(PRINT_WARNING, "R_DX12: MoveToNextFrame SetEventOnCompletion failed (0x%08lx)\n", hr);
			return;
		}
		WaitForSingleObjectEx(dx12.fenceEvent, INFINITE, FALSE);
	}

	dx12.fenceValues[dx12.frameIndex] = currentFenceValue + 1;
}

// ---------------------------------------------------------------------------
// DX12_CreateTextureFromRGBA
// ---------------------------------------------------------------------------

/**
 * @brief DX12_CreateTextureFromRGBA
 * @param[in] data    RGBA pixel data (width * height * 4 bytes)
 * @param[in] width   Texture width in pixels
 * @param[in] height  Texture height in pixels
 * @return            Populated dx12Texture_t; resource is NULL on failure
 *
 * Creates a D3D12 2D texture, uploads the pixel data via an upload heap,
 * transitions the resource to SRV state, and registers an SRV in slot 0 of
 * dx12.srvHeap.
 */
dx12Texture_t DX12_CreateTextureFromRGBA(const byte *data, int width, int height)
{
	dx12Texture_t tex;
	HRESULT       hr;

	Com_Memset(&tex, 0, sizeof(tex));

	// ---- Default-heap texture resource ----
	D3D12_HEAP_PROPERTIES defaultHeap = {};
	defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

	D3D12_RESOURCE_DESC texDesc = {};
	texDesc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	texDesc.Width            = (UINT)width;
	texDesc.Height           = (UINT)height;
	texDesc.DepthOrArraySize = 1;
	texDesc.MipLevels        = 1;
	texDesc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
	texDesc.SampleDesc.Count = 1;
	texDesc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	texDesc.Flags            = D3D12_RESOURCE_FLAG_NONE;

	hr = dx12.device->CreateCommittedResource(
		&defaultHeap,
		D3D12_HEAP_FLAG_NONE,
		&texDesc,
		D3D12_RESOURCE_STATE_COPY_DEST,
		NULL,
		IID_PPV_ARGS(&tex.resource)
		);
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_CreateTextureFromRGBA: CreateCommittedResource (texture) failed (0x%08lx)\n", hr);
		return tex;
	}

	// ---- Upload heap buffer ----
	D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
	UINT                               numRows;
	UINT64                             rowSizeBytes;
	UINT64                             uploadSize;
	dx12.device->GetCopyableFootprints(&texDesc, 0, 1, 0, &footprint, &numRows, &rowSizeBytes, &uploadSize);

	D3D12_HEAP_PROPERTIES uploadHeap = {};
	uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;

	D3D12_RESOURCE_DESC uploadDesc = {};
	uploadDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
	uploadDesc.Width            = uploadSize;
	uploadDesc.Height           = 1;
	uploadDesc.DepthOrArraySize = 1;
	uploadDesc.MipLevels        = 1;
	uploadDesc.Format           = DXGI_FORMAT_UNKNOWN;
	uploadDesc.SampleDesc.Count = 1;
	uploadDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

	ID3D12Resource *uploadBuffer = NULL;
	hr = dx12.device->CreateCommittedResource(
		&uploadHeap,
		D3D12_HEAP_FLAG_NONE,
		&uploadDesc,
		D3D12_RESOURCE_STATE_GENERIC_READ,
		NULL,
		IID_PPV_ARGS(&uploadBuffer)
		);
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_CreateTextureFromRGBA: CreateCommittedResource (upload) failed (0x%08lx)\n", hr);
		tex.resource->Release();
		tex.resource = NULL;
		return tex;
	}

	// ---- Copy pixel rows into the upload buffer ----
	UINT8      *pMapped   = NULL;
	D3D12_RANGE readRange = { 0, 0 };
	uploadBuffer->Map(0, &readRange, (void **)&pMapped);
	for (UINT row = 0; row < numRows; row++)
	{
		memcpy(
			pMapped + footprint.Offset + (UINT64)row * footprint.Footprint.RowPitch,
			data + (UINT64)row * (UINT)width * 4,
			(size_t)width * 4
			);
	}
	uploadBuffer->Unmap(0, NULL);

	// ---- Record upload + transition into a command list ----
	dx12.commandAllocators[0]->Reset();
	dx12.commandList->Reset(dx12.commandAllocators[0], NULL);

	D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
	srcLoc.pResource        = uploadBuffer;
	srcLoc.Type             = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
	srcLoc.PlacedFootprint  = footprint;

	D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
	dstLoc.pResource        = tex.resource;
	dstLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
	dstLoc.SubresourceIndex = 0;

	dx12.commandList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, NULL);

	D3D12_RESOURCE_BARRIER barrier = {};
	barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	barrier.Transition.pResource   = tex.resource;
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
	barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	dx12.commandList->ResourceBarrier(1, &barrier);

	dx12.commandList->Close();

	ID3D12CommandList *ppCmdLists[] = { dx12.commandList };
	dx12.commandQueue->ExecuteCommandLists(1, ppCmdLists);

	// Wait for upload to finish before releasing the staging buffer
	DX12_WaitForGpu();
	uploadBuffer->Release();

	// ---- Create SRV in slot 0 of the SRV heap ----
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Format                        = DXGI_FORMAT_R8G8B8A8_UNORM;
	srvDesc.ViewDimension                 = D3D12_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Shader4ComponentMapping       = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Texture2D.MipLevels           = 1;
	srvDesc.Texture2D.MostDetailedMip     = 0;
	srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;

	tex.cpuHandle = dx12.srvHeap->GetCPUDescriptorHandleForHeapStart();
	tex.gpuHandle = dx12.srvHeap->GetGPUDescriptorHandleForHeapStart();
	dx12.device->CreateShaderResourceView(tex.resource, &srvDesc, tex.cpuHandle);

	return tex;
}

// ---------------------------------------------------------------------------
// DX12_DrawTexturedQuad
// ---------------------------------------------------------------------------

/**
 * @brief DX12_DrawTexturedQuad
 * @param[in] x    Left edge in screen pixels
 * @param[in] y    Top edge in screen pixels
 * @param[in] w    Width in screen pixels
 * @param[in] h    Height in screen pixels
 * @param[in] tex  Texture to sample
 *
 * Converts pixel coordinates to NDC, writes 4 vertices into the persistent
 * quad upload buffer, binds the SRV heap, sets the descriptor table, and
 * issues a 4-vertex TRIANGLESTRIP draw call.
 *
 * Assumes the caller has already set the root signature, PSO, viewport and
 * scissor on dx12.commandList.
 */
void DX12_DrawTexturedQuad(float x, float y, float w, float h, dx12Texture_t *tex)
{
	// Convert screen-pixel coords to NDC (Y is flipped: screen Y=0 → top)
	float nx1 = (x / (float)dx12.vidWidth) * 2.0f - 1.0f;
	float ny1 = 1.0f - (y / (float)dx12.vidHeight) * 2.0f;
	float nx2 = ((x + w) / (float)dx12.vidWidth) * 2.0f - 1.0f;
	float ny2 = 1.0f - ((y + h) / (float)dx12.vidHeight) * 2.0f;

	dx12QuadVertex_t verts[4] =
	{
		{ { nx1, ny1 }, { 0.0f, 0.0f } },  // top-left
		{ { nx2, ny1 }, { 1.0f, 0.0f } },  // top-right
		{ { nx1, ny2 }, { 0.0f, 1.0f } },  // bottom-left
		{ { nx2, ny2 }, { 1.0f, 1.0f } },  // bottom-right
	};

	UINT8      *pMapped   = NULL;
	D3D12_RANGE readRange = { 0, 0 };
	dx12.quadVertexBuffer->Map(0, &readRange, (void **)&pMapped);
	memcpy(pMapped, verts, sizeof(verts));
	dx12.quadVertexBuffer->Unmap(0, NULL);

	// Bind SRV heap and set descriptor table (root param 0 → t0)
	ID3D12DescriptorHeap *heaps[] = { dx12.srvHeap };
	dx12.commandList->SetDescriptorHeaps(1, heaps);
	dx12.commandList->SetGraphicsRootDescriptorTable(0, tex->gpuHandle);

	dx12.commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	dx12.commandList->IASetVertexBuffers(0, 1, &dx12.quadVertexBufferView);
	dx12.commandList->DrawInstanced(4, 1, 0, 0);
}

// ---------------------------------------------------------------------------
// R_DX12_Init
// ---------------------------------------------------------------------------

/**
 * @brief R_DX12_Init
 *
 * Creates the DXGI factory, D3D12 device, command queue, swap chain, RTV heap,
 * render targets, command allocators, command list, fence, root signature,
 * PSO and vertex buffer for the triangle demo.
 */
qboolean R_DX12_Init(void)
{
	/*dx12.ri.Printf( PRINT_ALL, "R_DX12_Init: stub success\n" );
	return qtrue;*/

	HRESULT hr;
	char    glConfigString[1024];
	glconfig_t glConfig;

	dx12.ri.Printf(PRINT_ALL, "---- R_DX12_Init ----\n");

	// ----------------------------------------------------------------
	// Create the SDL window (without an OpenGL context) via GLimp_Init
	// ----------------------------------------------------------------
	Com_Memset(&glConfig, 0, sizeof(glConfig));
	Com_Memset(glConfigString, 0, sizeof(glConfigString));

	// Mark this as a dx12 window so sdl_glimp.c skips GL context creation
	Info_SetValueForKey(glConfigString, "type", "dx12");

	dx12.ri.GLimp_Init(&glConfig, glConfigString);

	// ----------------------------------------------------------------
	// Get the HWND from SDL via our new ri.GetHWND callback
	// ----------------------------------------------------------------
	if (dx12.ri.GetHWND)
	{
		dx12.hWnd = (HWND)dx12.ri.GetHWND();
	}

	if (!dx12.hWnd)
	{
		dx12.ri.Error(ERR_FATAL, "R_DX12_Init: could not obtain a valid window handle\n");
		return qfalse;
	}

	dx12.vidWidth  = glConfig.vidWidth  > 0 ? glConfig.vidWidth  : 1280;
	dx12.vidHeight = glConfig.vidHeight > 0 ? glConfig.vidHeight : 720;

#ifdef _DEBUG
	// Enable the DX12 debug layer in debug builds
	{
		ID3D12Debug *debugController = NULL;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
		{
			debugController->EnableDebugLayer();
			debugController->Release();
		}
	}
#endif

	// ----------------------------------------------------------------
	// DXGI Factory
	// ----------------------------------------------------------------
	IDXGIFactory4 *factory = NULL;
	hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
	if (FAILED(hr))
	{
		dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateDXGIFactory1 failed (0x%08lx)\n", hr);
		return qfalse;
	}

	// ----------------------------------------------------------------
	// D3D12 Device
	// ----------------------------------------------------------------
	hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dx12.device));
	if (FAILED(hr))
	{
		// Fallback to WARP software device
		IDXGIAdapter *warpAdapter = NULL;
		factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter));
		hr = D3D12CreateDevice(warpAdapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dx12.device));
		if (warpAdapter)
		{
			warpAdapter->Release();
		}
		if (FAILED(hr))
		{
			factory->Release();
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: D3D12CreateDevice failed (0x%08lx)\n", hr);
			return qfalse;
		}
		dx12.ri.Printf(PRINT_ALL, "R_DX12: Using WARP software device\n");
	}

	// ----------------------------------------------------------------
	// Command Queue
	// ----------------------------------------------------------------
	{
		D3D12_COMMAND_QUEUE_DESC queueDesc = {};
		queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		queueDesc.Type  = D3D12_COMMAND_LIST_TYPE_DIRECT;

		hr = dx12.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&dx12.commandQueue));
		if (FAILED(hr))
		{
			factory->Release();
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateCommandQueue failed (0x%08lx)\n", hr);
			return qfalse;
		}
	}

	// ----------------------------------------------------------------
	// Swap Chain
	// ----------------------------------------------------------------
	{
		DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
		swapChainDesc.BufferCount       = DX12_FRAME_COUNT;
		swapChainDesc.Width             = (UINT)dx12.vidWidth;
		swapChainDesc.Height            = (UINT)dx12.vidHeight;
		swapChainDesc.Format            = DXGI_FORMAT_R8G8B8A8_UNORM;
		swapChainDesc.BufferUsage       = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapChainDesc.SwapEffect        = DXGI_SWAP_EFFECT_FLIP_DISCARD;
		swapChainDesc.SampleDesc.Count  = 1;

		IDXGISwapChain1 *swapChain1 = NULL;
		hr = factory->CreateSwapChainForHwnd(
			dx12.commandQueue,
			dx12.hWnd,
			&swapChainDesc,
			NULL,
			NULL,
			&swapChain1
			);
		if (FAILED(hr))
		{
			factory->Release();
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateSwapChainForHwnd failed (0x%08lx)\n", hr);
			return qfalse;
		}

		factory->MakeWindowAssociation(dx12.hWnd, DXGI_MWA_NO_ALT_ENTER);

		hr = swapChain1->QueryInterface(IID_PPV_ARGS(&dx12.swapChain));
		swapChain1->Release();
		if (FAILED(hr))
		{
			factory->Release();
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: QueryInterface(IDXGISwapChain3) failed (0x%08lx)\n", hr);
			return qfalse;
		}

		dx12.frameIndex = dx12.swapChain->GetCurrentBackBufferIndex();
	}

	factory->Release();

	// ----------------------------------------------------------------
	// RTV Descriptor Heap
	// ----------------------------------------------------------------
	{
		D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
		rtvHeapDesc.NumDescriptors = DX12_FRAME_COUNT;
		rtvHeapDesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
		rtvHeapDesc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

		hr = dx12.device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&dx12.rtvHeap));
		if (FAILED(hr))
		{
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateDescriptorHeap failed (0x%08lx)\n", hr);
			return qfalse;
		}

		dx12.rtvDescriptorSize = dx12.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	}

	// ----------------------------------------------------------------
	// SRV Descriptor Heap (shader-visible, 1 slot for the test texture)
	// ----------------------------------------------------------------
	{
		D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
		srvHeapDesc.NumDescriptors = 1;
		srvHeapDesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		srvHeapDesc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

		hr = dx12.device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&dx12.srvHeap));
		if (FAILED(hr))
		{
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateDescriptorHeap (SRV) failed (0x%08lx)\n", hr);
			return qfalse;
		}

		dx12.srvDescriptorSize = dx12.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	// ----------------------------------------------------------------
	// Render Target Views
	// ----------------------------------------------------------------
	{
		D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = dx12.rtvHeap->GetCPUDescriptorHandleForHeapStart();

		for (UINT i = 0; i < DX12_FRAME_COUNT; i++)
		{
			hr = dx12.swapChain->GetBuffer(i, IID_PPV_ARGS(&dx12.renderTargets[i]));
			if (FAILED(hr))
			{
				dx12.ri.Error(ERR_FATAL, "R_DX12_Init: GetBuffer(%u) failed (0x%08lx)\n", i, hr);
				return qfalse;
			}
			dx12.device->CreateRenderTargetView(dx12.renderTargets[i], NULL, rtvHandle);
			rtvHandle.ptr += dx12.rtvDescriptorSize;
		}
	}

	// ----------------------------------------------------------------
	// Command Allocators
	// ----------------------------------------------------------------
	for (UINT i = 0; i < DX12_FRAME_COUNT; i++)
	{
		hr = dx12.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
		                                         IID_PPV_ARGS(&dx12.commandAllocators[i]));
		if (FAILED(hr))
		{
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateCommandAllocator[%u] failed (0x%08lx)\n", i, hr);
			return qfalse;
		}
	}

	// ----------------------------------------------------------------
	// Root Signature: one descriptor table (SRV at t0) + static sampler
	// ----------------------------------------------------------------
	{
		// One SRV range: t0, 1 descriptor
		D3D12_DESCRIPTOR_RANGE srvRange = {};
		srvRange.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		srvRange.NumDescriptors                    = 1;
		srvRange.BaseShaderRegister                = 0;
		srvRange.RegisterSpace                     = 0;
		srvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

		// Root param 0: descriptor table visible to pixel shader
		D3D12_ROOT_PARAMETER rootParam = {};
		rootParam.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		rootParam.DescriptorTable.NumDescriptorRanges = 1;
		rootParam.DescriptorTable.pDescriptorRanges   = &srvRange;
		rootParam.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_PIXEL;

		// Static linear-clamp sampler at s0
		D3D12_STATIC_SAMPLER_DESC staticSampler = {};
		staticSampler.Filter           = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
		staticSampler.AddressU         = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
		staticSampler.AddressV         = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
		staticSampler.AddressW         = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
		staticSampler.MipLODBias       = 0.0f;
		staticSampler.MaxAnisotropy    = 1;
		staticSampler.ComparisonFunc   = D3D12_COMPARISON_FUNC_NEVER;
		staticSampler.BorderColor      = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
		staticSampler.MinLOD           = 0.0f;
		staticSampler.MaxLOD           = 0.0f;
		staticSampler.ShaderRegister   = 0;
		staticSampler.RegisterSpace    = 0;
		staticSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

		D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
		rootSigDesc.NumParameters     = 1;
		rootSigDesc.pParameters       = &rootParam;
		rootSigDesc.NumStaticSamplers = 1;
		rootSigDesc.pStaticSamplers   = &staticSampler;
		rootSigDesc.Flags             = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

		ID3DBlob *signature = NULL;
		ID3DBlob *error     = NULL;

		hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
		                                 &signature, &error);
		if (FAILED(hr))
		{
			if (error)
			{
				dx12.ri.Printf(PRINT_WARNING, "R_DX12: Root signature serialization error: %s\n",
				               (const char *)error->GetBufferPointer());
				error->Release();
			}
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: D3D12SerializeRootSignature failed (0x%08lx)\n", hr);
			return qfalse;
		}

		hr = dx12.device->CreateRootSignature(0, signature->GetBufferPointer(),
		                                      signature->GetBufferSize(),
		                                      IID_PPV_ARGS(&dx12.rootSignature));
		signature->Release();
		if (error)
		{
			error->Release();
		}

		if (FAILED(hr))
		{
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateRootSignature failed (0x%08lx)\n", hr);
			return qfalse;
		}
	}

	// ----------------------------------------------------------------
	// Compile Shaders and Create PSO
	// ----------------------------------------------------------------
	{
		ID3DBlob *vertexShader = NULL;
		ID3DBlob *pixelShader  = NULL;
		ID3DBlob *errorBlob    = NULL;
		UINT      compileFlags = 0;

#ifdef _DEBUG
		compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

		SIZE_T srcLen = strlen(g_shaderSource);

		hr = D3DCompile(g_shaderSource, srcLen, "tr_dx12_shaders", NULL, NULL,
		                "VSMain", "vs_5_0", compileFlags, 0, &vertexShader, &errorBlob);
		if (FAILED(hr))
		{
			if (errorBlob)
			{
				dx12.ri.Printf(PRINT_WARNING, "R_DX12: VS compile error: %s\n",
				               (const char *)errorBlob->GetBufferPointer());
				errorBlob->Release();
			}
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: vertex shader compile failed (0x%08lx)\n", hr);
			return qfalse;
		}
		if (errorBlob) { errorBlob->Release(); errorBlob = NULL; }

		hr = D3DCompile(g_shaderSource, srcLen, "tr_dx12_shaders", NULL, NULL,
		                "PSMain", "ps_5_0", compileFlags, 0, &pixelShader, &errorBlob);
		if (FAILED(hr))
		{
			if (errorBlob)
			{
				dx12.ri.Printf(PRINT_WARNING, "R_DX12: PS compile error: %s\n",
				               (const char *)errorBlob->GetBufferPointer());
				errorBlob->Release();
			}
			vertexShader->Release();
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: pixel shader compile failed (0x%08lx)\n", hr);
			return qfalse;
		}
		if (errorBlob) { errorBlob->Release(); errorBlob = NULL; }

		// Input layout matching dx12QuadVertex_t: float2 pos + float2 uv
		D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0,  0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0,  8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		};

		D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
		psoDesc.InputLayout    = { inputElementDescs, 2 };
		psoDesc.pRootSignature = dx12.rootSignature;
		psoDesc.VS             = { vertexShader->GetBufferPointer(), vertexShader->GetBufferSize() };
		psoDesc.PS             = { pixelShader->GetBufferPointer(),  pixelShader->GetBufferSize() };
		// Default rasterizer: fill, cull back, no depth clip
		{
			D3D12_RASTERIZER_DESC &rs = psoDesc.RasterizerState;
			rs.FillMode              = D3D12_FILL_MODE_SOLID;
			rs.CullMode              = D3D12_CULL_MODE_BACK;
			rs.FrontCounterClockwise = FALSE;
			rs.DepthBias             = D3D12_DEFAULT_DEPTH_BIAS;
			rs.DepthBiasClamp        = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
			rs.SlopeScaledDepthBias  = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
			rs.DepthClipEnable       = TRUE;
			rs.MultisampleEnable     = FALSE;
			rs.AntialiasedLineEnable = FALSE;
			rs.ForcedSampleCount     = 0;
			rs.ConservativeRaster    = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
		}

		// Default blend: no blending
		{
			D3D12_BLEND_DESC &bd         = psoDesc.BlendState;
			bd.AlphaToCoverageEnable     = FALSE;
			bd.IndependentBlendEnable    = FALSE;
			for (UINT j = 0; j < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT; j++)
			{
				D3D12_RENDER_TARGET_BLEND_DESC &rt = bd.RenderTarget[j];
				rt.BlendEnable           = FALSE;
				rt.LogicOpEnable         = FALSE;
				rt.SrcBlend              = D3D12_BLEND_ONE;
				rt.DestBlend             = D3D12_BLEND_ZERO;
				rt.BlendOp               = D3D12_BLEND_OP_ADD;
				rt.SrcBlendAlpha         = D3D12_BLEND_ONE;
				rt.DestBlendAlpha        = D3D12_BLEND_ZERO;
				rt.BlendOpAlpha          = D3D12_BLEND_OP_ADD;
				rt.LogicOp               = D3D12_LOGIC_OP_NOOP;
				rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
			}
		}
		psoDesc.DepthStencilState.DepthEnable      = FALSE;
		psoDesc.DepthStencilState.StencilEnable    = FALSE;
		psoDesc.SampleMask                         = UINT_MAX;
		psoDesc.PrimitiveTopologyType              = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		psoDesc.NumRenderTargets                   = 1;
		psoDesc.RTVFormats[0]                      = DXGI_FORMAT_R8G8B8A8_UNORM;
		psoDesc.SampleDesc.Count                   = 1;

		hr = dx12.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&dx12.pipelineState));
		vertexShader->Release();
		pixelShader->Release();

		if (FAILED(hr))
		{
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateGraphicsPipelineState failed (0x%08lx)\n", hr);
			return qfalse;
		}
	}

	// ----------------------------------------------------------------
	// Command List
	// ----------------------------------------------------------------
	hr = dx12.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
	                                    dx12.commandAllocators[dx12.frameIndex],
	                                    dx12.pipelineState,
	                                    IID_PPV_ARGS(&dx12.commandList));
	if (FAILED(hr))
	{
		dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateCommandList failed (0x%08lx)\n", hr);
		return qfalse;
	}

	// Command lists are created in the recording state; close immediately.
	dx12.commandList->Close();

	// ----------------------------------------------------------------
	// Quad Vertex Buffer (upload heap, 4 vertices, updated each draw)
	// ----------------------------------------------------------------
	{
		const UINT quadVBSize = sizeof(dx12QuadVertex_t) * 4;

		D3D12_HEAP_PROPERTIES heapProps = {};
		heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

		D3D12_RESOURCE_DESC resDesc = {};
		resDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
		resDesc.Width            = quadVBSize;
		resDesc.Height           = 1;
		resDesc.DepthOrArraySize = 1;
		resDesc.MipLevels        = 1;
		resDesc.Format           = DXGI_FORMAT_UNKNOWN;
		resDesc.SampleDesc.Count = 1;
		resDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

		hr = dx12.device->CreateCommittedResource(
			&heapProps,
			D3D12_HEAP_FLAG_NONE,
			&resDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			NULL,
			IID_PPV_ARGS(&dx12.quadVertexBuffer)
			);
		if (FAILED(hr))
		{
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateCommittedResource (quad VB) failed (0x%08lx)\n", hr);
			return qfalse;
		}

		dx12.quadVertexBufferView.BufferLocation = dx12.quadVertexBuffer->GetGPUVirtualAddress();
		dx12.quadVertexBufferView.StrideInBytes  = sizeof(dx12QuadVertex_t);
		dx12.quadVertexBufferView.SizeInBytes    = quadVBSize;
	}

	// ----------------------------------------------------------------
	// Fence
	// ----------------------------------------------------------------
	hr = dx12.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&dx12.fence));
	if (FAILED(hr))
	{
		dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateFence failed (0x%08lx)\n", hr);
		return qfalse;
	}
	dx12.fenceValues[dx12.frameIndex] = 1;

	dx12.fenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
	if (!dx12.fenceEvent)
	{
		dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateEvent failed\n");
		return qfalse;
	}

	// ----------------------------------------------------------------
	// Viewport and Scissor
	// ----------------------------------------------------------------
	dx12.viewport.TopLeftX = 0.0f;
	dx12.viewport.TopLeftY = 0.0f;
	dx12.viewport.Width    = (float)dx12.vidWidth;
	dx12.viewport.Height   = (float)dx12.vidHeight;
	dx12.viewport.MinDepth = 0.0f;
	dx12.viewport.MaxDepth = 1.0f;

	dx12.scissorRect.left   = 0;
	dx12.scissorRect.top    = 0;
	dx12.scissorRect.right  = (LONG)dx12.vidWidth;
	dx12.scissorRect.bottom = (LONG)dx12.vidHeight;

	// ----------------------------------------------------------------
	// Test texture: 64×64 checkerboard (magenta + black)
	// ----------------------------------------------------------------
	{
		const int   TEX_W    = 64;
		const int   TEX_H    = 64;
		const int   CELL     = 8;
		const int   numBytes = TEX_W * TEX_H * 4;
		byte       *texData  = (byte *)malloc((size_t)numBytes);

		if (texData)
		{
			for (int py = 0; py < TEX_H; py++)
			{
				for (int px = 0; px < TEX_W; px++)
				{
					int  idx   = (py * TEX_W + px) * 4;
					int  white = (((px / CELL) + (py / CELL)) % 2 == 0);
					texData[idx + 0] = white ? 255 : 128; // R
					texData[idx + 1] = white ? 255 :   0; // G
					texData[idx + 2] = white ? 255 : 128; // B
					texData[idx + 3] = 255;               // A
				}
			}

			dx12.testTexture = DX12_CreateTextureFromRGBA(texData, TEX_W, TEX_H);
			free(texData);
		}

		if (!dx12.testTexture.resource)
		{
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: test texture creation failed\n");
			return qfalse;
		}
	}

	dx12.initialized = qtrue;
	dx12.ri.Printf(PRINT_ALL, "R_DX12: Initialized (%dx%d)\n", dx12.vidWidth, dx12.vidHeight);
	return qtrue;
}

// ---------------------------------------------------------------------------
// R_DX12_SwapBuffers – render one frame (clear + triangle) then present
// ---------------------------------------------------------------------------

/**
 * @brief R_DX12_SwapBuffers
 *
 * Resets the command allocator and list, transitions the back buffer,
 * clears the screen, draws the triangle, transitions back to present state,
 * executes and presents.
 */
void R_DX12_SwapBuffers(void)
{
	HRESULT hr;

	if (!dx12.initialized)
	{
		return;
	}

	// Reset allocator + command list for this frame
	hr = dx12.commandAllocators[dx12.frameIndex]->Reset();
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING, "R_DX12: commandAllocator->Reset failed (0x%08lx)\n", hr);
		return;
	}

	hr = dx12.commandList->Reset(dx12.commandAllocators[dx12.frameIndex], dx12.pipelineState);
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING, "R_DX12: commandList->Reset failed (0x%08lx)\n", hr);
		return;
	}

	// Transition back buffer: PRESENT → RENDER_TARGET
	D3D12_RESOURCE_BARRIER barrierToRT = {};
	barrierToRT.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrierToRT.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	barrierToRT.Transition.pResource   = dx12.renderTargets[dx12.frameIndex];
	barrierToRT.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
	barrierToRT.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
	barrierToRT.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	dx12.commandList->ResourceBarrier(1, &barrierToRT);

	// Get the RTV handle for the current back buffer
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = dx12.rtvHeap->GetCPUDescriptorHandleForHeapStart();
	rtvHandle.ptr += dx12.frameIndex * dx12.rtvDescriptorSize;

	dx12.commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, NULL);

	// Clear to a dark background
	const float clearColor[] = { 0.05f, 0.05f, 0.15f, 1.0f };
	dx12.commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, NULL);

	// Bind root signature + PSO, then draw the textured quad
	dx12.commandList->SetGraphicsRootSignature(dx12.rootSignature);
	dx12.commandList->SetPipelineState(dx12.pipelineState);
	dx12.commandList->RSSetViewports(1, &dx12.viewport);
	dx12.commandList->RSSetScissorRects(1, &dx12.scissorRect);

	DX12_DrawTexturedQuad(0.0f, 0.0f, (float)dx12.vidWidth, (float)dx12.vidHeight,
	                      &dx12.testTexture);

	// Transition back buffer: RENDER_TARGET → PRESENT
	D3D12_RESOURCE_BARRIER barrierToPresent = {};
	barrierToPresent.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrierToPresent.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	barrierToPresent.Transition.pResource   = dx12.renderTargets[dx12.frameIndex];
	barrierToPresent.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
	barrierToPresent.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
	barrierToPresent.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	dx12.commandList->ResourceBarrier(1, &barrierToPresent);

	// Close and execute the command list
	hr = dx12.commandList->Close();
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING, "R_DX12: commandList->Close failed (0x%08lx)\n", hr);
		return;
	}

	ID3D12CommandList *ppCommandLists[] = { dx12.commandList };
	dx12.commandQueue->ExecuteCommandLists(1, ppCommandLists);

	// Present with vsync disabled (0 sync interval)
	hr = dx12.swapChain->Present(0, 0);
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING, "R_DX12: Present failed (0x%08lx)\n", hr);
	}

	DX12_MoveToNextFrame();
}

// ---------------------------------------------------------------------------
// R_DX12_RenderCommandList – command dispatcher
// ---------------------------------------------------------------------------

/**
 * @brief R_DX12_RenderCommandList
 * @param[in] data  Render command list buffer
 *
 * Walks the ET:Legacy render command list. Calls R_DX12_SwapBuffers on
 * RC_SWAP_BUFFERS; ignores all other commands.
 */
void R_DX12_RenderCommandList(const void *data)
{
	if (!dx12.initialized)
	{
		return;
	}

	while (1)
	{
		data = PADP(data, sizeof(intptr_t));

		switch (*(const int *)data)
		{
		case RC_SWAP_BUFFERS:
			R_DX12_SwapBuffers();
			// Advance past the swapBuffersCommand_t (just an int commandId)
			data = (const char *)data + sizeof(int);
			break;

		case RC_END_OF_LIST:
		default:
			return;
		}
	}
}

// ---------------------------------------------------------------------------
// R_DX12_Shutdown
// ---------------------------------------------------------------------------

/**
 * @brief R_DX12_Shutdown
 * @param[in] destroyWindow  Whether to also tear down the SDL window
 *
 * Waits for the GPU to finish, then releases all DX12 resources in reverse
 * creation order.
 */
void R_DX12_Shutdown(qboolean destroyWindow)
{
	dx12.ri.Printf(PRINT_ALL, "---- R_DX12_Shutdown ----\n");

	if (!dx12.initialized)
	{
		return;
	}

	// Wait for GPU to finish outstanding work
	DX12_WaitForGpu();

	if (dx12.fenceEvent)
	{
		CloseHandle(dx12.fenceEvent);
		dx12.fenceEvent = NULL;
	}

	if (dx12.fence)
	{
		dx12.fence->Release();
		dx12.fence = NULL;
	}

	if (dx12.testTexture.resource)
	{
		dx12.testTexture.resource->Release();
		dx12.testTexture.resource = NULL;
	}

	if (dx12.quadVertexBuffer)
	{
		dx12.quadVertexBuffer->Release();
		dx12.quadVertexBuffer = NULL;
	}

	if (dx12.pipelineState)
	{
		dx12.pipelineState->Release();
		dx12.pipelineState = NULL;
	}

	if (dx12.rootSignature)
	{
		dx12.rootSignature->Release();
		dx12.rootSignature = NULL;
	}

	if (dx12.commandList)
	{
		dx12.commandList->Release();
		dx12.commandList = NULL;
	}

	for (UINT i = 0; i < DX12_FRAME_COUNT; i++)
	{
		if (dx12.commandAllocators[i])
		{
			dx12.commandAllocators[i]->Release();
			dx12.commandAllocators[i] = NULL;
		}
	}

	for (UINT i = 0; i < DX12_FRAME_COUNT; i++)
	{
		if (dx12.renderTargets[i])
		{
			dx12.renderTargets[i]->Release();
			dx12.renderTargets[i] = NULL;
		}
	}

	if (dx12.rtvHeap)
	{
		dx12.rtvHeap->Release();
		dx12.rtvHeap = NULL;
	}

	if (dx12.srvHeap)
	{
		dx12.srvHeap->Release();
		dx12.srvHeap = NULL;
	}

	if (dx12.swapChain)
	{
		dx12.swapChain->Release();
		dx12.swapChain = NULL;
	}

	if (dx12.commandQueue)
	{
		dx12.commandQueue->Release();
		dx12.commandQueue = NULL;
	}

	if (dx12.device)
	{
		dx12.device->Release();
		dx12.device = NULL;
	}

	dx12.initialized = qfalse;

	if (destroyWindow)
	{
		dx12.ri.GLimp_Shutdown();
	}
}

#endif // _WIN32
