/**
 * @file tr_dx12_backend.cpp
 * @brief DirectX 12 renderer backend – initialization, rendering and shutdown
 */

#include "tr_dx12_local.h"
#include "dx12_shader.h"
#include "dx12_world.h"
#include "dx12_model.h"
#include "dx12_scene.h"

#ifdef _WIN32

// ---------------------------------------------------------------------------
// HLSL shader source (inlined to avoid file-system dependency at startup)
// ---------------------------------------------------------------------------

static const char *g_shaderSource =
	"Texture2D    g_texture : register(t0);\n"
	"SamplerState g_sampler : register(s0);\n"
	"\n"
	"struct VSInput { float2 pos : POSITION; float2 uv : TEXCOORD; float4 col : COLOR; };\n"
	"struct PSInput { float4 pos : SV_POSITION; float2 uv : TEXCOORD; float4 col : COLOR; };\n"
	"\n"
	"PSInput VSMain(VSInput input)\n"
	"{\n"
	"    PSInput o;\n"
	"    o.pos = float4(input.pos, 0.0, 1.0);\n"
	"    o.uv  = input.uv;\n"
	"    o.col = input.col;\n"
	"    return o;\n"
	"}\n"
	"\n"
	"float4 PSMain(PSInput input) : SV_TARGET\n"
	"{\n"
	"    return g_texture.Sample(g_sampler, input.uv) * input.col;\n"
	"}\n";

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * @brief Wait for all outstanding GPU work on the main queue to complete.
 *
 * Uses a strictly-increasing signal value (++nextFenceValue) so the fence
 * value is always greater than any previously signaled value.  Using the
 * per-frame fenceValues[] as the signal target is incorrect because those
 * values were already signaled by DX12_MoveToNextFrame, and D3D12 requires
 * each new Signal call to use a strictly increasing value.
 */
static void DX12_WaitForGpu(void)
{
	HRESULT      hr;
	const UINT64 signalValue = ++dx12.nextFenceValue;

	hr = dx12.commandQueue->Signal(dx12.fence, signalValue);
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING, "R_DX12: Signal failed (0x%08lx)\n", hr);
		return;
	}

	if (dx12.fence->GetCompletedValue() < signalValue)
	{
		hr = dx12.fence->SetEventOnCompletion(signalValue, dx12.fenceEvent);
		if (FAILED(hr))
		{
			dx12.ri.Printf(PRINT_WARNING, "R_DX12: SetEventOnCompletion failed (0x%08lx)\n", hr);
			return;
		}
		WaitForSingleObjectEx(dx12.fenceEvent, INFINITE, FALSE);
	}

	// Record the signaled value for the current frame slot so MoveToNextFrame
	// can skip an unnecessary wait if the GPU already caught up.
	dx12.fenceValues[dx12.frameIndex] = signalValue;
}

/**
 * @brief DX12_FlushGpu
 *
 * Public thin wrapper around DX12_WaitForGpu so that callers outside this
 * translation unit (e.g. RE_DX12_purgeCache in tr_dx12_main.cpp) can flush
 * the GPU pipeline without needing access to the static helper.
 */
void DX12_FlushGpu(void)
{
	if (!dx12.initialized)
	{
		return;
	}
	DX12_WaitForGpu();
}

// ---------------------------------------------------------------------------
// DX12_CopyRenderTargetToTexture
// ---------------------------------------------------------------------------

/**
 * @brief DX12_CopyRenderTargetToTexture
 *
 * Mirrors GL's glCopyTexImage2D: copies the rectangular region
 * (x, y, x+w, y+h) of the current back-buffer into the DX12 texture
 * identified by @p entry.  If the texture resource does not yet exist
 * or its dimensions differ from (w × h) it is (re-)created first.
 *
 * Caller must ensure dx12.frameOpen is qtrue (i.e. DX12_BeginFrame has been
 * called and DX12_EndFrame has not yet been called).
 *
 * @param[in,out] entry   Pointer into dx12Shaders[].  entry->tex.resource may
 *                        be replaced if the dimensions do not match.
 * @param[in]     srvSlot SRV heap slot for this texture (= index in dx12Shaders[]).
 * @param[in]     x       Left pixel of the source rectangle in back-buffer coords.
 * @param[in]     y       Top  pixel of the source rectangle in back-buffer coords.
 * @param[in]     w       Width  of the rectangle in pixels.
 * @param[in]     h       Height of the rectangle in pixels.
 */
void DX12_CopyRenderTargetToTexture(dx12ShaderEntry_t *entry, int srvSlot,
                                    int x, int y, int w, int h)
{
	D3D12_RESOURCE_BARRIER          barriers[2];
	D3D12_TEXTURE_COPY_LOCATION     srcLoc;
	D3D12_TEXTURE_COPY_LOCATION     dstLoc;
	D3D12_BOX                       srcBox;
	D3D12_CPU_DESCRIPTOR_HANDLE     rtvHandle;
	D3D12_CPU_DESCRIPTOR_HANDLE     dsvHandle;
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
	D3D12_HEAP_PROPERTIES           defaultHeap;
	D3D12_RESOURCE_DESC             texDesc;
	HRESULT                         hr;
	qboolean                        hasDsv;

	if (!dx12.initialized || !dx12.frameOpen)
	{
		return;
	}

	if (w <= 0 || h <= 0)
	{
		return;
	}

	// Flush any pending 2D draw batch so its commands are recorded before the
	// resource-barrier that transitions the render target away from RTV state.
	DX12_Flush2D();

	// ---- 1. (Re-)create the texture resource if dimensions changed ----------
	if (!entry->tex.resource || entry->width != w || entry->height != h)
	{
		if (entry->tex.resource)
		{
			entry->tex.resource->Release();
			entry->tex.resource = NULL;
		}

		Com_Memset(&defaultHeap, 0, sizeof(defaultHeap));
		defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

		Com_Memset(&texDesc, 0, sizeof(texDesc));
		texDesc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		texDesc.Width            = (UINT)w;
		texDesc.Height           = (UINT)h;
		texDesc.DepthOrArraySize = 1;
		texDesc.MipLevels        = 1;
		texDesc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
		texDesc.SampleDesc.Count = 1;
		texDesc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		texDesc.Flags            = D3D12_RESOURCE_FLAG_NONE;

		// Start in COPY_DEST – we are about to copy into it immediately.
		hr = dx12.device->CreateCommittedResource(
			&defaultHeap,
			D3D12_HEAP_FLAG_NONE,
			&texDesc,
			D3D12_RESOURCE_STATE_COPY_DEST,
			NULL,
			IID_PPV_ARGS(&entry->tex.resource));

		if (FAILED(hr))
		{
			dx12.ri.Printf(PRINT_WARNING,
			               "DX12_CopyRenderTargetToTexture: CreateCommittedResource failed (0x%08lx)\n",
			               hr);
			return;
		}

		entry->width  = w;
		entry->height = h;

		// Re-create SRV at the same heap slot.
		Com_Memset(&srvDesc, 0, sizeof(srvDesc));
		srvDesc.Format                        = DXGI_FORMAT_R8G8B8A8_UNORM;
		srvDesc.ViewDimension                 = D3D12_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Shader4ComponentMapping       = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Texture2D.MipLevels           = 1;
		srvDesc.Texture2D.MostDetailedMip     = 0;
		srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;

		entry->tex.cpuHandle.ptr = dx12.srvHeap->GetCPUDescriptorHandleForHeapStart().ptr
		                           + (SIZE_T)srvSlot * dx12.srvDescriptorSize;
		entry->tex.gpuHandle.ptr = dx12.srvHeap->GetGPUDescriptorHandleForHeapStart().ptr
		                           + (UINT64)srvSlot * dx12.srvDescriptorSize;
		dx12.device->CreateShaderResourceView(entry->tex.resource, &srvDesc,
		                                      entry->tex.cpuHandle);

		// Texture is already in COPY_DEST – skip the transition barrier below.
		goto do_copy;
	}

	// ---- 2. Transition existing texture: PIXEL_SHADER_RESOURCE → COPY_DEST -
	Com_Memset(&barriers[0], 0, sizeof(barriers[0]));
	barriers[0].Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barriers[0].Transition.pResource   = entry->tex.resource;
	barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
	barriers[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
	barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	dx12.commandList->ResourceBarrier(1, &barriers[0]);

do_copy:
	// ---- 3. Transition render target: RENDER_TARGET → COPY_SOURCE ----------
	Com_Memset(&barriers[0], 0, sizeof(barriers[0]));
	barriers[0].Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barriers[0].Transition.pResource   = dx12.renderTargets[dx12.frameIndex];
	barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
	barriers[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
	barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	dx12.commandList->ResourceBarrier(1, &barriers[0]);

	// ---- 4. Copy the back-buffer sub-region into the texture ----------------
	Com_Memset(&srcLoc, 0, sizeof(srcLoc));
	srcLoc.pResource        = dx12.renderTargets[dx12.frameIndex];
	srcLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
	srcLoc.SubresourceIndex = 0;

	Com_Memset(&dstLoc, 0, sizeof(dstLoc));
	dstLoc.pResource        = entry->tex.resource;
	dstLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
	dstLoc.SubresourceIndex = 0;

	srcBox.left   = (UINT)x;
	srcBox.top    = (UINT)y;
	srcBox.front  = 0;
	srcBox.right  = (UINT)(x + w);
	srcBox.bottom = (UINT)(y + h);
	srcBox.back   = 1;

	dx12.commandList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, &srcBox);

	// ---- 5. Transition both resources back to their rendering states --------
	Com_Memset(&barriers[0], 0, sizeof(barriers[0]));
	barriers[0].Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barriers[0].Transition.pResource   = dx12.renderTargets[dx12.frameIndex];
	barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
	barriers[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
	barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

	Com_Memset(&barriers[1], 0, sizeof(barriers[1]));
	barriers[1].Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barriers[1].Transition.pResource   = entry->tex.resource;
	barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barriers[1].Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
	barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

	dx12.commandList->ResourceBarrier(2, barriers);

	// ---- 6. Re-bind the render target so subsequent 2D drawing continues ----
	rtvHandle.ptr = dx12.rtvHeap->GetCPUDescriptorHandleForHeapStart().ptr
	                + dx12.frameIndex * dx12.rtvDescriptorSize;

	hasDsv = (dx12.dsvHeap && dx12.depthStencil[dx12.frameIndex]) ? qtrue : qfalse;
	if (hasDsv)
	{
		dsvHandle.ptr = dx12.dsvHeap->GetCPUDescriptorHandleForHeapStart().ptr
		                + dx12.frameIndex * dx12.dsvDescriptorSize;
		dx12.commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
	}
	else
	{
		dx12.commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, NULL);
	}
}

// ---------------------------------------------------------------------------
// DX12_ReadbackRenderTarget
// ---------------------------------------------------------------------------

/**
 * @brief DX12_ReadbackRenderTarget
 *
 * Reads the current back-buffer into @p rgbOut as packed RGB bytes with
 * 4-byte row padding, matching what glReadPixels(GL_RGB, GL_UNSIGNED_BYTE)
 * with the default GL_PACK_ALIGNMENT of 4 would produce.  This is what the
 * engine's AVI capture code expects in its captureBuffer.
 *
 * The function:
 *   1. Flushes any pending 2D draw batch.
 *   2. Allocates a GPU readback (UPLOAD heap) buffer sized by the D3D12
 *      copyable footprint of the back-buffer.
 *   3. Transitions the back-buffer: RENDER_TARGET → COPY_SOURCE.
 *   4. CopyTextureRegion → readback buffer.
 *   5. Transitions back-buffer: COPY_SOURCE → RENDER_TARGET and re-binds RTV.
 *   6. Closes + executes the command list, then waits for the GPU.
 *   7. Maps the readback buffer, converts RGBA rows to RGB rows (padded to
 *      4 bytes) into @p rgbOut, then unmaps and releases the readback buffer.
 *   8. Re-opens the command list (resets allocator + list, restores root sig,
 *      PSO, viewport, scissor, heap, and RTV bindings) so the caller can
 *      continue recording draw commands for the rest of the frame.
 *
 * @param[out] rgbOut   Caller-supplied buffer; must be at least
 *                      PAD(width*3, 4) * height bytes.
 * @param[in]  width    Frame width in pixels.
 * @param[in]  height   Frame height in pixels.
 * @return qtrue on success, qfalse on any D3D12 or guard failure.
 */
qboolean DX12_ReadbackRenderTarget(byte *rgbOut, int width, int height)
{
	D3D12_HEAP_PROPERTIES              readbackHeap;
	D3D12_RESOURCE_DESC                bufDesc;
	D3D12_RESOURCE_DESC                rtDesc;
	D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
	UINT                               numRows;
	UINT64                             rowSizeBytes;
	UINT64                             totalBytes;
	ID3D12Resource                     *readbackBuf;
	D3D12_RESOURCE_BARRIER             barriers[2];
	D3D12_TEXTURE_COPY_LOCATION        srcLoc;
	D3D12_TEXTURE_COPY_LOCATION        dstLoc;
	D3D12_CPU_DESCRIPTOR_HANDLE        rtvHandle;
	ID3D12DescriptorHeap               *heaps[1];
	HRESULT                            hr;
	UINT64                             readbackRowPitch;
	int                                row;
	int                                linelen;
	int                                padwidth;
	void                               *mapped;
	const byte                         *src;
	byte                               *dst;
	int                                col;
	qboolean                           success;

	success = qtrue;

	if (!dx12.initialized || !dx12.frameOpen)
	{
		return qfalse;
	}

	if (!rgbOut || width <= 0 || height <= 0)
	{
		return qfalse;
	}

	// --- 1. Flush 2D batch ------------------------------------------------
	DX12_Flush2D();

	// --- 2. Allocate readback buffer ----------------------------------------
	// Determine the D3D12 copyable footprint for the back-buffer's subresource
	// so we get the correct (256-byte-aligned) row pitch.
	rtDesc = dx12.renderTargets[dx12.frameIndex]->GetDesc();
	dx12.device->GetCopyableFootprints(&rtDesc, 0, 1, 0,
	                                   &footprint, &numRows, &rowSizeBytes, &totalBytes);
	readbackRowPitch = footprint.Footprint.RowPitch;   // already 256-aligned

	Com_Memset(&readbackHeap, 0, sizeof(readbackHeap));
	readbackHeap.Type = D3D12_HEAP_TYPE_READBACK;

	Com_Memset(&bufDesc, 0, sizeof(bufDesc));
	bufDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
	bufDesc.Width            = totalBytes;
	bufDesc.Height           = 1;
	bufDesc.DepthOrArraySize = 1;
	bufDesc.MipLevels        = 1;
	bufDesc.Format           = DXGI_FORMAT_UNKNOWN;
	bufDesc.SampleDesc.Count = 1;
	bufDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	bufDesc.Flags            = D3D12_RESOURCE_FLAG_NONE;

	readbackBuf = NULL;
	hr          = dx12.device->CreateCommittedResource(
		&readbackHeap,
		D3D12_HEAP_FLAG_NONE,
		&bufDesc,
		D3D12_RESOURCE_STATE_COPY_DEST,
		NULL,
		IID_PPV_ARGS(&readbackBuf));
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "DX12_ReadbackRenderTarget: CreateCommittedResource failed (0x%08lx)\n", hr);
		return qfalse;
	}

	// --- 3. Transition back-buffer: RENDER_TARGET → COPY_SOURCE -------------
	Com_Memset(&barriers[0], 0, sizeof(barriers[0]));
	barriers[0].Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barriers[0].Transition.pResource   = dx12.renderTargets[dx12.frameIndex];
	barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
	barriers[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
	barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	dx12.commandList->ResourceBarrier(1, &barriers[0]);

	// --- 4. Copy texture → readback buffer -----------------------------------
	Com_Memset(&srcLoc, 0, sizeof(srcLoc));
	srcLoc.pResource        = dx12.renderTargets[dx12.frameIndex];
	srcLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
	srcLoc.SubresourceIndex = 0;

	Com_Memset(&dstLoc, 0, sizeof(dstLoc));
	dstLoc.pResource       = readbackBuf;
	dstLoc.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
	dstLoc.PlacedFootprint = footprint;

	dx12.commandList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, NULL);

	// --- 5. Transition back-buffer: COPY_SOURCE → RENDER_TARGET + rebind ----
	Com_Memset(&barriers[0], 0, sizeof(barriers[0]));
	barriers[0].Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barriers[0].Transition.pResource   = dx12.renderTargets[dx12.frameIndex];
	barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
	barriers[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
	barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	dx12.commandList->ResourceBarrier(1, &barriers[0]);

	rtvHandle.ptr = dx12.rtvHeap->GetCPUDescriptorHandleForHeapStart().ptr
	                + dx12.frameIndex * dx12.rtvDescriptorSize;
	if (dx12.dsvHeap && dx12.depthStencil[dx12.frameIndex])
	{
		D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;

		dsvHandle.ptr = dx12.dsvHeap->GetCPUDescriptorHandleForHeapStart().ptr
		                + dx12.frameIndex * dx12.dsvDescriptorSize;
		dx12.commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
	}
	else
	{
		dx12.commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, NULL);
	}

	// --- 6. Execute and wait for GPU -----------------------------------------
	dx12.commandList->Close();
	{
		ID3D12CommandList *lists[] = { dx12.commandList };

		dx12.commandQueue->ExecuteCommandLists(1, lists);
	}
	DX12_WaitForUpload(dx12.commandQueue);

	// --- 7. Map, convert RGBA → RGB (padwidth = PAD(width*3, 4)), unmap ------
	mapped = NULL;
	{
		D3D12_RANGE readRange = { 0, (SIZE_T)totalBytes };

		hr = readbackBuf->Map(0, &readRange, &mapped);
	}
	if (FAILED(hr) || !mapped)
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "DX12_ReadbackRenderTarget: Map failed (0x%08lx)\n", hr);
		readbackBuf->Release();
		readbackBuf = NULL;
		success     = qfalse;
		// Still need to re-open the command list before returning.
		goto reopen;
	}

	linelen  = width * 3;
	padwidth = (linelen + 3) & ~3;   // PAD(linelen, 4)

	for (row = 0; row < height; row++)
	{
		src = (const byte *)mapped + (UINT64)row * readbackRowPitch;
		dst = rgbOut + (UINT64)row * padwidth;
		for (col = 0; col < width; col++)
		{
			dst[col * 3 + 0] = src[col * 4 + 0];   // R
			dst[col * 3 + 1] = src[col * 4 + 1];   // G
			dst[col * 3 + 2] = src[col * 4 + 2];   // B
			// alpha channel discarded
		}
		// zero-pad the row tail (0..3 bytes) so callers don't read garbage
		if (padwidth > linelen)
		{
			Com_Memset(dst + linelen, 0, (size_t)(padwidth - linelen));
		}
	}

	{
		D3D12_RANGE writeRange = { 0, 0 };   // we did not write

		readbackBuf->Unmap(0, &writeRange);
	}
	readbackBuf->Release();
	readbackBuf = NULL;

	// --- 8. Re-open command list (reset allocator + list, restore state) -----
reopen:
	{
		D3D12_CPU_DESCRIPTOR_HANDLE reopenDsv;

		dx12.commandAllocators[dx12.frameIndex]->Reset();
		dx12.commandList->Reset(dx12.commandAllocators[dx12.frameIndex], dx12.pipelineState);

		// Back-buffer is already in RENDER_TARGET state – no barrier needed.
		dx12.commandList->SetGraphicsRootSignature(dx12.rootSignature);
		dx12.commandList->SetPipelineState(dx12.pipelineState);
		dx12.commandList->RSSetViewports(1, &dx12.viewport);
		dx12.commandList->RSSetScissorRects(1, &dx12.currentScissor);

		heaps[0] = dx12.srvHeap;
		dx12.commandList->SetDescriptorHeaps(1, heaps);

		if (dx12.dsvHeap && dx12.depthStencil[dx12.frameIndex])
		{
			reopenDsv.ptr = dx12.dsvHeap->GetCPUDescriptorHandleForHeapStart().ptr
			                + dx12.frameIndex * dx12.dsvDescriptorSize;
			dx12.commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &reopenDsv);
		}
		else
		{
			dx12.commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, NULL);
		}
	}

	return success;
}

/**
 * @brief Advance to the next frame, waiting for the GPU if necessary
 */
static void DX12_MoveToNextFrame(void)
{
	HRESULT      hr;
	const UINT64 signalValue = ++dx12.nextFenceValue;

	hr = dx12.commandQueue->Signal(dx12.fence, signalValue);
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING, "R_DX12: MoveToNextFrame Signal failed (0x%08lx)\n", hr);
		return;
	}

	// Save the fence value for this frame slot so BeginFrame can wait on it
	dx12.fenceValues[dx12.frameIndex] = signalValue;

	// Advance to the next back-buffer
	dx12.frameIndex = dx12.swapChain->GetCurrentBackBufferIndex();

	// Wait only if the GPU has not yet finished with the new frame's allocator
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
}

/**
 * @brief Block the CPU until all commands already submitted to @p queue have
 *        completed on the GPU.
 *
 * Uses the shared dx12.fence and dx12.fenceEvent with a monotonically
 * increasing signal value rather than allocating a new fence and Win32 event
 * on every call.  The old implementation created a fence+event pair for every
 * texture or model upload (hundreds of calls at map load), which is expensive
 * and unnecessary.
 *
 * @p queue must be dx12.commandQueue; the parameter is kept for API
 * compatibility with existing call sites.
 */
void DX12_WaitForUpload(ID3D12CommandQueue *queue)
{
	HRESULT      hr;
	const UINT64 signalValue = ++dx12.nextFenceValue;

	// Require shared fence and event (both are created during R_DX12_Init,
	// before any upload call is made).
	if (!dx12.fence || !dx12.fenceEvent)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_WaitForUpload: fence/event not initialized\n");
		return;
	}

	hr = queue->Signal(dx12.fence, signalValue);
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_WaitForUpload: Signal failed (0x%08lx)\n", hr);
		return;
	}

	if (dx12.fence->GetCompletedValue() < signalValue)
	{
		hr = dx12.fence->SetEventOnCompletion(signalValue, dx12.fenceEvent);
		if (FAILED(hr))
		{
			dx12.ri.Printf(PRINT_WARNING, "DX12_WaitForUpload: SetEventOnCompletion failed (0x%08lx)\n", hr);
			return;
		}
		WaitForSingleObjectEx(dx12.fenceEvent, INFINITE, FALSE);
	}
}

// ---------------------------------------------------------------------------
// DX12_CreateTextureFromRGBA
// ---------------------------------------------------------------------------

/**
 * @brief DX12_CreateTextureFromRGBA
 * @param[in] data     RGBA pixel data (width * height * 4 bytes)
 * @param[in] width    Texture width in pixels
 * @param[in] height   Texture height in pixels
 * @param[in] srvSlot  Index in dx12.srvHeap at which to create the SRV
 * @return             Populated dx12Texture_t; resource is NULL on failure
 *
 * Creates a D3D12 2D texture, uploads the pixel data via an upload heap,
 * transitions the resource to pixel-shader SRV state, and creates an SRV at
 * the specified heap slot.  Uses dx12.uploadCmdAllocator / dx12.uploadCmdList
 * (dedicated upload pipeline, independent of the per-frame rendering list),
 * so it is safe to call while a frame is open.
 */
dx12Texture_t DX12_CreateTextureFromRGBA(const byte *data, int width, int height, int srvSlot)
{
	dx12Texture_t tex;
	HRESULT       hr;

	Com_Memset(&tex, 0, sizeof(tex));

	if (srvSlot < 0 || srvSlot >= DX12_MAX_TEXTURES)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_CreateTextureFromRGBA: invalid srvSlot %d\n", srvSlot);
		return tex;
	}

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
	UINT8       *pMapped  = NULL;
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

	// ---- Record upload + transition into the dedicated upload command list ----
	dx12.uploadCmdAllocator->Reset();
	dx12.uploadCmdList->Reset(dx12.uploadCmdAllocator, NULL);

	D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
	srcLoc.pResource       = uploadBuffer;
	srcLoc.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
	srcLoc.PlacedFootprint = footprint;

	D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
	dstLoc.pResource        = tex.resource;
	dstLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
	dstLoc.SubresourceIndex = 0;

	dx12.uploadCmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, NULL);

	D3D12_RESOURCE_BARRIER barrier = {};
	barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	barrier.Transition.pResource   = tex.resource;
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
	barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	dx12.uploadCmdList->ResourceBarrier(1, &barrier);

	dx12.uploadCmdList->Close();

	ID3D12CommandList *ppCmdLists[] = { dx12.uploadCmdList };
	dx12.commandQueue->ExecuteCommandLists(1, ppCmdLists);

	// Wait for upload to finish before releasing the staging buffer
	DX12_WaitForUpload(dx12.commandQueue);
	uploadBuffer->Release();

	// ---- Create SRV in the requested slot of the SRV heap ----
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Format                        = DXGI_FORMAT_R8G8B8A8_UNORM;
	srvDesc.ViewDimension                 = D3D12_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Shader4ComponentMapping       = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Texture2D.MipLevels           = 1;
	srvDesc.Texture2D.MostDetailedMip     = 0;
	srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;

	tex.cpuHandle.ptr = dx12.srvHeap->GetCPUDescriptorHandleForHeapStart().ptr
	                    + (SIZE_T)srvSlot * dx12.srvDescriptorSize;
	tex.gpuHandle.ptr = dx12.srvHeap->GetGPUDescriptorHandleForHeapStart().ptr
	                    + (UINT64)srvSlot * dx12.srvDescriptorSize;
	dx12.device->CreateShaderResourceView(tex.resource, &srvDesc, tex.cpuHandle);

	return tex;
}

// ---------------------------------------------------------------------------
// DX12_BeginFrame
// ---------------------------------------------------------------------------

/**
 * @brief DX12_BeginFrame
 *
 * Opens the command list for the current back-buffer, issues the
 * PRESENT→RENDER_TARGET barrier, clears the screen, and binds the root
 * signature, PSO, viewport, scissor, and SRV heap.  Must be paired with
 * DX12_EndFrame().
 */
void DX12_BeginFrame(void)
{
	HRESULT                     hr;
	D3D12_RESOURCE_BARRIER      barrierToRT;
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
	const float                 clearColor[] = { 0.05f, 0.05f, 0.15f, 1.0f };
	ID3D12DescriptorHeap        *heaps[1];

	if (!dx12.initialized)
	{
		return;
	}

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

	// Transition: PRESENT → RENDER_TARGET
	Com_Memset(&barrierToRT, 0, sizeof(barrierToRT));
	barrierToRT.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrierToRT.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	barrierToRT.Transition.pResource   = dx12.renderTargets[dx12.frameIndex];
	barrierToRT.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
	barrierToRT.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
	barrierToRT.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	dx12.commandList->ResourceBarrier(1, &barrierToRT);

	rtvHandle.ptr = dx12.rtvHeap->GetCPUDescriptorHandleForHeapStart().ptr
	                + dx12.frameIndex * dx12.rtvDescriptorSize;

	// Bind RTV + DSV (if depth buffer is available for this frame)
	if (dx12.dsvHeap && dx12.depthStencil[dx12.frameIndex])
	{
		D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;

		dsvHandle.ptr = dx12.dsvHeap->GetCPUDescriptorHandleForHeapStart().ptr
		                + dx12.frameIndex * dx12.dsvDescriptorSize;
		dx12.commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
		dx12.commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, NULL);
		dx12.commandList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, NULL);
	}
	else
	{
		dx12.commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, NULL);
		dx12.commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, NULL);
	}

	// Bind pipeline state
	dx12.commandList->SetGraphicsRootSignature(dx12.rootSignature);
	dx12.commandList->SetPipelineState(dx12.pipelineState);
	dx12.commandList->RSSetViewports(1, &dx12.viewport);
	dx12.commandList->RSSetScissorRects(1, &dx12.scissorRect);

	// Bind SRV heap once for the entire frame
	heaps[0] = dx12.srvHeap;
	dx12.commandList->SetDescriptorHeaps(1, heaps);

	// Reset the 2D vertex ring-buffer and batch state for this frame
	dx12.quadVBOffset = 0;

	// Reset scissor to full-screen (use the physical swap-chain rect set at init,
	// not dx12.vidWidth/vidHeight which may have been overwritten with the game's
	// virtual resolution by RE_DX12_BeginRegistration).
	dx12.currentScissor = dx12.scissorRect;

	// Reset the 2D batch state for the new frame; slot 0 of the SRV heap is
	// always the white fallback texture, so it is safe to use as the default.
	dx12.batch2DCount    = 0;
	dx12.batch2DStart    = 0;
	dx12.batch2DScissor  = dx12.currentScissor;
	dx12.batch2DTopology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	if (dx12.srvHeap)
	{
		dx12.batch2DTexHandle = dx12.srvHeap->GetGPUDescriptorHandleForHeapStart();
	}

	dx12.frameOpen = qtrue;
}

void DX12_InitSwapchain(void)
{
	HRESULT hr;

	// Get window handle and size
	void *hwndVoid = dx12.ri.GetHWND();
	HWND hwnd      = ( HWND )hwndVoid;
	if (!hwnd)
	{
		dx12.ri.Error(ERR_FATAL, "DX12_InitSwapchain: GetHWND returned NULL\n");
	}

	RECT rc;
	GetClientRect(hwnd, &rc);
	UINT width  = rc.right - rc.left;
	UINT height = rc.bottom - rc.top;
	if (width == 0 || height == 0)
	{
		width  = 640;
		height = 480;
	}

	dx12.hWnd      = hwnd;
	dx12.vidWidth  = ( int )width;
	dx12.vidHeight = ( int )height;

	// Release old swapchain/render targets/depth buffers if any
	for ( int i = 0; i < DX12_FRAME_COUNT; i++ )
	{
		if (dx12.depthStencil[i])
		{
			dx12.depthStencil[i]->Release();
			dx12.depthStencil[i] = nullptr;
		}
		if (dx12.renderTargets[i])
		{
			dx12.renderTargets[i]->Release();
			dx12.renderTargets[i] = nullptr;
		}
	}
	if (dx12.swapChain)
	{
		dx12.swapChain->Release();
		dx12.swapChain = nullptr;
	}

	// Create a local factory (you don't store one in dx12Globals_t)
	IDXGIFactory4 *factory = nullptr;
	hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
	if (FAILED(hr))
	{
		dx12.ri.Error(ERR_FATAL, "DX12_InitSwapchain: CreateDXGIFactory1 failed (0x%08lx)\n", hr);
	}

	// Describe swapchain
	DXGI_SWAP_CHAIN_DESC1 scDesc = {};
	scDesc.Width            = width;
	scDesc.Height           = height;
	scDesc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
	scDesc.SampleDesc.Count = 1;
	scDesc.BufferUsage      = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	scDesc.BufferCount      = DX12_FRAME_COUNT;
	scDesc.SwapEffect       = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	scDesc.Scaling          = DXGI_SCALING_STRETCH;
	scDesc.AlphaMode        = DXGI_ALPHA_MODE_IGNORE;

	IDXGISwapChain1 *swapChain1 = nullptr;
	hr = factory->CreateSwapChainForHwnd(
		dx12.commandQueue,
		hwnd,
		&scDesc,
		nullptr,
		nullptr,
		&swapChain1);
	factory->Release();

	if (FAILED(hr))
	{
		dx12.ri.Error(ERR_FATAL, "DX12_InitSwapchain: CreateSwapChainForHwnd failed (0x%08lx)\n", hr);
	}

	hr = swapChain1->QueryInterface(IID_PPV_ARGS(&dx12.swapChain));
	swapChain1->Release();
	if (FAILED(hr))
	{
		dx12.ri.Error(ERR_FATAL, "DX12_InitSwapchain: QueryInterface IDXGISwapChain3 failed (0x%08lx)\n", hr);
	}

	dx12.frameIndex = dx12.swapChain->GetCurrentBackBufferIndex();

	// Create RTVs for each backbuffer
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle =
		dx12.rtvHeap->GetCPUDescriptorHandleForHeapStart();
	for ( UINT i = 0; i < DX12_FRAME_COUNT; i++ )
	{
		hr = dx12.swapChain->GetBuffer(i, IID_PPV_ARGS(&dx12.renderTargets[i]));
		if (FAILED(hr))
		{
			dx12.ri.Error(ERR_FATAL, "DX12_InitSwapchain: GetBuffer[%u] failed (0x%08lx)\n", i, hr);
		}

		dx12.device->CreateRenderTargetView(dx12.renderTargets[i], nullptr, rtvHandle);
		rtvHandle.ptr += dx12.rtvDescriptorSize;
	}

	// Set viewport + scissor
	dx12.viewport.TopLeftX = 0.0f;
	dx12.viewport.TopLeftY = 0.0f;
	dx12.viewport.Width    = ( float )width;
	dx12.viewport.Height   = ( float )height;
	dx12.viewport.MinDepth = 0.0f;
	dx12.viewport.MaxDepth = 1.0f;

	dx12.scissorRect.left   = 0;
	dx12.scissorRect.top    = 0;
	dx12.scissorRect.right  = ( LONG )width;
	dx12.scissorRect.bottom = ( LONG )height;

	dx12.currentScissor = dx12.scissorRect;

	// Create per-frame depth stencil textures (D32_FLOAT) if DSV heap is ready
	if (dx12.dsvHeap)
	{
		D3D12_HEAP_PROPERTIES dsHeapProps = {};
		dsHeapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

		D3D12_RESOURCE_DESC dsDesc = {};
		dsDesc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		dsDesc.Width            = width;
		dsDesc.Height           = height;
		dsDesc.DepthOrArraySize = 1;
		dsDesc.MipLevels        = 1;
		dsDesc.Format           = DXGI_FORMAT_D32_FLOAT;
		dsDesc.SampleDesc.Count = 1;
		dsDesc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		dsDesc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

		D3D12_CLEAR_VALUE dsClear = {};
		dsClear.Format               = DXGI_FORMAT_D32_FLOAT;
		dsClear.DepthStencil.Depth   = 1.0f;
		dsClear.DepthStencil.Stencil = 0;

		D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
		dsvDesc.Format             = DXGI_FORMAT_D32_FLOAT;
		dsvDesc.ViewDimension      = D3D12_DSV_DIMENSION_TEXTURE2D;
		dsvDesc.Flags              = D3D12_DSV_FLAG_NONE;
		dsvDesc.Texture2D.MipSlice = 0;

		D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dx12.dsvHeap->GetCPUDescriptorHandleForHeapStart();

		for ( UINT i = 0; i < DX12_FRAME_COUNT; i++ )
		{
			hr = dx12.device->CreateCommittedResource(
				&dsHeapProps,
				D3D12_HEAP_FLAG_NONE,
				&dsDesc,
				D3D12_RESOURCE_STATE_DEPTH_WRITE,
				&dsClear,
				IID_PPV_ARGS(&dx12.depthStencil[i]));

			if (FAILED(hr))
			{
				dx12.ri.Error(ERR_FATAL,
				              "DX12_InitSwapchain: CreateCommittedResource (depth[%u]) failed (0x%08lx)\n",
				              i, hr);
			}

			dx12.device->CreateDepthStencilView(dx12.depthStencil[i], &dsvDesc, dsvHandle);
			dsvHandle.ptr += dx12.dsvDescriptorSize;
		}
	}
}

/**
 * @brief R_DX12_Init
 *
 * Creates the DXGI factory, D3D12 device, command queue, swap chain, RTV heap,
 * SRV heap, render targets, command allocators, command list, fence, root
 * signature, PSO, and 2D vertex ring-buffer.  Ends by calling DX12_InitTextures.
 */
qboolean R_DX12_Init(void)
{
	HRESULT    hr;
	char       glConfigString[1024];
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
	// DSV Descriptor Heap (one slot per swap-chain buffer)
	// ----------------------------------------------------------------
	{
		D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
		dsvHeapDesc.NumDescriptors = DX12_FRAME_COUNT;
		dsvHeapDesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
		dsvHeapDesc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

		hr = dx12.device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&dx12.dsvHeap));
		if (FAILED(hr))
		{
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateDescriptorHeap (DSV) failed (0x%08lx)\n", hr);
			return qfalse;
		}

		dx12.dsvDescriptorSize = dx12.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
	}

	// ----------------------------------------------------------------
	// Swap Chain
	// ----------------------------------------------------------------

	DX12_InitSwapchain();

	factory->Release();

	// ----------------------------------------------------------------
	// SRV Descriptor Heap (shader-visible, DX12_MAX_TEXTURES slots)
	// ----------------------------------------------------------------
	{
		D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
		srvHeapDesc.NumDescriptors = DX12_MAX_TEXTURES;
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
	// Dedicated upload command allocator + list
	// Used exclusively by WLD_UploadBuffer / MDL_UploadBuffer so that
	// resource uploads never reset the per-frame rendering allocator/list.
	// ----------------------------------------------------------------
	hr = dx12.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
	                                         IID_PPV_ARGS(&dx12.uploadCmdAllocator));
	if (FAILED(hr))
	{
		dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateCommandAllocator (upload) failed (0x%08lx)\n", hr);
		return qfalse;
	}

	hr = dx12.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
	                                    dx12.uploadCmdAllocator, NULL,
	                                    IID_PPV_ARGS(&dx12.uploadCmdList));
	if (FAILED(hr))
	{
		dx12.ri.Error(ERR_FATAL, "R_DX12_Init: CreateCommandList (upload) failed (0x%08lx)\n", hr);
		return qfalse;
	}

	// Command lists are created in the recording state; close immediately so
	// the first UploadBuffer call can Reset + re-open it.
	dx12.uploadCmdList->Close();

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
		UINT     compileFlags  = 0;

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
		if (errorBlob)
		{
			errorBlob->Release(); errorBlob = NULL;
		}

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
		if (errorBlob)
		{
			errorBlob->Release(); errorBlob = NULL;
		}

		// Input layout matching dx12QuadVertex_t: float2 pos + float2 uv + float4 color
		D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 0,  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 8,  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		};

		D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
		psoDesc.InputLayout    = { inputElementDescs, 3 };
		psoDesc.pRootSignature = dx12.rootSignature;
		psoDesc.VS             = { vertexShader->GetBufferPointer(), vertexShader->GetBufferSize() };
		psoDesc.PS             = { pixelShader->GetBufferPointer(), pixelShader->GetBufferSize() };
		// Rasterizer: no back-face culling (correct for 2D TRIANGLESTRIP quads)
		{
			D3D12_RASTERIZER_DESC &rs = psoDesc.RasterizerState;
			rs.FillMode              = D3D12_FILL_MODE_SOLID;
			rs.CullMode              = D3D12_CULL_MODE_NONE;
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

		// Alpha blending: SRC_ALPHA × src + (1 − SRC_ALPHA) × dst
		{
			D3D12_BLEND_DESC &bd = psoDesc.BlendState;
			bd.AlphaToCoverageEnable  = FALSE;
			bd.IndependentBlendEnable = FALSE;
			{
				D3D12_RENDER_TARGET_BLEND_DESC &rt = bd.RenderTarget[0];
				rt.BlendEnable           = TRUE;
				rt.LogicOpEnable         = FALSE;
				rt.SrcBlend              = D3D12_BLEND_SRC_ALPHA;
				rt.DestBlend             = D3D12_BLEND_INV_SRC_ALPHA;
				rt.BlendOp               = D3D12_BLEND_OP_ADD;
				rt.SrcBlendAlpha         = D3D12_BLEND_ONE;
				rt.DestBlendAlpha        = D3D12_BLEND_INV_SRC_ALPHA;
				rt.BlendOpAlpha          = D3D12_BLEND_OP_ADD;
				rt.LogicOp               = D3D12_LOGIC_OP_NOOP;
				rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
			}
		}
		psoDesc.DepthStencilState.DepthEnable   = FALSE;
		psoDesc.DepthStencilState.StencilEnable = FALSE;
		psoDesc.SampleMask                      = UINT_MAX;
		psoDesc.PrimitiveTopologyType           = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		psoDesc.NumRenderTargets                = 1;
		psoDesc.RTVFormats[0]                   = DXGI_FORMAT_R8G8B8A8_UNORM;
		psoDesc.DSVFormat                       = DXGI_FORMAT_D32_FLOAT;
		psoDesc.SampleDesc.Count                = 1;

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
	// 2D Vertex Ring-Buffer (upload heap, persistently mapped)
	// ----------------------------------------------------------------
	{
		const UINT quadVBSize = (UINT)(sizeof(dx12QuadVertex_t) * DX12_MAX_2D_VERTS);

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

		// Persistently map the upload heap – stays mapped for the lifetime of the buffer
		D3D12_RANGE readRange = { 0, 0 };
		hr = dx12.quadVertexBuffer->Map(0, &readRange, (void **)&dx12.quadVBMapped);
		if (FAILED(hr))
		{
			dx12.ri.Error(ERR_FATAL, "R_DX12_Init: quad VB Map failed (0x%08lx)\n", hr);
			return qfalse;
		}

		dx12.quadVBOffset = 0;
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
	dx12.nextFenceValue = 0;
	dx12.fenceValues[0] = 0;
	dx12.fenceValues[1] = 0;

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
	// Per-frame defaults
	// ----------------------------------------------------------------
	dx12.color2D[0] = 1.0f;
	dx12.color2D[1] = 1.0f;
	dx12.color2D[2] = 1.0f;
	dx12.color2D[3] = 1.0f;
	dx12.frameOpen  = qfalse;

	// ----------------------------------------------------------------
	// Texture registry (white fallback at slot 0)
	// ----------------------------------------------------------------
	DX12_InitTextures();

	dx12.initialized = qtrue;
	dx12.ri.Printf(PRINT_ALL, "R_DX12: Initialized (%dx%d)\n", dx12.vidWidth, dx12.vidHeight);

	// Initialize the 3D scene pipeline after core DX12 objects are ready
	DX12_SceneInit();

	return qtrue;
}

// ---------------------------------------------------------------------------
// DX12_EndFrame – close frame and present
// ---------------------------------------------------------------------------

/**
 * @brief DX12_EndFrame
 *
 * Transitions the back buffer from RENDER_TARGET to PRESENT, closes and
 * executes the command list, presents the swap chain, and advances to the
 * next frame.  Must be paired with DX12_BeginFrame().
 */
void DX12_EndFrame(void)
{
	HRESULT                hr;
	D3D12_RESOURCE_BARRIER barrierToPresent;
	ID3D12CommandList      *ppCommandLists[1];

	if (!dx12.initialized || !dx12.frameOpen)
	{
		return;
	}

	// Transition: RENDER_TARGET → PRESENT
	Com_Memset(&barrierToPresent, 0, sizeof(barrierToPresent));
	barrierToPresent.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrierToPresent.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	barrierToPresent.Transition.pResource   = dx12.renderTargets[dx12.frameIndex];
	barrierToPresent.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
	barrierToPresent.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
	barrierToPresent.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	dx12.commandList->ResourceBarrier(1, &barrierToPresent);

	hr = dx12.commandList->Close();
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING, "R_DX12: commandList->Close failed (0x%08lx)\n", hr);
		dx12.frameOpen = qfalse;
		return;
	}

	ppCommandLists[0] = dx12.commandList;
	dx12.commandQueue->ExecuteCommandLists(1, ppCommandLists);

	hr = dx12.swapChain->Present(0, 0);
	if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET)
	{
		// Device was lost – log the reason and mark renderer as uninitialized
		HRESULT reason = dx12.device ? dx12.device->GetDeviceRemovedReason() : hr;
		dx12.ri.Printf(PRINT_WARNING,
		               "R_DX12: device removed during Present (reason 0x%08lx)\n", reason);
		dx12.initialized = qfalse;
	}
	else if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING, "R_DX12: Present failed (0x%08lx)\n", hr);
	}

	dx12.frameOpen = qfalse;
	DX12_MoveToNextFrame();
}

// ---------------------------------------------------------------------------
// R_DX12_RenderCommandList – command dispatcher
// ---------------------------------------------------------------------------

/**
 * @brief R_DX12_RenderCommandList
 * @param[in] data  Render command list buffer
 *
 * Walks the ET:Legacy render command list. Calls DX12_EndFrame on
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
			DX12_EndFrame();
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
 * Two-mode shutdown:
 *
 *   destroyWindow == qfalse (soft reset, called between map loads):
 *     Waits for the GPU, releases all per-map content resources (world,
 *     models, scene, scratch textures, textures) and resets all CPU-side
 *     registries via DX12_ClearPerSessionState().  The D3D12 device,
 *     command queues, descriptor heaps, PSO and swap chain are kept alive
 *     so that re.purgeCache() and the next re.BeginRegistration() can run
 *     without a costly full device recreation.  dx12.initialized remains
 *     qtrue after a soft reset.
 *
 *   destroyWindow == qtrue (hard shutdown, called on vid_restart / quit):
 *     Performs the soft reset first, then additionally releases all device-
 *     level D3D12 infrastructure, sets dx12.initialized = qfalse and
 *     destroys the SDL window.
 */
void R_DX12_Shutdown(qboolean destroyWindow)
{
	dx12.ri.Printf(PRINT_ALL, "---- R_DX12_Shutdown ----\n");

	if (!dx12.initialized)
	{
		return;
	}

	// ---- 1.  Wait for the GPU to drain all outstanding work ----------------
	DX12_WaitForGpu();

	// ---- 2.  Release per-map content (world geometry, 3D scene pipeline,
	//          model GPU buffers, scratch textures, material/texture cache) --
	DX12_SceneShutdown();
	DX12_ShutdownModels();
	DX12_ShutdownWorld();
	DX12_ShutdownScratchTextures();
	DX12_ShutdownTextures();

	// Reset the per-session CPU-side registries (model names, skins).
	// DX12_ShutdownTextures already zeroed dx12NumShaders / dx12NumMaterials.
	DX12_ClearPerSessionState();

	// ---- Soft reset: keep D3D12 device and infrastructure alive -----------
	// re.purgeCache() will be called immediately afterwards by CL_ShutdownAll
	// while dx12.initialized is still qtrue, allowing it to reinitialise the
	// fallback textures.  The next re.BeginRegistration() then uses the else
	// branch (device already alive) to reinit textures and the 3D scene
	// pipeline rather than recreating the entire device.
	if (!destroyWindow)
	{
		return;
	}

	// ---- Hard shutdown: release all D3D12 infrastructure ------------------

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

	if (dx12.quadVertexBuffer)
	{
		// Buffer was persistently mapped – Unmap before release
		dx12.quadVertexBuffer->Unmap(0, NULL);
		dx12.quadVBMapped = NULL;
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

	if (dx12.uploadCmdList)
	{
		dx12.uploadCmdList->Release();
		dx12.uploadCmdList = NULL;
	}

	if (dx12.uploadCmdAllocator)
	{
		dx12.uploadCmdAllocator->Release();
		dx12.uploadCmdAllocator = NULL;
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

	for (UINT i = 0; i < DX12_FRAME_COUNT; i++)
	{
		if (dx12.depthStencil[i])
		{
			dx12.depthStencil[i]->Release();
			dx12.depthStencil[i] = NULL;
		}
	}

	if (dx12.dsvHeap)
	{
		dx12.dsvHeap->Release();
		dx12.dsvHeap = NULL;
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

	dx12.ri.GLimp_Shutdown();
}

#endif // _WIN32
