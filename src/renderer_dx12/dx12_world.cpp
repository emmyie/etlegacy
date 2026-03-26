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
 * @file dx12_world.cpp
 * @brief DX12 BSP world geometry loading.
 *
 * Implements DX12_LoadWorld() which:
 *   1. Reads the BSP file and validates the header.
 *   2. Registers all BSP shader entries as DX12 materials.
 *   3. Uploads all BSP lightmaps as 128×128 RGBA textures.
 *   4. Parses fog volumes (LUMP_FOGS) into dx12WorldFog_t.
 *   5. Flattens all draw-vertices (LUMP_DRAWVERTS) and draw-indices
 *      (LUMP_DRAWINDEXES) into CPU arrays and uploads them to GPU
 *      DEFAULT-heap vertex / index buffers.
 *   6. Builds a dx12DrawSurf_t list from LUMP_SURFACES, skipping nodraw
 *      and flare surfaces.
 *   7. Sorts the draw-surface list for efficient rendering.
 *   8. Parses LUMP_MODELS into dx12WorldModel_t submodel descriptors.
 *
 * @note Patch (MST_PATCH) surfaces are tessellated into triangle lists
 *       using simple uniform tessellation at a fixed LOD level.
 */

#include "dx12_world.h"
#include "dx12_shader.h"
#include "dx12_image.h"

#ifdef _WIN32

#include "../qcommon/qfiles.h"

#include <string.h>  // memcpy, memset
#include <stdlib.h>  // malloc, free, qsort
#include <math.h>    // sqrtf for patch tessellation

// ---------------------------------------------------------------------------
// Global world state
// ---------------------------------------------------------------------------

dx12World_t dx12World;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * @brief Upload a CPU buffer to a GPU DEFAULT-heap resource via a temporary
 *        UPLOAD-heap intermediary.
 *
 * Opens and closes the command list internally, then waits for the GPU to
 * finish before releasing the upload buffer.  Must be called outside of an
 * active frame (i.e. between DX12_EndFrame and DX12_BeginFrame).
 *
 * @param data      Pointer to the CPU-side data to upload.
 * @param sizeBytes Size of the data in bytes.
 * @param outBuffer Receives the newly created DEFAULT-heap resource on success.
 * @return qtrue on success; qfalse if any D3D12 call fails.
 */
static qboolean WLD_UploadBuffer(const void *data, UINT64 sizeBytes,
                                  ID3D12Resource **outBuffer)
{
	HRESULT hr;

	D3D12_HEAP_PROPERTIES defaultHeap = {};
	defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

	D3D12_RESOURCE_DESC bufDesc = {};
	bufDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
	bufDesc.Width            = sizeBytes;
	bufDesc.Height           = 1;
	bufDesc.DepthOrArraySize = 1;
	bufDesc.MipLevels        = 1;
	bufDesc.Format           = DXGI_FORMAT_UNKNOWN;
	bufDesc.SampleDesc.Count = 1;
	bufDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

	hr = dx12.device->CreateCommittedResource(
		&defaultHeap,
		D3D12_HEAP_FLAG_NONE,
		&bufDesc,
		D3D12_RESOURCE_STATE_COPY_DEST,
		NULL,
		IID_PPV_ARGS(outBuffer));

	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "WLD_UploadBuffer: CreateCommittedResource (default) failed (0x%08lx)\n", hr);
		return qfalse;
	}

	// Create upload heap
	D3D12_HEAP_PROPERTIES uploadHeap = {};
	uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;

	ID3D12Resource *uploadBuf = NULL;

	hr = dx12.device->CreateCommittedResource(
		&uploadHeap,
		D3D12_HEAP_FLAG_NONE,
		&bufDesc,
		D3D12_RESOURCE_STATE_GENERIC_READ,
		NULL,
		IID_PPV_ARGS(&uploadBuf));

	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "WLD_UploadBuffer: CreateCommittedResource (upload) failed (0x%08lx)\n", hr);
		(*outBuffer)->Release();
		*outBuffer = NULL;
		return qfalse;
	}

	// Copy data into the upload heap
	{
		D3D12_RANGE readRange = { 0, 0 };
		void       *mapped    = NULL;

		hr = uploadBuf->Map(0, &readRange, &mapped);
		if (FAILED(hr))
		{
			dx12.ri.Printf(PRINT_WARNING,
			               "WLD_UploadBuffer: Map failed (0x%08lx)\n", hr);
			uploadBuf->Release();
			(*outBuffer)->Release();
			*outBuffer = NULL;
			return qfalse;
		}

		memcpy(mapped, data, (size_t)sizeBytes);
		uploadBuf->Unmap(0, NULL);
	}

	// Reset command allocator + list for this transfer
	hr = dx12.commandAllocators[dx12.frameIndex]->Reset();
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "WLD_UploadBuffer: command allocator Reset failed (0x%08lx)\n", hr);
		uploadBuf->Release();
		(*outBuffer)->Release();
		*outBuffer = NULL;
		return qfalse;
	}

	hr = dx12.commandList->Reset(dx12.commandAllocators[dx12.frameIndex], NULL);
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "WLD_UploadBuffer: command list Reset failed (0x%08lx)\n", hr);
		uploadBuf->Release();
		(*outBuffer)->Release();
		*outBuffer = NULL;
		return qfalse;
	}

	// Record the copy
	dx12.commandList->CopyBufferRegion(*outBuffer, 0, uploadBuf, 0, sizeBytes);

	// Transition the default buffer to the generic-read state
	D3D12_RESOURCE_BARRIER barrier = {};
	barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Transition.pResource   = *outBuffer;
	barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_GENERIC_READ;
	dx12.commandList->ResourceBarrier(1, &barrier);

	hr = dx12.commandList->Close();
	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "WLD_UploadBuffer: command list Close failed (0x%08lx)\n", hr);
		uploadBuf->Release();
		(*outBuffer)->Release();
		*outBuffer = NULL;
		return qfalse;
	}

	// Execute and wait
	ID3D12CommandList *lists[] = { dx12.commandList };
	dx12.commandQueue->ExecuteCommandLists(1, lists);

	dx12.fenceValues[dx12.frameIndex]++;
	dx12.commandQueue->Signal(dx12.fence, dx12.fenceValues[dx12.frameIndex]);
	dx12.fence->SetEventOnCompletion(dx12.fenceValues[dx12.frameIndex], dx12.fenceEvent);
	WaitForSingleObjectEx(dx12.fenceEvent, INFINITE, FALSE);

	uploadBuf->Release();
	return qtrue;
}

// ---------------------------------------------------------------------------
// Patch tessellation helper
// ---------------------------------------------------------------------------

/** Grid LOD used for all BSP patches.  4 = 9 subdivisions per row/column. */
#define PATCH_LOD 4

/**
 * @brief Bilinearly interpolate one dx12WorldVertex_t across a 3×3 control grid.
 *
 * @param[in]  grid    Flat 3×3 array of dx12WorldVertex_t control points.
 * @param[in]  s       Horizontal parameter in [0, 1].
 * @param[in]  t       Vertical parameter in [0, 1].
 * @param[out] out     Interpolated vertex.
 */
static void WLD_BezierEval(const dx12WorldVertex_t grid[9],
                            float s, float t,
                            dx12WorldVertex_t *out)
{
	// Quadratic Bernstein basis weights
	float bs[3], bt[3];
	int   a, b, k;

	bs[0] = (1.0f - s) * (1.0f - s);
	bs[1] = 2.0f * s * (1.0f - s);
	bs[2] = s * s;

	bt[0] = (1.0f - t) * (1.0f - t);
	bt[1] = 2.0f * t * (1.0f - t);
	bt[2] = t * t;

	memset(out, 0, sizeof(*out));

	for (b = 0; b < 3; b++)
	{
		for (a = 0; a < 3; a++)
		{
			float w               = bs[a] * bt[b];
			const dx12WorldVertex_t *cp = &grid[b * 3 + a];

			for (k = 0; k < 3; k++)
			{
				out->xyz[k]    += w * cp->xyz[k];
				out->normal[k] += w * cp->normal[k];
			}
			for (k = 0; k < 2; k++)
			{
				out->st[k] += w * cp->st[k];
				out->lm[k] += w * cp->lm[k];
			}
			for (k = 0; k < 4; k++)
			{
				out->color[k] += w * cp->color[k];
			}
		}
	}

	// Renormalize the interpolated normal
	{
		float len = sqrtf(out->normal[0] * out->normal[0]
		                  + out->normal[1] * out->normal[1]
		                  + out->normal[2] * out->normal[2]);

		if (len > 0.0001f)
		{
			out->normal[0] /= len;
			out->normal[1] /= len;
			out->normal[2] /= len;
		}
	}
}

// ---------------------------------------------------------------------------
// Draw surface comparison (for qsort)
// ---------------------------------------------------------------------------

/**
 * @brief Sort key for draw surfaces.
 *
 * Order: opaque < fog < sky < translucent; then ascending materialHandle.
 */
static int WLD_SurfCategory(const dx12DrawSurf_t *s)
{
	if (s->isTranslucent)
	{
		return 3;
	}
	if (s->isSky)
	{
		return 2;
	}
	if (s->isFog)
	{
		return 1;
	}
	return 0;
}

static int WLD_SortSurfs(const void *a, const void *b)
{
	const dx12DrawSurf_t *sa = (const dx12DrawSurf_t *)a;
	const dx12DrawSurf_t *sb = (const dx12DrawSurf_t *)b;
	int                   ca = WLD_SurfCategory(sa);
	int                   cb = WLD_SurfCategory(sb);

	if (ca != cb)
	{
		return ca - cb;
	}

	if (sa->materialHandle != sb->materialHandle)
	{
		return (sa->materialHandle < sb->materialHandle) ? -1 : 1;
	}

	return sa->fogIndex - sb->fogIndex;
}

// ---------------------------------------------------------------------------
// DX12_ShutdownWorld
// ---------------------------------------------------------------------------

/**
 * @brief DX12_ShutdownWorld
 *
 * Releases all GPU resources associated with the currently loaded world and
 * resets dx12World to zero.  Safe to call when no world is loaded.
 */
void DX12_ShutdownWorld(void)
{
	if (!dx12World.loaded)
	{
		return;
	}

	if (dx12World.vertexBuffer)
	{
		dx12World.vertexBuffer->Release();
		dx12World.vertexBuffer = NULL;
	}

	if (dx12World.indexBuffer)
	{
		dx12World.indexBuffer->Release();
		dx12World.indexBuffer = NULL;
	}

	Com_Memset(&dx12World, 0, sizeof(dx12World));
}

// ---------------------------------------------------------------------------
// DX12_LoadWorld
// ---------------------------------------------------------------------------

/**
 * @brief DX12_LoadWorld
 * @param[in] name  BSP file path.
 *
 * Loads all world geometry from the BSP file and uploads it to the GPU.
 */
void DX12_LoadWorld(const char *name)
{
	void      *buffer    = NULL;
	int        bufLen    = 0;
	dheader_t *header    = NULL;
	byte      *fileBase  = NULL;
	int        i;

	if (!name || !name[0])
	{
		return;
	}

	// Release any previously loaded world
	DX12_ShutdownWorld();

	bufLen = dx12.ri.FS_ReadFile(name, &buffer);
	if (bufLen <= 0 || !buffer)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_LoadWorld: '%s' not found\n", name);
		return;
	}

	header   = (dheader_t *)buffer;
	fileBase = (byte *)header;

	// Validate magic + version
	if (LittleLong(header->ident) != BSP_IDENT)
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "DX12_LoadWorld: '%s' is not a BSP file\n", name);
		dx12.ri.FS_FreeFile(buffer);
		return;
	}

	if (LittleLong(header->version) != BSP_VERSION)
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "DX12_LoadWorld: '%s' has wrong BSP version (%d, expected %d)\n",
		               name, LittleLong(header->version), BSP_VERSION);
		dx12.ri.FS_FreeFile(buffer);
		return;
	}

	// Byte-swap the header lump table
	for (i = 0; i < (int)(sizeof(dheader_t) / 4); i++)
	{
		((int *)header)[i] = LittleLong(((int *)header)[i]);
	}

	Q_strncpyz(dx12World.name, name, sizeof(dx12World.name));

	// ----------------------------------------------------------------
	// 1.  BSP shader lump → register DX12 materials
	// ----------------------------------------------------------------
	{
		lump_t    *lump = &header->lumps[LUMP_SHADERS];
		dshader_t *in   = (dshader_t *)(fileBase + lump->fileofs);
		int        count;

		if (lump->filelen % (int)sizeof(dshader_t))
		{
			dx12.ri.Printf(PRINT_WARNING, "DX12_LoadWorld: bad LUMP_SHADERS size\n");
			dx12.ri.FS_FreeFile(buffer);
			return;
		}

		count                   = (int)(lump->filelen / sizeof(dshader_t));
		dx12World.numShaders    = count < DX12_MAX_WORLD_SHADERS ? count : DX12_MAX_WORLD_SHADERS;

		for (i = 0; i < dx12World.numShaders; i++)
		{
			dx12World.shaderHandles[i] = DX12_RegisterMaterial(in[i].shader);
		}

		dx12.ri.Printf(PRINT_DEVELOPER, "DX12_LoadWorld: registered %d BSP shaders\n",
		               dx12World.numShaders);
	}

	// ----------------------------------------------------------------
	// 2.  Lightmap lump → upload 128×128 RGB textures
	// ----------------------------------------------------------------
	{
		lump_t      *lump = &header->lumps[LUMP_LIGHTMAPS];
		const byte  *src  = fileBase + lump->fileofs;
		int          lmSize = LIGHTMAP_WIDTH * LIGHTMAP_HEIGHT * 3; // RGB on-disk
		int          count;

		count                    = (lump->filelen > 0) ? (lump->filelen / lmSize) : 0;
		dx12World.numLightmaps   = count < DX12_MAX_LIGHTMAPS ? count : DX12_MAX_LIGHTMAPS;

		for (i = 0; i < dx12World.numLightmaps; i++)
		{
			// Expand 3-channel (RGB) to 4-channel (RGBA) for the DX12 RGBA texture format
			byte rgba[LIGHTMAP_WIDTH * LIGHTMAP_HEIGHT * 4];
			int  px;

			for (px = 0; px < LIGHTMAP_WIDTH * LIGHTMAP_HEIGHT; px++)
			{
				rgba[px * 4 + 0] = src[i * lmSize + px * 3 + 0];
				rgba[px * 4 + 1] = src[i * lmSize + px * 3 + 1];
				rgba[px * 4 + 2] = src[i * lmSize + px * 3 + 2];
				rgba[px * 4 + 3] = 255;
			}

			{
				char        lmName[MAX_QPATH];
				int         slot = dx12NumShaders;
				dx12Texture_t tex;

				// Guard against overflow – lightmap SRV slots live in the texture registry
				if (slot >= DX12_MAX_TEXTURES)
				{
					dx12.ri.Printf(PRINT_WARNING,
					               "DX12_LoadWorld: texture registry full, skipping lightmap %d\n", i);
					dx12World.lightmapHandles[i] = 0;
					continue;
				}

				snprintf(lmName, sizeof(lmName), "*lightmap%d", i);
				tex = DX12_CreateTextureFromRGBA(rgba, LIGHTMAP_WIDTH, LIGHTMAP_HEIGHT, slot);

				if (tex.resource)
				{
					Q_strncpyz(dx12Shaders[slot].name, lmName, sizeof(dx12Shaders[slot].name));
					dx12Shaders[slot].width  = LIGHTMAP_WIDTH;
					dx12Shaders[slot].height = LIGHTMAP_HEIGHT;
					dx12Shaders[slot].tex    = tex;
					dx12Shaders[slot].valid  = qtrue;
					dx12NumShaders++;
					dx12World.lightmapHandles[i] = (qhandle_t)slot;
				}
				else
				{
					dx12World.lightmapHandles[i] = 0;
				}
			}
		}

		dx12.ri.Printf(PRINT_DEVELOPER, "DX12_LoadWorld: uploaded %d lightmaps\n",
		               dx12World.numLightmaps);
	}

	// ----------------------------------------------------------------
	// 3.  Fog lump → dx12WorldFog_t
	// ----------------------------------------------------------------
	{
		lump_t   *lump     = &header->lumps[LUMP_FOGS];
		dfog_t   *fogIn    = (dfog_t *)(fileBase + lump->fileofs);
		int       fogCount = 0;

		if (lump->filelen > 0)
		{
			if (lump->filelen % (int)sizeof(dfog_t))
			{
				dx12.ri.Printf(PRINT_WARNING, "DX12_LoadWorld: bad LUMP_FOGS size\n");
			}
			else
			{
				fogCount = (int)(lump->filelen / sizeof(dfog_t));
			}
		}

		// Slot 0: "no fog" sentinel
		Com_Memset(&dx12World.fogs[0], 0, sizeof(dx12WorldFog_t));
		dx12World.fogs[0].mins[0] = dx12World.fogs[0].mins[1] = dx12World.fogs[0].mins[2] =  1e30f;
		dx12World.fogs[0].maxs[0] = dx12World.fogs[0].maxs[1] = dx12World.fogs[0].maxs[2] = -1e30f;
		dx12World.fogs[0].originalBrush = -2;

		if (fogCount > DX12_MAX_WORLD_FOGS - 1)
		{
			fogCount = DX12_MAX_WORLD_FOGS - 1;
		}

		dx12World.numFogs = fogCount + 1; // +1 for the sentinel at index 0

		for (i = 0; i < fogCount; i++)
		{
			dx12WorldFog_t *out = &dx12World.fogs[i + 1];
			int             brushNum;

			Com_Memset(out, 0, sizeof(*out));

			brushNum            = LittleLong(fogIn[i].brushNum);
			out->originalBrush  = brushNum;
			out->materialHandle = DX12_RegisterMaterial(fogIn[i].shader);

			if (brushNum == -1)
			{
				// Global fog – covers the whole world
				out->mins[0] = out->mins[1] = out->mins[2] = -1e30f;
				out->maxs[0] = out->maxs[1] = out->maxs[2] =  1e30f;
			}
			else
			{
				// Per-brush fog – resolve AABB from brush planes
				lump_t       *brushLump  = &header->lumps[LUMP_BRUSHES];
				lump_t       *sideLump   = &header->lumps[LUMP_BRUSHSIDES];
				lump_t       *planeLump  = &header->lumps[LUMP_PLANES];
				dbrush_t     *brushes    = (dbrush_t *)(fileBase + brushLump->fileofs);
				dbrushside_t *sides      = (dbrushside_t *)(fileBase + sideLump->fileofs);
				dplane_t     *planes     = (dplane_t *)(fileBase + planeLump->fileofs);
				int           brushCount = (int)(brushLump->filelen / sizeof(dbrush_t));
				int           sideCount  = (int)(sideLump->filelen / sizeof(dbrushside_t));
				int           planeCount = (int)(planeLump->filelen / sizeof(dplane_t));

				if ((unsigned)brushNum < (unsigned)brushCount)
				{
					int firstSide = LittleLong(brushes[brushNum].firstSide);

					if (firstSide + 5 < sideCount)
					{
						int s, p;

						// Axial sides are always the first 6.
						// BSP convention: mins[axis] = -dist of negative-facing plane;
						// maxs[axis] = dist of positive-facing plane.
						s = firstSide + 0; p = LittleLong(sides[s].planeNum);
						if (p >= 0 && p < planeCount)
						{
								memcpy(&out->mins[0], &planes[p].dist, sizeof(float));
							out->mins[0] = -out->mins[0];
						}

						s = firstSide + 1; p = LittleLong(sides[s].planeNum);
						if (p >= 0 && p < planeCount)
						{
							memcpy(&out->maxs[0], &planes[p].dist, sizeof(float));
						}

						s = firstSide + 2; p = LittleLong(sides[s].planeNum);
						if (p >= 0 && p < planeCount)
						{
							memcpy(&out->mins[1], &planes[p].dist, sizeof(float));
							out->mins[1] = -out->mins[1];
						}

						s = firstSide + 3; p = LittleLong(sides[s].planeNum);
						if (p >= 0 && p < planeCount)
						{
							memcpy(&out->maxs[1], &planes[p].dist, sizeof(float));
						}

						s = firstSide + 4; p = LittleLong(sides[s].planeNum);
						if (p >= 0 && p < planeCount)
						{
							memcpy(&out->mins[2], &planes[p].dist, sizeof(float));
							out->mins[2] = -out->mins[2];
						}

						s = firstSide + 5; p = LittleLong(sides[s].planeNum);
						if (p >= 0 && p < planeCount)
						{
							memcpy(&out->maxs[2], &planes[p].dist, sizeof(float));
						}
					}
				}
			}
		}

		dx12.ri.Printf(PRINT_DEVELOPER, "DX12_LoadWorld: loaded %d fog volumes\n", fogCount);
	}

	// ----------------------------------------------------------------
	// 4.  Draw-verts + draw-indexes → CPU staging arrays
	//     (also tessellate patches inline)
	// ----------------------------------------------------------------

	// Allocate generous staging arrays on the heap (freed before returning)
	{
		lump_t     *vertsLump   = &header->lumps[LUMP_DRAWVERTS];
		lump_t     *indexLump   = &header->lumps[LUMP_DRAWINDEXES];
		lump_t     *surfLump    = &header->lumps[LUMP_SURFACES];
		lump_t     *modelLump   = &header->lumps[LUMP_MODELS];
		drawVert_t *bspVerts    = NULL;
		int        *bspIndexes  = NULL;
		dsurface_t *bspSurfaces = NULL;
		dmodel_t   *bspModels   = NULL;
		int         bspVertCount  = 0;
		int         bspIndexCount = 0;
		int         bspSurfCount  = 0;
		int         bspModelCount = 0;

		// Validate and obtain pointers for each lump
		if (vertsLump->filelen % (int)sizeof(drawVert_t))
		{
			dx12.ri.Printf(PRINT_WARNING, "DX12_LoadWorld: bad LUMP_DRAWVERTS size\n");
			dx12.ri.FS_FreeFile(buffer);
			return;
		}

		if (indexLump->filelen % (int)sizeof(int))
		{
			dx12.ri.Printf(PRINT_WARNING, "DX12_LoadWorld: bad LUMP_DRAWINDEXES size\n");
			dx12.ri.FS_FreeFile(buffer);
			return;
		}

		if (surfLump->filelen % (int)sizeof(dsurface_t))
		{
			dx12.ri.Printf(PRINT_WARNING, "DX12_LoadWorld: bad LUMP_SURFACES size\n");
			dx12.ri.FS_FreeFile(buffer);
			return;
		}

		bspVerts     = (drawVert_t *)(fileBase + vertsLump->fileofs);
		bspIndexes   = (int *)(fileBase + indexLump->fileofs);
		bspSurfaces  = (dsurface_t *)(fileBase + surfLump->fileofs);
		bspVertCount  = (int)(vertsLump->filelen / sizeof(drawVert_t));
		bspIndexCount = (int)(indexLump->filelen / sizeof(int));
		bspSurfCount  = (int)(surfLump->filelen / sizeof(dsurface_t));

		if (modelLump->filelen > 0 && (modelLump->filelen % (int)sizeof(dmodel_t)) == 0)
		{
			bspModels     = (dmodel_t *)(fileBase + modelLump->fileofs);
			bspModelCount = (int)(modelLump->filelen / sizeof(dmodel_t));
		}

		// Estimate worst-case staging sizes.
		// Patches can expand: a (w×h) grid produces
		//   (w-1)/2 * (h-1)/2 * PATCH_LOD² * 6 indices and
		//   (w-1)/2 * (h-1)/2 * (PATCH_LOD+1)² verts.
		// We over-estimate to keep the code simple.
		int maxStagingVerts   = bspVertCount + bspSurfCount * (PATCH_LOD + 1) * (PATCH_LOD + 1);
		int maxStagingIndexes = bspIndexCount + bspSurfCount * PATCH_LOD * PATCH_LOD * 6;

		dx12WorldVertex_t *stagingVerts   = (dx12WorldVertex_t *)malloc(
			(size_t)maxStagingVerts * sizeof(dx12WorldVertex_t));
		int               *stagingIndexes = (int *)malloc(
			(size_t)maxStagingIndexes * sizeof(int));

		if (!stagingVerts || !stagingIndexes)
		{
			dx12.ri.Printf(PRINT_WARNING, "DX12_LoadWorld: out of memory for staging buffers\n");
			if (stagingVerts)
			{
				free(stagingVerts);
			}
			if (stagingIndexes)
			{
				free(stagingIndexes);
			}
			dx12.ri.FS_FreeFile(buffer);
			return;
		}

		int vtxWrite = 0;  // next free vertex slot in stagingVerts
		int idxWrite = 0;  // next free index  slot in stagingIndexes

		// Convert one drawVert_t → dx12WorldVertex_t
		// Inline lambda emulated as a local macro for C compatibility
#define CONVERT_VERT(bv, dv)                                      \
		do {                                                          \
			(dv).xyz[0]    = (bv).xyz[0];                             \
			(dv).xyz[1]    = (bv).xyz[1];                             \
			(dv).xyz[2]    = (bv).xyz[2];                             \
			(dv).st[0]     = (bv).st[0];                              \
			(dv).st[1]     = (bv).st[1];                              \
			(dv).lm[0]     = (bv).lightmap[0];                        \
			(dv).lm[1]     = (bv).lightmap[1];                        \
			(dv).normal[0] = (bv).normal[0];                          \
			(dv).normal[1] = (bv).normal[1];                          \
			(dv).normal[2] = (bv).normal[2];                          \
			(dv).color[0]  = (bv).color[0] / 255.0f;                 \
			(dv).color[1]  = (bv).color[1] / 255.0f;                 \
			(dv).color[2]  = (bv).color[2] / 255.0f;                 \
			(dv).color[3]  = (bv).color[3] / 255.0f;                 \
		} while (0)

		// ----------------------------------------------------------------
		// 5.  Surface lump → dx12DrawSurf_t + fill staging arrays
		// ----------------------------------------------------------------
		for (i = 0; i < bspSurfCount; i++)
		{
			dsurface_t *in      = &bspSurfaces[i];
			int         sType   = LittleLong(in->surfaceType);
			int         shNum   = LittleLong(in->shaderNum);
			int         fogNum  = LittleLong(in->fogNum);
			int         fv      = LittleLong(in->firstVert);
			int         nv      = LittleLong(in->numVerts);
			int         fi      = LittleLong(in->firstIndex);
			int         ni      = LittleLong(in->numIndexes);
			qhandle_t   matH    = 0;
			dx12Material_t *mat = NULL;

			// Validate shader index
			if (shNum >= 0 && shNum < dx12World.numShaders)
			{
				matH = dx12World.shaderHandles[shNum];
				mat  = DX12_GetMaterial(matH);
			}

			// Skip nodraw surfaces and flares
			if (sType == MST_FLARE)
			{
				continue;
			}

			if (mat && mat->isNodraw)
			{
				continue;
			}

			if (dx12World.numDrawSurfs >= DX12_MAX_DRAW_SURFS)
			{
				dx12.ri.Printf(PRINT_WARNING,
				               "DX12_LoadWorld: draw surface limit reached (%d)\n",
				               DX12_MAX_DRAW_SURFS);
				break;
			}

			dx12DrawSurf_t *ds = &dx12World.drawSurfs[dx12World.numDrawSurfs];
			Com_Memset(ds, 0, sizeof(*ds));

			ds->materialHandle = matH;
			ds->fogIndex       = (fogNum >= 0 && fogNum < dx12World.numFogs - 1) ? (fogNum + 1) : 0;
			ds->surfaceType    = sType;
			ds->isSky          = mat ? mat->isSky         : qfalse;
			ds->isTranslucent  = mat ? mat->isTranslucent : qfalse;
			ds->isFog          = mat ? mat->isFog         : qfalse;

			if (sType == MST_PATCH)
			{
				// Tessellate a Bezier patch into triangles.
				// ET BSP patches are stored as a (patchWidth × patchHeight)
				// control-point grid of 3×3 quad patches.
				int pw      = LittleLong(in->patchWidth);
				int ph      = LittleLong(in->patchHeight);
				int numPW   = (pw - 1) / 2; // number of 3×3 patches horizontally
				int numPH   = (ph - 1) / 2; // number of 3×3 patches vertically
				int steps   = PATCH_LOD;    // steps per patch edge
				int px, py, row, col;

				ds->firstVertex = vtxWrite;
				ds->firstIndex  = idxWrite;

				for (py = 0; py < numPH; py++)
				{
					for (px = 0; px < numPW; px++)
					{
						// Build the 3×3 control grid for this sub-patch
						dx12WorldVertex_t grid[9];
						int               cpBase = fv + py * 2 * pw + px * 2;
						int               r, c;

						for (r = 0; r < 3; r++)
						{
							for (c = 0; c < 3; c++)
							{
								int cpIdx = cpBase + r * pw + c;

								if (cpIdx < 0 || cpIdx >= bspVertCount)
								{
									Com_Memset(&grid[r * 3 + c], 0, sizeof(dx12WorldVertex_t));
									continue;
								}
								CONVERT_VERT(bspVerts[cpIdx], grid[r * 3 + c]);
							}
						}

						// Tessellate and emit vertices + indices
						int patchVtxBase = vtxWrite;

						for (row = 0; row <= steps; row++)
						{
							for (col = 0; col <= steps; col++)
							{
								float             s = (float)col / (float)steps;
								float             t = (float)row / (float)steps;
								dx12WorldVertex_t v;

								WLD_BezierEval(grid, s, t, &v);

								if (vtxWrite >= maxStagingVerts)
								{
									goto patch_overflow;
								}
								stagingVerts[vtxWrite++] = v;
							}
						}

						// Build quad-strip indices for this patch
						for (row = 0; row < steps; row++)
						{
							for (col = 0; col < steps; col++)
							{
								int tl = patchVtxBase + row * (steps + 1) + col;
								int tr = tl + 1;
								int bl = tl + (steps + 1);
								int br = bl + 1;

								if (idxWrite + 6 > maxStagingIndexes)
								{
									goto patch_overflow;
								}

								stagingIndexes[idxWrite++] = tl;
								stagingIndexes[idxWrite++] = bl;
								stagingIndexes[idxWrite++] = tr;
								stagingIndexes[idxWrite++] = tr;
								stagingIndexes[idxWrite++] = bl;
								stagingIndexes[idxWrite++] = br;
							}
						}
					}
				}

patch_overflow:
				ds->numVertices = vtxWrite - ds->firstVertex;
				ds->numIndexes  = idxWrite - ds->firstIndex;
			}
			else
			{
				// Planar / triangle-soup: copy raw BSP verts + remap indices
				int j;

				ds->firstVertex = vtxWrite;
				ds->firstIndex  = idxWrite;

				// Copy vertices
				for (j = 0; j < nv; j++)
				{
					int bvIdx = fv + j;

					if (bvIdx < 0 || bvIdx >= bspVertCount || vtxWrite >= maxStagingVerts)
					{
						break;
					}
					CONVERT_VERT(bspVerts[bvIdx], stagingVerts[vtxWrite]);
					vtxWrite++;
				}

				// Copy indices, rebased to our staging array
				for (j = 0; j < ni; j++)
				{
					int bsiIdx = fi + j;

					if (bsiIdx < 0 || bsiIdx >= bspIndexCount || idxWrite >= maxStagingIndexes)
					{
						break;
					}
					stagingIndexes[idxWrite++] = ds->firstVertex + LittleLong(bspIndexes[bsiIdx]);
				}

				ds->numVertices = vtxWrite - ds->firstVertex;
				ds->numIndexes  = idxWrite - ds->firstIndex;
			}

#undef CONVERT_VERT

			if (ds->numIndexes > 0 && ds->numVertices > 0)
			{
				dx12World.numDrawSurfs++;
			}
		}

		dx12.ri.Printf(PRINT_DEVELOPER,
		               "DX12_LoadWorld: %d draw surfaces, %d vertices, %d indices\n",
		               dx12World.numDrawSurfs, vtxWrite, idxWrite);

		// ----------------------------------------------------------------
		// 6.  Sort draw surface list
		// ----------------------------------------------------------------
		if (dx12World.numDrawSurfs > 1)
		{
			qsort(dx12World.drawSurfs, (size_t)dx12World.numDrawSurfs,
			      sizeof(dx12DrawSurf_t), WLD_SortSurfs);
		}

		// ----------------------------------------------------------------
		// 7.  Upload vertex + index staging arrays to GPU
		// ----------------------------------------------------------------
		if (vtxWrite > 0 && idxWrite > 0)
		{
			UINT64 vbBytes = (UINT64)vtxWrite * sizeof(dx12WorldVertex_t);
			UINT64 ibBytes = (UINT64)idxWrite * sizeof(int);

			if (!WLD_UploadBuffer(stagingVerts, vbBytes, &dx12World.vertexBuffer))
			{
				dx12.ri.Printf(PRINT_WARNING,
				               "DX12_LoadWorld: vertex buffer upload failed\n");
			}
			else
			{
				dx12World.numVertices = (UINT)vtxWrite;
			}

			if (!WLD_UploadBuffer(stagingIndexes, ibBytes, &dx12World.indexBuffer))
			{
				dx12.ri.Printf(PRINT_WARNING,
				               "DX12_LoadWorld: index buffer upload failed\n");
			}
			else
			{
				dx12World.numIndexes = (UINT)idxWrite;
			}
		}

		free(stagingVerts);
		free(stagingIndexes);

		// ----------------------------------------------------------------
		// 8.  Submodel lump → dx12WorldModel_t
		// ----------------------------------------------------------------
		if (bspModels && bspModelCount > 0)
		{
			int surfBase  = 0; // running count of draw surfaces so far for offset mapping
			int modelIdx;

			// Build a mapping from BSP surface index → dx12DrawSurf_t index.
			// Since we may have skipped some surfaces, we pre-compute offsets.
			// Simple approach: scan bspSurfaces again to build the table.
			int *surfMap = (int *)malloc((size_t)bspSurfCount * sizeof(int));
			if (surfMap)
			{
				int drawIdx = 0;

				for (i = 0; i < bspSurfCount; i++)
				{
					dsurface_t *in    = &bspSurfaces[i];
					int         sType = LittleLong(in->surfaceType);
					int         shNum = LittleLong(in->shaderNum);
					qhandle_t   matH  = (shNum >= 0 && shNum < dx12World.numShaders)
					                    ? dx12World.shaderHandles[shNum] : 0;
					dx12Material_t *mat = DX12_GetMaterial(matH);

					if (sType == MST_FLARE || (mat && mat->isNodraw))
					{
						surfMap[i] = -1;
					}
					else
					{
						surfMap[i] = drawIdx++;
					}
				}

				dx12World.numModels = bspModelCount < DX12_MAX_WORLD_MODELS
				                      ? bspModelCount : DX12_MAX_WORLD_MODELS;

				for (modelIdx = 0; modelIdx < dx12World.numModels; modelIdx++)
				{
					dx12WorldModel_t *out = &dx12World.models[modelIdx];
					dmodel_t         *in  = &bspModels[modelIdx];
					int               fs  = LittleLong(in->firstSurface);
					int               ns  = LittleLong(in->numSurfaces);
					int               si;

					for (i = 0; i < 3; i++)
					{
						out->mins[i] = in->mins[i];
						out->maxs[i] = in->maxs[i];
					}

					// Map BSP surface indices to the draw surface list.
					// firstSurface = first mapped draw index; numSurfaces = count.
					out->firstSurface = -1;
					out->numSurfaces  = 0;

					for (si = 0; si < ns; si++)
					{
						int bspSI = fs + si;

						if (bspSI >= 0 && bspSI < bspSurfCount)
						{
							int di = surfMap[bspSI];

							if (di >= 0)
							{
								if (out->firstSurface < 0)
								{
									out->firstSurface = di;
								}
								out->numSurfaces++;
							}
						}
					}

					(void)surfBase;
				}

				free(surfMap);
			}
		}
	}

	dx12.ri.FS_FreeFile(buffer);

	dx12World.loaded = qtrue;

	dx12.ri.Printf(PRINT_ALL,
	               "DX12_LoadWorld: '%s' loaded – %d surfs, %d verts, %d idxs, %d fogs\n",
	               name, dx12World.numDrawSurfs,
	               (int)dx12World.numVertices, (int)dx12World.numIndexes,
	               dx12World.numFogs - 1);
}

#endif // _WIN32
