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
 * @file dx12_model.cpp
 * @brief DX12 MD3 model loader and entity draw dispatch.
 *
 * Parses MD3 binary format, extracts frame-0 geometry, uploads vertex and
 * index buffers to GPU default-heap resources, then issues indexed draw calls
 * through the existing 3D scene pipeline (pso3D / rootSignature3D).
 *
 * Vertex format used is dx12WorldVertex_t so models share the same PSO as
 * world geometry; no additional pipeline state is required.
 */

#include "dx12_model.h"
#include "dx12_world.h"    // dx12WorldVertex_t
#include "dx12_shader.h"   // DX12_RegisterTexture, DX12_GetTexture, DX12_RegisterMaterial, DX12_GetMaterial
#include "dx12_skeletal.h" // DX12_GetBoneTagMDS, DX12_GetBoneTagMDM, DX12_FreeSkeletal

#ifdef _WIN32

#include "../qcommon/qfiles.h"  // md3Header_t, md3Surface_t, MD3_IDENT, MD3_XYZ_SCALE …
#include <stdlib.h>             // malloc / free
#include <string.h>             // memcpy, memset
#include <math.h>               // cosf / sinf

// ---------------------------------------------------------------------------
// Global model registry (parallel to dx12ModelNames[] in tr_dx12_main.cpp)
// ---------------------------------------------------------------------------

dx12ModelEntry_t dx12ModelData[DX12_MAX_MODELS];

// ---------------------------------------------------------------------------
// Internal upload helper
// ---------------------------------------------------------------------------

/**
 * @brief Upload a CPU buffer to a GPU default-heap resource.
 *
 * Mirrors WLD_UploadBuffer() in dx12_world.cpp.  Creates a temporary upload
 * heap, copies the data, issues a CopyBufferRegion command, transitions the
 * destination to GENERIC_READ, executes and synchronously waits.
 *
 * @param data      CPU-side source bytes.
 * @param sizeBytes Number of bytes to upload.
 * @param outBuffer Receives the newly created ID3D12Resource on success.
 * @return          qtrue on success; *outBuffer is released and set to NULL on failure.
 */
static qboolean MDL_UploadBuffer(const void *data, UINT64 sizeBytes,
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
		// D3D12 buffers are always effectively created in COMMON state;
		// using COPY_DEST triggers a debug layer warning.  COMMON buffers
		// are implicitly promoted to COPY_DEST when used as a copy
		// destination, so the barrier below (COPY_DEST → GENERIC_READ)
		// is still valid.
		D3D12_RESOURCE_STATE_COMMON,
		NULL,
		IID_PPV_ARGS(outBuffer));

	if (FAILED(hr))
	{
		dx12.ri.Printf(PRINT_WARNING,
		               "MDL_UploadBuffer: CreateCommittedResource (default) failed (0x%08lx)\n", hr);
		return qfalse;
	}

	// Temporary upload heap
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
		               "MDL_UploadBuffer: CreateCommittedResource (upload) failed (0x%08lx)\n", hr);
		(*outBuffer)->Release();
		*outBuffer = NULL;
		return qfalse;
	}

	{
		D3D12_RANGE readRange = { 0, 0 };
		void       *mapped    = NULL;

		hr = uploadBuf->Map(0, &readRange, &mapped);
		if (FAILED(hr))
		{
			dx12.ri.Printf(PRINT_WARNING,
			               "MDL_UploadBuffer: Map failed (0x%08lx)\n", hr);
			uploadBuf->Release();
			(*outBuffer)->Release();
			*outBuffer = NULL;
			return qfalse;
		}

		memcpy(mapped, data, (size_t)sizeBytes);
		uploadBuf->Unmap(0, NULL);
	}

	// Reset dedicated upload command allocator + list for this transfer
	hr = dx12.uploadCmdAllocator->Reset();
	if (FAILED(hr))
	{
		uploadBuf->Release();
		(*outBuffer)->Release();
		*outBuffer = NULL;
		return qfalse;
	}

	hr = dx12.uploadCmdList->Reset(dx12.uploadCmdAllocator, NULL);
	if (FAILED(hr))
	{
		uploadBuf->Release();
		(*outBuffer)->Release();
		*outBuffer = NULL;
		return qfalse;
	}

	dx12.uploadCmdList->CopyBufferRegion(*outBuffer, 0, uploadBuf, 0, sizeBytes);

	{
		D3D12_RESOURCE_BARRIER barrier = {};
		barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier.Transition.pResource   = *outBuffer;
		barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_GENERIC_READ;
		dx12.uploadCmdList->ResourceBarrier(1, &barrier);
	}

	hr = dx12.uploadCmdList->Close();
	if (FAILED(hr))
	{
		uploadBuf->Release();
		(*outBuffer)->Release();
		*outBuffer = NULL;
		return qfalse;
	}

	{
		ID3D12CommandList *lists[] = { dx12.uploadCmdList };
		dx12.commandQueue->ExecuteCommandLists(1, lists);
		DX12_WaitForUpload(dx12.commandQueue);
	}

	uploadBuf->Release();
	return qtrue;
}

// ---------------------------------------------------------------------------
// MD3 normal decode
// ---------------------------------------------------------------------------

/**
 * @brief Decode a packed MD3 spherical-coordinates normal to a unit xyz vector.
 *
 * The 16-bit encoding stores two 8-bit latitude/longitude values:
 *   lat = (encoded >> 8) & 0xFF
 *   lng = (encoded     ) & 0xFF
 * Both are mapped from [0, 255] to [0, 2π].
 *
 * @param encoded  16-bit packed normal from md3XyzNormal_t.
 * @param nx,ny,nz Receive the decoded normal components.
 */
static void MDL_DecodeNormal(short encoded, float *nx, float *ny, float *nz)
{
	float lat = ((encoded >> 8) & 0xFF) * (float)(3.14159265f / 128.0f);
	float lng = (encoded & 0xFF)        * (float)(3.14159265f / 128.0f);

	*nx = cosf(lat) * sinf(lng);
	*ny = sinf(lat) * sinf(lng);
	*nz = cosf(lng);
}

// ---------------------------------------------------------------------------
// DX12_LoadMD3
// ---------------------------------------------------------------------------

/**
 * @brief DX12_LoadMD3
 * @param[in] slot  Model registry slot (handle - 1).
 * @param[in] name  Game-path of the .md3 file.
 * @return          qtrue when ≥1 surface was uploaded successfully.
 *
 * Only frame 0 is processed; animation is intentionally not supported in
 * this minimal renderer.  MD3 surfaces beyond DX12_MAX_MODEL_SURFACES are
 * silently skipped.
 */
qboolean DX12_LoadMD3(int slot, const char *name)
{
	void         *fileData = NULL;
	int           fileLen;
	md3Header_t  *header;
	md3Surface_t *surf;
	int           s;
	dx12ModelEntry_t *entry;

	if (slot < 0 || slot >= DX12_MAX_MODELS)
	{
		return qfalse;
	}

	entry = &dx12ModelData[slot];
	Com_Memset(entry, 0, sizeof(*entry));

	fileLen = dx12.ri.FS_ReadFile(name, &fileData);
	if (fileLen <= 0 || !fileData)
	{
		return qfalse;
	}

	if (fileLen < (int)sizeof(md3Header_t))
	{
		dx12.ri.FS_FreeFile(fileData);
		return qfalse;
	}

	header = (md3Header_t *)fileData;

	// Validate MD3 ident and version
	if (header->ident != MD3_IDENT || header->version != MD3_VERSION)
	{
		// Silently skip non-MD3 formats (MDX, MDM, etc.)
		dx12.ri.FS_FreeFile(fileData);
		return qfalse;
	}

	if (header->numSurfaces <= 0 || header->ofsSurfaces < (int)sizeof(md3Header_t)
	    || header->ofsSurfaces >= fileLen)
	{
		dx12.ri.FS_FreeFile(fileData);
		return qfalse;
	}

	// Initialise AABB to inverted extremes so we can accumulate per-vertex
	entry->mins[0] = entry->mins[1] = entry->mins[2] =  1.0e9f;
	entry->maxs[0] = entry->maxs[1] = entry->maxs[2] = -1.0e9f;

	surf = (md3Surface_t *)((byte *)fileData + header->ofsSurfaces);

	for (s = 0; s < header->numSurfaces && s < DX12_MAX_MODEL_SURFACES; s++)
	{
		md3XyzNormal_t   *xyzn;
		md3St_t          *st;
		md3Triangle_t    *tris;
		md3Shader_t      *shaders;
		UINT              numVerts;
		UINT              numTris;
		dx12WorldVertex_t *cpuVerts;
		int               *cpuIdx;
		dx12ModelSurface_t *ms;
		int                v, t;

		// Basic bounds check for the surface header
		if ((byte *)surf + sizeof(md3Surface_t) > (byte *)fileData + fileLen)
		{
			break;
		}

		numVerts = (UINT)surf->numVerts;
		numTris  = (UINT)surf->numTriangles;

		if (numVerts == 0 || numTris == 0)
		{
			surf = (md3Surface_t *)((byte *)surf + surf->ofsEnd);
			continue;
		}

		// Sub-array pointers (relative to the start of this surface block)
		xyzn    = (md3XyzNormal_t *)((byte *)surf + surf->ofsXyzNormals);
		st      = (md3St_t *)((byte *)surf + surf->ofsSt);
		tris    = (md3Triangle_t *)((byte *)surf + surf->ofsTriangles);
		shaders = (md3Shader_t *)((byte *)surf + surf->ofsShaders);

		// Validate that the sub-arrays fit within the file
		if ((byte *)xyzn + numVerts * sizeof(md3XyzNormal_t) > (byte *)fileData + fileLen ||
		    (byte *)st   + numVerts * sizeof(md3St_t)        > (byte *)fileData + fileLen ||
		    (byte *)tris + numTris  * sizeof(md3Triangle_t)  > (byte *)fileData + fileLen)
		{
			surf = (md3Surface_t *)((byte *)surf + surf->ofsEnd);
			continue;
		}

		cpuVerts = (dx12WorldVertex_t *)dx12.ri.Z_Malloc(numVerts * sizeof(dx12WorldVertex_t));
		cpuIdx   = (int *)dx12.ri.Z_Malloc(numTris * 3 * sizeof(int));

		if (!cpuVerts || !cpuIdx)
		{
			if (cpuVerts)
			{
				dx12.ri.Free(cpuVerts);
			}
			if (cpuIdx)
			{
				dx12.ri.Free(cpuIdx);
			}
			surf = (md3Surface_t *)((byte *)surf + surf->ofsEnd);
			continue;
		}

		// Build frame-0 vertices
		for (v = 0; v < (int)numVerts; v++)
		{
			float nx, ny, nz;
			float px = xyzn[v].xyz[0] * (float)MD3_XYZ_SCALE;
			float py = xyzn[v].xyz[1] * (float)MD3_XYZ_SCALE;
			float pz = xyzn[v].xyz[2] * (float)MD3_XYZ_SCALE;

			MDL_DecodeNormal(xyzn[v].normal, &nx, &ny, &nz);

			cpuVerts[v].xyz[0]    = px;
			cpuVerts[v].xyz[1]    = py;
			cpuVerts[v].xyz[2]    = pz;
			cpuVerts[v].st[0]     = st[v].st[0];
			cpuVerts[v].st[1]     = st[v].st[1];
			cpuVerts[v].lm[0]     = 0.0f;  // no lightmap for models
			cpuVerts[v].lm[1]     = 0.0f;
			cpuVerts[v].normal[0] = nx;
			cpuVerts[v].normal[1] = ny;
			cpuVerts[v].normal[2] = nz;
			cpuVerts[v].color[0]  = 1.0f;  // full-bright modulate
			cpuVerts[v].color[1]  = 1.0f;
			cpuVerts[v].color[2]  = 1.0f;
			cpuVerts[v].color[3]  = 1.0f;

			// Accumulate AABB
			if (px < entry->mins[0]) { entry->mins[0] = px; }
			if (py < entry->mins[1]) { entry->mins[1] = py; }
			if (pz < entry->mins[2]) { entry->mins[2] = pz; }
			if (px > entry->maxs[0]) { entry->maxs[0] = px; }
			if (py > entry->maxs[1]) { entry->maxs[1] = py; }
			if (pz > entry->maxs[2]) { entry->maxs[2] = pz; }
		}

		// Copy triangle indices
		for (t = 0; t < (int)numTris; t++)
		{
			cpuIdx[t * 3 + 0] = tris[t].indexes[0];
			cpuIdx[t * 3 + 1] = tris[t].indexes[1];
			cpuIdx[t * 3 + 2] = tris[t].indexes[2];
		}

		// Upload to GPU
		ms = &entry->surfaces[entry->numSurfaces];

		if (!MDL_UploadBuffer(cpuVerts, (UINT64)numVerts * sizeof(dx12WorldVertex_t),
		                      &ms->vertexBuffer)
		    || !MDL_UploadBuffer(cpuIdx, (UINT64)numTris * 3 * sizeof(int),
		                         &ms->indexBuffer))
		{
			if (ms->vertexBuffer)
			{
				ms->vertexBuffer->Release();
				ms->vertexBuffer = NULL;
			}
			if (ms->indexBuffer)
			{
				ms->indexBuffer->Release();
				ms->indexBuffer = NULL;
			}
			dx12.ri.Free(cpuVerts);
			dx12.ri.Free(cpuIdx);
			surf = (md3Surface_t *)((byte *)surf + surf->ofsEnd);
			continue;
		}

		dx12.ri.Free(cpuVerts);
		dx12.ri.Free(cpuIdx);

		ms->numVertices = numVerts;
		ms->numIndices  = numTris * 3;

		// Store the surface name (lower-cased) for skin lookup at draw time
		Q_strncpyz(ms->surfName, surf->name, sizeof(ms->surfName));
		Q_strlwr(ms->surfName);

		// Register the surface's diffuse texture / shader (fallback when no skin)
		if (surf->numShaders > 0 && shaders[0].name[0])
		{
			ms->texHandle     = DX12_RegisterTexture(shaders[0].name);
			if (!ms->texHandle)
			{
				ms->texHandle = DX12_RegisterMaterial(shaders[0].name);
			}
			// Always register as a full material so multi-stage effects (alpha test,
			// animated textures, blend modes) work for entity model surfaces.
			ms->materialHandle = DX12_RegisterMaterial(shaders[0].name);
		}

		entry->numSurfaces++;

		// Advance to the next surface via the embedded chain offset
		surf = (md3Surface_t *)((byte *)surf + surf->ofsEnd);
	}

	// -----------------------------------------------------------------------
	// Parse tag data (all frames) for DX12_LerpTag
	// -----------------------------------------------------------------------
	if (header->numTags > 0 && header->numFrames > 0
	    && header->ofsTags >= (int)sizeof(md3Header_t)
	    && header->ofsTags < fileLen)
	{
		int     tagTotal = header->numFrames * header->numTags;
		size_t  tagBytes = (size_t)tagTotal * sizeof(md3Tag_t);

		// Validate the tag block fits inside the file
		if (header->ofsTags + (int)tagBytes <= fileLen)
		{
			entry->tags = (md3Tag_t *)dx12.ri.Z_Malloc(tagBytes);
			if (entry->tags)
			{
				memcpy(entry->tags,
				       (byte *)fileData + header->ofsTags,
				       tagBytes);
				entry->numTags   = header->numTags;
				entry->numFrames = header->numFrames;
			}
		}
	}

	dx12.ri.FS_FreeFile(fileData);

	if (entry->numSurfaces > 0)
	{
		entry->valid     = qtrue;
		entry->modelType = DX12_MOD_MD3;  // mark as MD3 for LerpTag dispatch
		dx12.ri.Printf(PRINT_DEVELOPER,
		               "DX12_LoadMD3: loaded '%s' (%d surface%s, %d tags)\n",
		               name, entry->numSurfaces,
		               entry->numSurfaces == 1 ? "" : "s",
		               entry->numTags);
	}

	return entry->valid;
}

// ---------------------------------------------------------------------------
// DX12_LoadMDC
// ---------------------------------------------------------------------------

/**
 * @brief Load a Compressed MD3 (MDC) model into a GPU model slot.
 *
 * MDC is the compressed MD3 variant used for gibs, shells, and weapon effects
 * in the original Enemy Territory game data.  The on-disk format stores vertex
 * positions for 'base' frames in the same md3XyzNormal_t layout as MD3, plus
 * compressed delta frames for the remaining poses.  For DX12 static rendering
 * (frame-0 only) only the first base frame is needed, so the compressed data
 * is never touched.
 *
 * Surface layout is otherwise identical to MD3 – same md3Triangle_t,
 * md3St_t, and md3Shader_t blocks – so the GPU upload path is shared.
 *
 * @param slot  Registry slot (handle - 1).
 * @param name  VFS game-path of the .mdc file.
 * @return      qtrue when at least one surface was uploaded successfully.
 */
qboolean DX12_LoadMDC(int slot, const char *name)
{
	void             *fileData = NULL;
	int               fileLen;
	mdcHeader_t      *header;
	mdcSurface_t     *surf;
	int               s;
	dx12ModelEntry_t *entry;

	if (slot < 0 || slot >= DX12_MAX_MODELS)
	{
		return qfalse;
	}

	entry = &dx12ModelData[slot];
	Com_Memset(entry, 0, sizeof(*entry));

	fileLen = dx12.ri.FS_ReadFile(name, &fileData);
	if (fileLen <= 0 || !fileData)
	{
		return qfalse;
	}

	if (fileLen < (int)sizeof(mdcHeader_t))
	{
		dx12.ri.FS_FreeFile(fileData);
		return qfalse;
	}

	header = (mdcHeader_t *)fileData;

	if (header->ident != MDC_IDENT || header->version != MDC_VERSION)
	{
		dx12.ri.FS_FreeFile(fileData);
		return qfalse;
	}

	if (header->numSurfaces <= 0 || header->ofsSurfaces < (int)sizeof(mdcHeader_t)
	    || header->ofsSurfaces >= fileLen)
	{
		dx12.ri.FS_FreeFile(fileData);
		return qfalse;
	}

	entry->mins[0] = entry->mins[1] = entry->mins[2] =  1.0e9f;
	entry->maxs[0] = entry->maxs[1] = entry->maxs[2] = -1.0e9f;

	surf = (mdcSurface_t *)((byte *)fileData + header->ofsSurfaces);

	for (s = 0; s < header->numSurfaces && s < DX12_MAX_MODEL_SURFACES; s++)
	{
		md3XyzNormal_t    *xyzn;
		md3St_t           *st;
		md3Triangle_t     *tris;
		md3Shader_t       *shaders;
		short             *frameBaseFrames;
		int                baseFrameIdx;
		UINT               numVerts;
		UINT               numTris;
		dx12WorldVertex_t *cpuVerts;
		int               *cpuIdx;
		dx12ModelSurface_t *ms;
		int                v, t;

		if ((byte *)surf + sizeof(mdcSurface_t) > (byte *)fileData + fileLen)
		{
			break;
		}

		numVerts = (UINT)surf->numVerts;
		numTris  = (UINT)surf->numTriangles;

		if (numVerts == 0 || numTris == 0)
		{
			surf = (mdcSurface_t *)((byte *)surf + surf->ofsEnd);
			continue;
		}

		// For frame 0: look up which base frame to use via the per-frame index
		// table (ofsFrameBaseFrames), then index into the base-frame XYZ block.
		frameBaseFrames = (short *)((byte *)surf + surf->ofsFrameBaseFrames);
		baseFrameIdx    = (int)frameBaseFrames[0];

		xyzn    = (md3XyzNormal_t *)((byte *)surf + surf->ofsXyzNormals) + baseFrameIdx * (int)numVerts;
		st      = (md3St_t *)((byte *)surf + surf->ofsSt);
		tris    = (md3Triangle_t *)((byte *)surf + surf->ofsTriangles);
		shaders = (md3Shader_t *)((byte *)surf + surf->ofsShaders);

		// Validate sub-arrays fit within the file
		if ((byte *)xyzn + numVerts * sizeof(md3XyzNormal_t) > (byte *)fileData + fileLen ||
		    (byte *)st   + numVerts * sizeof(md3St_t)        > (byte *)fileData + fileLen ||
		    (byte *)tris + numTris  * sizeof(md3Triangle_t)  > (byte *)fileData + fileLen)
		{
			surf = (mdcSurface_t *)((byte *)surf + surf->ofsEnd);
			continue;
		}

		cpuVerts = (dx12WorldVertex_t *)dx12.ri.Z_Malloc(numVerts * sizeof(dx12WorldVertex_t));
		cpuIdx   = (int *)dx12.ri.Z_Malloc(numTris * 3 * sizeof(int));

		if (!cpuVerts || !cpuIdx)
		{
			if (cpuVerts) { dx12.ri.Free(cpuVerts); }
			if (cpuIdx)   { dx12.ri.Free(cpuIdx); }
			surf = (mdcSurface_t *)((byte *)surf + surf->ofsEnd);
			continue;
		}

		// Decode base-frame vertices (same md3XyzNormal_t format as MD3)
		for (v = 0; v < (int)numVerts; v++)
		{
			float nx, ny, nz;
			float px = xyzn[v].xyz[0] * (float)MD3_XYZ_SCALE;
			float py = xyzn[v].xyz[1] * (float)MD3_XYZ_SCALE;
			float pz = xyzn[v].xyz[2] * (float)MD3_XYZ_SCALE;

			MDL_DecodeNormal(xyzn[v].normal, &nx, &ny, &nz);

			cpuVerts[v].xyz[0]    = px;
			cpuVerts[v].xyz[1]    = py;
			cpuVerts[v].xyz[2]    = pz;
			cpuVerts[v].st[0]     = st[v].st[0];
			cpuVerts[v].st[1]     = st[v].st[1];
			cpuVerts[v].lm[0]     = 0.0f;
			cpuVerts[v].lm[1]     = 0.0f;
			cpuVerts[v].normal[0] = nx;
			cpuVerts[v].normal[1] = ny;
			cpuVerts[v].normal[2] = nz;
			cpuVerts[v].color[0]  = 1.0f;
			cpuVerts[v].color[1]  = 1.0f;
			cpuVerts[v].color[2]  = 1.0f;
			cpuVerts[v].color[3]  = 1.0f;

			if (px < entry->mins[0]) { entry->mins[0] = px; }
			if (py < entry->mins[1]) { entry->mins[1] = py; }
			if (pz < entry->mins[2]) { entry->mins[2] = pz; }
			if (px > entry->maxs[0]) { entry->maxs[0] = px; }
			if (py > entry->maxs[1]) { entry->maxs[1] = py; }
			if (pz > entry->maxs[2]) { entry->maxs[2] = pz; }
		}

		for (t = 0; t < (int)numTris; t++)
		{
			cpuIdx[t * 3 + 0] = tris[t].indexes[0];
			cpuIdx[t * 3 + 1] = tris[t].indexes[1];
			cpuIdx[t * 3 + 2] = tris[t].indexes[2];
		}

		ms = &entry->surfaces[entry->numSurfaces];

		if (!MDL_UploadBuffer(cpuVerts, (UINT64)numVerts * sizeof(dx12WorldVertex_t),
		                      &ms->vertexBuffer)
		    || !MDL_UploadBuffer(cpuIdx, (UINT64)numTris * 3 * sizeof(int),
		                         &ms->indexBuffer))
		{
			if (ms->vertexBuffer) { ms->vertexBuffer->Release(); ms->vertexBuffer = NULL; }
			if (ms->indexBuffer)  { ms->indexBuffer->Release();  ms->indexBuffer  = NULL; }
			dx12.ri.Free(cpuVerts);
			dx12.ri.Free(cpuIdx);
			surf = (mdcSurface_t *)((byte *)surf + surf->ofsEnd);
			continue;
		}

		dx12.ri.Free(cpuVerts);
		dx12.ri.Free(cpuIdx);

		ms->numVertices = numVerts;
		ms->numIndices  = numTris * 3;

		// Store the surface name (lower-cased) for skin lookup at draw time
		Q_strncpyz(ms->surfName, surf->name, sizeof(ms->surfName));
		Q_strlwr(ms->surfName);

		if (surf->numShaders > 0 && shaders[0].name[0])
		{
			ms->texHandle = DX12_RegisterTexture(shaders[0].name);
			if (!ms->texHandle)
			{
				ms->texHandle = DX12_RegisterMaterial(shaders[0].name);
			}
			// Always register as a full material for multi-stage effects.
			ms->materialHandle = DX12_RegisterMaterial(shaders[0].name);
		}

		entry->numSurfaces++;
		surf = (mdcSurface_t *)((byte *)surf + surf->ofsEnd);
	}

	// -----------------------------------------------------------------------
	// Parse tag data (all frames) for DX12_LerpTag.
	//
	// MDC stores per-frame tags as mdcTag_t (compressed short xyz + short
	// Euler angles) rather than the full md3Tag_t float matrix.  We decode
	// them to md3Tag_t format here at load time, exactly mirroring what
	// GL's R_LerpTag does on-demand for every frame lookup (tr_model.c).
	// After conversion the MD3 tag interpolation path in DX12_LerpTag works
	// identically for both MD3 and MDC models.
	// -----------------------------------------------------------------------
	if (header->numTags > 0 && header->numFrames > 0
	    && header->ofsTags >= (int)sizeof(mdcHeader_t)
	    && header->ofsTagNames >= (int)sizeof(mdcHeader_t)
	    && header->ofsTags < fileLen
	    && header->ofsTagNames < fileLen)
	{
		int           tagTotal   = header->numFrames * header->numTags;
		size_t        tagBytes   = (size_t)tagTotal * sizeof(md3Tag_t);
		size_t        nameBytes  = (size_t)header->numTags * sizeof(mdcTagName_t);
		size_t        mdcBytes   = (size_t)tagTotal * sizeof(mdcTag_t);

		// Validate both tag-data blocks fit inside the file
		if (header->ofsTags     + (int)mdcBytes  <= fileLen
		    && header->ofsTagNames + (int)nameBytes <= fileLen)
		{
			const mdcTagName_t *tagNames = (const mdcTagName_t *)((byte *)fileData + header->ofsTagNames);
			const mdcTag_t     *mdcTags  = (const mdcTag_t *)((byte *)fileData + header->ofsTags);

			entry->tags = (md3Tag_t *)dx12.ri.Z_Malloc(tagBytes);
			if (entry->tags)
			{
				int frameIdx, tagIdx;

				for (frameIdx = 0; frameIdx < header->numFrames; frameIdx++)
				{
					for (tagIdx = 0; tagIdx < header->numTags; tagIdx++)
					{
						const mdcTag_t *src  = &mdcTags[frameIdx * header->numTags + tagIdx];
						md3Tag_t       *dst  = &entry->tags[frameIdx * header->numTags + tagIdx];
						vec3_t          angles;
						int             component;

						// Decompress origin
						for (component = 0; component < 3; component++)
						{
							dst->origin[component] = (float)(src->xyz[component]) * (float)MD3_XYZ_SCALE;
						}

						// Decompress rotation: angles → axis matrix
						angles[0] = (float)(src->angles[0]) * (float)MDC_TAG_ANGLE_SCALE;
						angles[1] = (float)(src->angles[1]) * (float)MDC_TAG_ANGLE_SCALE;
						angles[2] = (float)(src->angles[2]) * (float)MDC_TAG_ANGLE_SCALE;
						AnglesToAxis(angles, dst->axis);

						// Copy tag name (only needs to be set once per tag index, but
						// we set it for every frame so any frame's tag is self-describing)
						Q_strncpyz(dst->name, tagNames[tagIdx].name, sizeof(dst->name));
					}
				}

				entry->numTags   = header->numTags;
				entry->numFrames = header->numFrames;
			}
		}
	}

	dx12.ri.FS_FreeFile(fileData);

	if (entry->numSurfaces > 0)
	{
		entry->valid     = qtrue;
		entry->modelType = DX12_MOD_MD3;  // same draw path as MD3
		dx12.ri.Printf(PRINT_DEVELOPER,
		               "DX12_LoadMDC: loaded '%s' (%d surface%s, %d tag%s)\n",
		               name, entry->numSurfaces,
		               entry->numSurfaces == 1 ? "" : "s",
		               entry->numTags,
		               entry->numTags == 1 ? "" : "s");
	}

	return entry->valid;
}

// ---------------------------------------------------------------------------
// DX12_ApplyTagTransform
// ---------------------------------------------------------------------------

/**
 * @brief Row-major 4×4 matrix multiply used internally by DX12_ApplyTagTransform.
 *        out = a * b.  out may NOT alias a or b.
 */
static void MDL_Mat4Mul(float out[4][4], const float a[4][4], const float b[4][4])
{
	int i, j, k;

	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			float s = 0.0f;

			for (k = 0; k < 4; k++)
			{
				s += a[i][k] * b[k][j];
			}
			out[i][j] = s;
		}
	}
}

/**
 * @brief Compute the child model's world matrix by concatenating a parent
 *        world matrix with a tag local matrix.
 *
 * A tag (md3Tag_t) stores a local-space origin and axis that are in the
 * same Q3 coordinate convention as an entity's origin/axis:
 *   tag.axis[0] = tag local forward (+X)
 *   tag.axis[1] = tag local left    (+Y)
 *   tag.axis[2] = tag local up      (+Z)
 *
 * The tag local matrix M_tag has exactly the same form as the model matrix
 * built by BuildModelMatrix in dx12_scene.cpp.  The child world matrix is:
 *   M_child = M_parent * M_tag
 *
 * The result is a row-major math matrix ready for Mat4Transpose + CB upload.
 *
 * @param[out] out     Child world matrix (row-major, pre-transpose).
 * @param[in]  parent  Parent world matrix (row-major, pre-transpose).
 * @param[in]  tag     Tag whose origin and axis are in parent-local space.
 */
void DX12_ApplyTagTransform(float out[4][4], const float parent[4][4], const md3Tag_t *tag)
{
	float tagLocal[4][4];

	// Build tag local matrix – same column layout as BuildModelMatrix:
	//   col 0 = tag.axis[0]  (local +X → parent forward)
	//   col 1 = tag.axis[1]  (local +Y → parent left)
	//   col 2 = tag.axis[2]  (local +Z → parent up)
	//   col 3 = tag.origin   (translation in parent-local space)
	tagLocal[0][0] = tag->axis[0][0]; tagLocal[0][1] = tag->axis[1][0]; tagLocal[0][2] = tag->axis[2][0]; tagLocal[0][3] = tag->origin[0];
	tagLocal[1][0] = tag->axis[0][1]; tagLocal[1][1] = tag->axis[1][1]; tagLocal[1][2] = tag->axis[2][1]; tagLocal[1][3] = tag->origin[1];
	tagLocal[2][0] = tag->axis[0][2]; tagLocal[2][1] = tag->axis[1][2]; tagLocal[2][2] = tag->axis[2][2]; tagLocal[2][3] = tag->origin[2];
	tagLocal[3][0] = 0.0f;           tagLocal[3][1] = 0.0f;           tagLocal[3][2] = 0.0f;           tagLocal[3][3] = 1.0f;

	// child_world = parent_world * tag_local
	MDL_Mat4Mul(out, parent, tagLocal);
}

// ---------------------------------------------------------------------------
// DX12_BuildSkeletalSurfaceIB
// ---------------------------------------------------------------------------

/**
 * @brief DX12_BuildSkeletalSurfaceIB – upload a static index buffer for a skeletal surface.
 */
qboolean DX12_BuildSkeletalSurfaceIB(dx12ModelSurface_t *ms, const int *triIndexes, int numTriangles)
{
	if (!ms || !triIndexes || numTriangles <= 0)
	{
		return qfalse;
	}

	if (!MDL_UploadBuffer(triIndexes, (UINT64)numTriangles * 3 * sizeof(int), &ms->indexBuffer))
	{
		return qfalse;
	}

	ms->numIndices = (UINT)(numTriangles * 3);
	return qtrue;
}

// ---------------------------------------------------------------------------
// DX12_AllocSkeletalVerts
// ---------------------------------------------------------------------------

/**
 * @brief Allocate numVerts contiguous slots in the frame-scope skeletal scratch VB.
 * Returns a CPU pointer to write into and sets *outGpuVA to the GPU VA for binding.
 * Returns NULL if the scratch buffer is full or unavailable.
 */
static UINT8 *DX12_AllocSkeletalVerts(int numVerts, D3D12_GPU_VIRTUAL_ADDRESS *outGpuVA)
{
	int offset;

	if (!dx12Scene.skelVB || !dx12Scene.skelVBMapped) { *outGpuVA = 0; return NULL; }
	if (dx12Scene.skelVBWriteOffset + numVerts > DX12_MAX_SKELETAL_VERTS) { *outGpuVA = 0; return NULL; }
	offset = dx12Scene.skelVBWriteOffset;
	dx12Scene.skelVBWriteOffset += numVerts;
	*outGpuVA = dx12Scene.skelVB->GetGPUVirtualAddress() + (UINT64)offset * sizeof(dx12WorldVertex_t);
	return dx12Scene.skelVBMapped + (size_t)offset * sizeof(dx12WorldVertex_t);
}

// ---------------------------------------------------------------------------
// DX12_DrawEntity
// ---------------------------------------------------------------------------

/**
 * @brief DX12_DrawEntity
 * @param[in] ent      Scene entity (origin, axis, hModel).
 * @param[in] cbGpuVA  GPU virtual address of the per-entity constant buffer
 *                     (already populated with the entity's model matrix).
 *
 * For each surface: binds the diffuse SRV (root param 1) and an identical
 * fallback for the lightmap slot (root param 2, since models have no
 * lightmaps), binds the VB/IB, and issues DrawIndexedInstanced.
 */
void DX12_DrawEntity(const dx12SceneEntity_t *ent, D3D12_GPU_VIRTUAL_ADDRESS cbGpuVA)
{
	int               idx;
	dx12ModelEntry_t *entry;
	int               s;

	if (!ent)
	{
		return;
	}

	idx = (int)ent->hModel - 1;

	if (idx < 0 || idx >= DX12_MAX_MODELS)
	{
		return;
	}

	entry = &dx12ModelData[idx];
	if (!entry->valid)
	{
		return;
	}

	// Bind constant buffer (root param 0) – already set by caller but be explicit
	dx12.commandList->SetGraphicsRootConstantBufferView(0, cbGpuVA);

	// -------------------------------------------------------------------------
	// MDS – skeletal mesh with embedded animation
	// -------------------------------------------------------------------------
	if (entry->modelType == DX12_MOD_MDS)
	{
		mdsHeader_t  *mds;
		mdsLOD_t     *lod;
		mdsSurface_t *rawSurf;
		refEntity_t   refent;

		if (!entry->rawData)
		{
			return;
		}

		mds     = (mdsHeader_t *)entry->rawData;
		lod     = (mdsLOD_t *)((byte *)mds + mds->ofsSurfaces);
		rawSurf = (mdsSurface_t *)((byte *)lod + lod->ofsSurfaces);

		Com_Memset(&refent, 0, sizeof(refent));
		refent.frame          = ent->frame;
		refent.oldframe       = ent->oldframe;
		refent.backlerp       = ent->backlerp;
		refent.torsoFrame     = ent->torsoFrame;
		refent.oldTorsoFrame  = ent->oldTorsoFrame;
		refent.torsoBacklerp  = ent->torsoBacklerp;
		refent.torsoAxis[0][0] = ent->torsoAxis[0][0]; refent.torsoAxis[0][1] = ent->torsoAxis[0][1]; refent.torsoAxis[0][2] = ent->torsoAxis[0][2];
		refent.torsoAxis[1][0] = ent->torsoAxis[1][0]; refent.torsoAxis[1][1] = ent->torsoAxis[1][1]; refent.torsoAxis[1][2] = ent->torsoAxis[1][2];
		refent.torsoAxis[2][0] = ent->torsoAxis[2][0]; refent.torsoAxis[2][1] = ent->torsoAxis[2][1]; refent.torsoAxis[2][2] = ent->torsoAxis[2][2];

		for (s = 0; s < entry->numSurfaces; s++)
		{
			dx12ModelSurface_t         *ms = &entry->surfaces[s];
			dx12Texture_t              *tex;
			D3D12_VERTEX_BUFFER_VIEW    vbv = {};
			D3D12_INDEX_BUFFER_VIEW     ibv = {};
			D3D12_GPU_DESCRIPTOR_HANDLE srvDiffuse;
			UINT8                      *cpuPtr;
			D3D12_GPU_VIRTUAL_ADDRESS   gpuVA;
			qhandle_t                   resolvedHandle = 0;

			if (!ms->indexBuffer || ms->numIndices == 0 || ms->numVertices == 0)
			{
				rawSurf = (mdsSurface_t *)((byte *)rawSurf + rawSurf->ofsEnd);
				continue;
			}

			if (ent->customSkin)
			{
				resolvedHandle = DX12_GetSkinTexture(ent->customSkin, ms->surfName);
			}
			if (!resolvedHandle)
			{
				resolvedHandle = ms->texHandle;
			}
			if (!resolvedHandle)
			{
				rawSurf = (mdsSurface_t *)((byte *)rawSurf + rawSurf->ofsEnd);
				continue;
			}

			tex = DX12_GetTexture(resolvedHandle);
			if (tex && tex->resource)
			{
				srvDiffuse = tex->gpuHandle;
			}
			else
			{
				srvDiffuse = dx12.srvHeap->GetGPUDescriptorHandleForHeapStart();
			}

			cpuPtr = DX12_AllocSkeletalVerts((int)ms->numVertices, &gpuVA);
			if (!cpuPtr)
			{
				rawSurf = (mdsSurface_t *)((byte *)rawSurf + rawSurf->ofsEnd);
				continue;
			}

			DX12_SkinMDSSurface(mds, rawSurf, &refent, (dx12WorldVertex_t *)cpuPtr);

			{
				D3D12_GPU_DESCRIPTOR_HANDLE srvWhite;
				srvWhite.ptr = dx12.srvHeap->GetGPUDescriptorHandleForHeapStart().ptr;
				dx12.commandList->SetGraphicsRootDescriptorTable(1, srvDiffuse);
				dx12.commandList->SetGraphicsRootDescriptorTable(2, srvWhite);
			}

			vbv.BufferLocation = gpuVA;
			vbv.SizeInBytes    = ms->numVertices * (UINT)sizeof(dx12WorldVertex_t);
			vbv.StrideInBytes  = (UINT)sizeof(dx12WorldVertex_t);

			ibv.BufferLocation = ms->indexBuffer->GetGPUVirtualAddress();
			ibv.SizeInBytes    = ms->numIndices * (UINT)sizeof(int);
			ibv.Format         = DXGI_FORMAT_R32_UINT;

			dx12.commandList->IASetVertexBuffers(0, 1, &vbv);
			dx12.commandList->IASetIndexBuffer(&ibv);
			dx12.commandList->DrawIndexedInstanced(ms->numIndices, 1, 0, 0, 0);

			rawSurf = (mdsSurface_t *)((byte *)rawSurf + rawSurf->ofsEnd);
		}
		return;
	}

	// -------------------------------------------------------------------------
	// MDM – skeletal mesh with separate MDX animation
	// -------------------------------------------------------------------------
	if (entry->modelType == DX12_MOD_MDM)
	{
		mdmHeader_t  *mdm;
		mdmSurface_t *rawSurf;
		refEntity_t   refent;
		mdxHeader_t  *mdxFrame, *mdxOldFrame, *mdxTorso, *mdxOldTorso;

		if (!entry->rawData)
		{
			return;
		}

		mdxFrame    = (ent->frameModel        > 0) ? (mdxHeader_t *)dx12ModelData[(int)ent->frameModel        - 1].rawData : NULL;
		mdxOldFrame = (ent->oldframeModel     > 0) ? (mdxHeader_t *)dx12ModelData[(int)ent->oldframeModel     - 1].rawData : NULL;
		mdxTorso    = (ent->torsoFrameModel   > 0) ? (mdxHeader_t *)dx12ModelData[(int)ent->torsoFrameModel   - 1].rawData : NULL;
		mdxOldTorso = (ent->oldTorsoFrameModel > 0) ? (mdxHeader_t *)dx12ModelData[(int)ent->oldTorsoFrameModel - 1].rawData : NULL;

		if (!mdxFrame || !mdxOldFrame || !mdxTorso || !mdxOldTorso)
		{
			dx12.ri.Printf(PRINT_DEVELOPER, "DX12: MDM entity missing MDX companion(s), skipping draw\n");
			return;
		}

		mdm     = (mdmHeader_t *)entry->rawData;
		rawSurf = (mdmSurface_t *)((byte *)mdm + mdm->ofsSurfaces);

		Com_Memset(&refent, 0, sizeof(refent));
		refent.frame              = ent->frame;
		refent.oldframe           = ent->oldframe;
		refent.backlerp           = ent->backlerp;
		refent.torsoFrame         = ent->torsoFrame;
		refent.oldTorsoFrame      = ent->oldTorsoFrame;
		refent.torsoBacklerp      = ent->torsoBacklerp;
		refent.frameModel         = ent->frameModel;
		refent.oldframeModel      = ent->oldframeModel;
		refent.torsoFrameModel    = ent->torsoFrameModel;
		refent.oldTorsoFrameModel = ent->oldTorsoFrameModel;
		refent.torsoAxis[0][0] = ent->torsoAxis[0][0]; refent.torsoAxis[0][1] = ent->torsoAxis[0][1]; refent.torsoAxis[0][2] = ent->torsoAxis[0][2];
		refent.torsoAxis[1][0] = ent->torsoAxis[1][0]; refent.torsoAxis[1][1] = ent->torsoAxis[1][1]; refent.torsoAxis[1][2] = ent->torsoAxis[1][2];
		refent.torsoAxis[2][0] = ent->torsoAxis[2][0]; refent.torsoAxis[2][1] = ent->torsoAxis[2][1]; refent.torsoAxis[2][2] = ent->torsoAxis[2][2];

		for (s = 0; s < entry->numSurfaces; s++)
		{
			dx12ModelSurface_t         *ms = &entry->surfaces[s];
			dx12Texture_t              *tex;
			D3D12_VERTEX_BUFFER_VIEW    vbv = {};
			D3D12_INDEX_BUFFER_VIEW     ibv = {};
			D3D12_GPU_DESCRIPTOR_HANDLE srvDiffuse;
			UINT8                      *cpuPtr;
			D3D12_GPU_VIRTUAL_ADDRESS   gpuVA;
			qhandle_t                   resolvedHandle = 0;

			if (!ms->indexBuffer || ms->numIndices == 0 || ms->numVertices == 0)
			{
				rawSurf = (mdmSurface_t *)((byte *)rawSurf + rawSurf->ofsEnd);
				continue;
			}

			if (ent->customSkin)
			{
				resolvedHandle = DX12_GetSkinTexture(ent->customSkin, ms->surfName);
			}
			if (!resolvedHandle)
			{
				resolvedHandle = ms->texHandle;
			}
			if (!resolvedHandle)
			{
				rawSurf = (mdmSurface_t *)((byte *)rawSurf + rawSurf->ofsEnd);
				continue;
			}

			tex = DX12_GetTexture(resolvedHandle);
			if (tex && tex->resource)
			{
				srvDiffuse = tex->gpuHandle;
			}
			else
			{
				srvDiffuse = dx12.srvHeap->GetGPUDescriptorHandleForHeapStart();
			}

			cpuPtr = DX12_AllocSkeletalVerts((int)ms->numVertices, &gpuVA);
			if (!cpuPtr)
			{
				rawSurf = (mdmSurface_t *)((byte *)rawSurf + rawSurf->ofsEnd);
				continue;
			}

			DX12_SkinMDMSurface(rawSurf, &refent, mdxFrame, mdxOldFrame, mdxTorso, mdxOldTorso, (dx12WorldVertex_t *)cpuPtr);

			{
				D3D12_GPU_DESCRIPTOR_HANDLE srvWhite;
				srvWhite.ptr = dx12.srvHeap->GetGPUDescriptorHandleForHeapStart().ptr;
				dx12.commandList->SetGraphicsRootDescriptorTable(1, srvDiffuse);
				dx12.commandList->SetGraphicsRootDescriptorTable(2, srvWhite);
			}

			vbv.BufferLocation = gpuVA;
			vbv.SizeInBytes    = ms->numVertices * (UINT)sizeof(dx12WorldVertex_t);
			vbv.StrideInBytes  = (UINT)sizeof(dx12WorldVertex_t);

			ibv.BufferLocation = ms->indexBuffer->GetGPUVirtualAddress();
			ibv.SizeInBytes    = ms->numIndices * (UINT)sizeof(int);
			ibv.Format         = DXGI_FORMAT_R32_UINT;

			dx12.commandList->IASetVertexBuffers(0, 1, &vbv);
			dx12.commandList->IASetIndexBuffer(&ibv);
			dx12.commandList->DrawIndexedInstanced(ms->numIndices, 1, 0, 0, 0);

			rawSurf = (mdmSurface_t *)((byte *)rawSurf + rawSurf->ofsEnd);
		}
		return;
	}

	// -------------------------------------------------------------------------
	// MD3 / MDC – rigid mesh (static GPU VB/IB)
	// Draw each surface through the full material stage pipeline so that alpha
	// testing, animated textures, tcMod scroll/rotate, and multi-stage blend
	// effects work on entity models the same as on world BSP surfaces.
	// -------------------------------------------------------------------------
	for (s = 0; s < entry->numSurfaces; s++)
	{
		dx12ModelSurface_t *ms = &entry->surfaces[s];
		qhandle_t           skinTexHandle = 0;

		if (!ms->vertexBuffer || !ms->indexBuffer || ms->numIndices == 0)
		{
			continue;
		}

		// Resolve skin override (takes priority over material stage textures on
		// stage 0, matching GL R_SetupEntityLighting / skin lookup behaviour).
		if (ent->customSkin)
		{
			skinTexHandle = DX12_GetSkinTexture(ent->customSkin, ms->surfName);
		}

		// If no skin resolved and no material/texture is available, skip the
		// surface rather than drawing it with the white fallback.
		if (!skinTexHandle && !ms->materialHandle && !ms->texHandle)
		{
			dx12.ri.Printf(PRINT_DEVELOPER,
			               "DX12: Missing skin/material for surface '%s' (skin %d)\n",
			               ms->surfName, (int)ent->customSkin);
			continue;
		}

		// Full material-stage draw (alpha test, blend, tcMod, rgbGen/alphaGen …)
		DX12_DrawEntityModelSurface(ms, skinTexHandle, cbGpuVA);
	}
}

// ---------------------------------------------------------------------------
// DX12_ShutdownModels
// ---------------------------------------------------------------------------

/**
 * @brief DX12_ShutdownModels
 *
 * Release all GPU model resources.  Safe to call multiple times.
 */
void DX12_ShutdownModels(void)
{
	int i, s;

	for (i = 0; i < DX12_MAX_MODELS; i++)
	{
		if (!dx12ModelData[i].valid)
		{
			continue;
		}

		for (s = 0; s < dx12ModelData[i].numSurfaces; s++)
		{
			if (dx12ModelData[i].surfaces[s].vertexBuffer)
			{
				dx12ModelData[i].surfaces[s].vertexBuffer->Release();
				dx12ModelData[i].surfaces[s].vertexBuffer = NULL;
			}
			if (dx12ModelData[i].surfaces[s].indexBuffer)
			{
				dx12ModelData[i].surfaces[s].indexBuffer->Release();
				dx12ModelData[i].surfaces[s].indexBuffer = NULL;
			}
		}

		dx12ModelData[i].valid       = qfalse;
		dx12ModelData[i].numSurfaces = 0;

		// Free MD3 tag data
		if (dx12ModelData[i].tags)
		{
			dx12.ri.Free(dx12ModelData[i].tags);
			dx12ModelData[i].tags     = NULL;
		}
		dx12ModelData[i].numTags   = 0;
		dx12ModelData[i].numFrames = 0;

		// Free skeletal raw data (MDS / MDM / MDX)
		if (dx12ModelData[i].rawData)
		{
			DX12_FreeSkeletal(dx12ModelData[i].rawData);
			dx12ModelData[i].rawData     = NULL;
			dx12ModelData[i].rawDataSize = 0;
		}
		dx12ModelData[i].modelType = DX12_MOD_UNKNOWN;
	}
}

// ---------------------------------------------------------------------------
// DX12_LerpTag
// ---------------------------------------------------------------------------

/**
 * @brief DX12_LerpTag – mirror of GL's R_LerpTag.
 *
 * Dispatches to the appropriate tag-lookup path based on the model type:
 *   - MD3: interpolate between two MD3 tag frames.
 *   - MDS: compute bone hierarchy and extract tag bone orientation.
 *   - MDM: compute bone hierarchy from MDX companion and extract tag orientation.
 *
 * @param[out] tag       Receives the interpolated orientation.
 * @param[in]  refent    Entity to query (hModel, frame, oldframe, backlerp…).
 * @param[in]  tagName   Tag name to search for (case-sensitive).
 * @param[in]  startIndex Start scanning from this index (for duplicate names).
 * @return     Tag index found, or -1 on failure.
 */
int DX12_LerpTag(orientation_t *tag, const refEntity_t *refent,
                 const char *tagName, int startIndex)
{
	dx12ModelEntry_t *entry;
	md3Tag_t         *startTag;
	md3Tag_t         *endTag;
	int               idx;
	int               startFrame;
	int               endFrame;
	float             frontLerp;
	float             backLerp;
	int               i;

	if (!tag || !refent || !tagName)
	{
		return -1;
	}

	idx = (int)refent->hModel - 1;
	if (idx < 0 || idx >= DX12_MAX_MODELS)
	{
		AxisClear(tag->axis);
		VectorClear(tag->origin);
		return -1;
	}

	entry = &dx12ModelData[idx];

	// ---------------------------------------------------------------------------
	// MDS: bone-based tag lookup (player bodies, e.g. tag_footleft)
	// ---------------------------------------------------------------------------
	if (entry->modelType == DX12_MOD_MDS && entry->rawData)
	{
		return DX12_GetBoneTagMDS(tag, (mdsHeader_t *)entry->rawData, refent, tagName, startIndex);
	}

	// ---------------------------------------------------------------------------
	// MDM: bone-based tag lookup with MDX companion animation data
	// ---------------------------------------------------------------------------
	if (entry->modelType == DX12_MOD_MDM && entry->rawData)
	{
		// Resolve the four MDX frame headers from the DX12 model registry.
		// refent->frameModel, oldframeModel, torsoFrameModel, oldTorsoFrameModel
		// are handles returned by RE_DX12_RegisterModel for the companion MDX files.
		mdxHeader_t *mdxFrame     = NULL;
		mdxHeader_t *mdxOldFrame  = NULL;
		mdxHeader_t *mdxTorso     = NULL;
		mdxHeader_t *mdxOldTorso  = NULL;

		int fi  = (int)refent->frameModel - 1;
		int ofi = (int)refent->oldframeModel - 1;
		int ti  = (int)refent->torsoFrameModel - 1;
		int oti = (int)refent->oldTorsoFrameModel - 1;

		if (fi >= 0 && fi < DX12_MAX_MODELS  && dx12ModelData[fi].modelType  == DX12_MOD_MDX)
		{
			mdxFrame = (mdxHeader_t *)dx12ModelData[fi].rawData;
		}
		if (ofi >= 0 && ofi < DX12_MAX_MODELS && dx12ModelData[ofi].modelType == DX12_MOD_MDX)
		{
			mdxOldFrame = (mdxHeader_t *)dx12ModelData[ofi].rawData;
		}
		if (ti >= 0 && ti < DX12_MAX_MODELS  && dx12ModelData[ti].modelType  == DX12_MOD_MDX)
		{
			mdxTorso = (mdxHeader_t *)dx12ModelData[ti].rawData;
		}
		if (oti >= 0 && oti < DX12_MAX_MODELS && dx12ModelData[oti].modelType == DX12_MOD_MDX)
		{
			mdxOldTorso = (mdxHeader_t *)dx12ModelData[oti].rawData;
		}

		// If any MDX header is missing, fall back to the same as the frame one
		// (prevents a null-pointer crash; accuracy degrades gracefully).
		// If all four handles were invalid, mdxFrame remains NULL and we fall
		// through to the guard below which returns -1 cleanly.
		if (!mdxFrame)     { mdxFrame     = mdxOldFrame ? mdxOldFrame : (mdxTorso ? mdxTorso : mdxOldTorso); }
		if (!mdxOldFrame)  { mdxOldFrame  = mdxFrame; }
		if (!mdxTorso)     { mdxTorso     = mdxFrame; }
		if (!mdxOldTorso)  { mdxOldTorso  = mdxTorso; }

		if (!mdxFrame)
		{
			// No valid MDX companion was registered – cannot compute tag.
			AxisClear(tag->axis);
			VectorClear(tag->origin);
			return -1;
		}

		return DX12_GetBoneTagMDM(tag, (mdmHeader_t *)entry->rawData,
		                          mdxFrame, mdxOldFrame, mdxTorso, mdxOldTorso,
		                          refent, tagName, startIndex);
	}

	// ---------------------------------------------------------------------------
	// MD3: interpolate between per-frame tag arrays
	// ---------------------------------------------------------------------------
	if (!entry->valid || !entry->tags || entry->numTags <= 0 || entry->numFrames <= 0)
	{
		AxisClear(tag->axis);
		VectorClear(tag->origin);
		return -1;
	}

	// Clamp frames to valid range
	startFrame = refent->oldframe;
	endFrame   = refent->frame;
	if (startFrame >= entry->numFrames) { startFrame = entry->numFrames - 1; }
	if (startFrame < 0)                 { startFrame = 0; }
	if (endFrame >= entry->numFrames)   { endFrame = entry->numFrames - 1; }
	if (endFrame < 0)                   { endFrame = 0; }

	backLerp  = refent->backlerp;
	frontLerp = 1.0f - backLerp;

	// Find the tag by name (scanning from startIndex)
	startTag = NULL;
	endTag   = NULL;
	idx      = -1;

	for (i = startIndex; i < entry->numTags; i++)
	{
		md3Tag_t *t = &entry->tags[startFrame * entry->numTags + i];

		if (!strcmp(t->name, tagName))
		{
			startTag = t;
			endTag   = &entry->tags[endFrame * entry->numTags + i];
			idx      = i;
			break;
		}
	}

	if (!startTag || !endTag)
	{
		AxisClear(tag->axis);
		VectorClear(tag->origin);
		return -1;
	}

	// Interpolate origin and axis
	for (i = 0; i < 3; i++)
	{
		tag->origin[i]  = startTag->origin[i] * backLerp + endTag->origin[i] * frontLerp;
		tag->axis[0][i] = startTag->axis[0][i] * backLerp + endTag->axis[0][i] * frontLerp;
		tag->axis[1][i] = startTag->axis[1][i] * backLerp + endTag->axis[1][i] * frontLerp;
		tag->axis[2][i] = startTag->axis[2][i] * backLerp + endTag->axis[2][i] * frontLerp;
	}

	VectorNormalize(tag->axis[0]);
	VectorNormalize(tag->axis[1]);
	VectorNormalize(tag->axis[2]);

	return idx;
}

#endif // _WIN32
