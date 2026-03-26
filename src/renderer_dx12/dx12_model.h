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
 * @file dx12_model.h
 * @brief DX12 MD3 model loader – GPU mesh upload and entity draw dispatch.
 *
 * Handles parsing of .md3 files, uploading frame-0 geometry to default-heap
 * GPU buffers, and issuing indexed draw calls during the scene entity pass.
 * MDX/MDM skeletal models are recognised by their file extension but not
 * decoded – the handle is still registered so the animation system can key
 * animation pools, but no geometry is rendered.
 *
 * The model slot index exactly mirrors the dx12ModelNames[] array in
 * tr_dx12_main.cpp: handle = slot + 1, so models[handle - 1] retrieves the
 * GPU data for a given handle.
 */

#ifndef DX12_MODEL_H
#define DX12_MODEL_H

#ifdef _WIN32

#include "tr_dx12_local.h"
#include "dx12_scene.h"   // dx12SceneEntity_t

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/** Maximum GPU surfaces per model (one per MD3 surface). */
#define DX12_MAX_MODEL_SURFACES 32

/**
 * Must match DX12_MAX_MOD_KNOWN in tr_dx12_main.cpp so that
 * dx12ModelData[handle - 1] is always a valid index.
 */
#define DX12_MAX_MODELS 2048

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/**
 * @struct dx12ModelSurface_t
 * @brief GPU buffers for a single MD3 surface.
 */
typedef struct
{
	ID3D12Resource *vertexBuffer; ///< Default-heap VB (dx12WorldVertex_t[])
	ID3D12Resource *indexBuffer;  ///< Default-heap IB (UINT32[])
	UINT            numVertices;
	UINT            numIndices;
	qhandle_t       texHandle;    ///< Diffuse texture / material handle
} dx12ModelSurface_t;

/**
 * @struct dx12ModelEntry_t
 * @brief Loaded model data for one registry slot.
 *
 * Indexed by handle - 1 (parallel to dx12ModelNames[] in tr_dx12_main.cpp).
 */
typedef struct
{
	dx12ModelSurface_t surfaces[DX12_MAX_MODEL_SURFACES];
	int                numSurfaces;
	float              mins[3]; ///< Frame-0 AABB minimum (local space)
	float              maxs[3]; ///< Frame-0 AABB maximum (local space)
	qboolean           valid;   ///< qtrue when GPU buffers are ready
} dx12ModelEntry_t;

/** Parallel to dx12ModelNames[] in tr_dx12_main.cpp. */
extern dx12ModelEntry_t dx12ModelData[DX12_MAX_MODELS];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * @brief Attempt to load an MD3 model from the VFS into GPU buffers.
 *
 * @param slot  Registry slot index (handle - 1).
 * @param name  Game-path of the .md3 file.
 * @return      qtrue when at least one surface was uploaded successfully.
 *
 * Returns qfalse without printing errors when the file is not found or has
 * an unrecognised format (e.g. MDX/MDM skeletal models are silently skipped).
 */
qboolean DX12_LoadMD3(int slot, const char *name);

/**
 * @brief Draw all GPU surfaces of an entity's model.
 *
 * Assumes the 3D root signature (rootSignature3D) and PSO (pso3D) are already
 * bound.  Binds the entity's per-surface VB/IB and diffuse texture, then
 * issues one DrawIndexedInstanced call per surface.
 *
 * @param ent      Scene entity snapshot (origin, axis, hModel).
 * @param cbGpuVA  GPU virtual address of the already-populated constant-buffer
 *                 slot (dx12SceneConstants_t with this entity's model matrix).
 */
void DX12_DrawEntity(const dx12SceneEntity_t *ent, D3D12_GPU_VIRTUAL_ADDRESS cbGpuVA);

/**
 * @brief Release all GPU model resources (VBs and IBs).
 *
 * Should be called from R_DX12_Shutdown() before releasing the device.
 */
void DX12_ShutdownModels(void);

#endif // _WIN32
#endif // DX12_MODEL_H
