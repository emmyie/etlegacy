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

#pragma once

#ifdef _WIN32

#include "tr_dx12_local.h"
#include "dx12_scene.h"      // dx12SceneEntity_t
#include "dx12_skeletal.h"   // dx12ModelType_t

extern "C" {
#include "../qcommon/qfiles.h"  // md3Tag_t, md3Header_t, mdsHeader_t, mdmHeader_t, mdxHeader_t
}

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
	ID3D12Resource *vertexBuffer;          ///< Default-heap VB (dx12WorldVertex_t[])
	ID3D12Resource *indexBuffer;           ///< Default-heap IB (UINT32[])
	UINT            numVertices;
	UINT            numIndices;
	qhandle_t       texHandle;             ///< Diffuse texture handle (raw image, from DX12_RegisterTexture)
	qhandle_t       materialHandle;        ///< Material handle (from DX12_RegisterMaterial) for multi-stage draw
	char            surfName[MAX_QPATH];   ///< MD3 surface name (lower-cased), used for skin lookup
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
	qboolean           isBad;   ///< qtrue when every load attempt failed – mirrors GL MOD_BAD.
	                            ///<   Dedup scan returns 0 for bad slots so callers don't use stale handles.

	// MD3 tag data (all frames × all tags) – used by DX12_LerpTag.
	// Allocated by DX12_LoadMD3, freed by DX12_ShutdownModels.
	md3Tag_t *tags;     ///< Flattened array: tags[frame * numTags + tagIdx]
	int       numTags;  ///< Tags per frame
	int       numFrames;///< Total animation frames

	// Model type – controls which tag/bounds path to use.
	dx12ModelType_t modelType;

	// Raw skeletal data – malloc'd heap copy of the on-disk binary.
	// Interpretation depends on modelType:
	//   DX12_MOD_MDS → mdsHeader_t*  (embedded bone + frame data)
	//   DX12_MOD_MDM → mdmHeader_t*  (tag data; animation via companion MDX)
	//   DX12_MOD_MDX → mdxHeader_t*  (pure animation data for MDM)
	// Freed by DX12_ShutdownModels.
	void *rawData;
	int   rawDataSize;

	// LOD slot map (MD3 only; populated by RE_DX12_RegisterModelAllLODs).
	// lodSlots[i] is the dx12ModelData[] index (handle - 1) for LOD level i.
	// After RegisterModelAllLODs all MD3_MAX_LODS entries are valid (≥ 0).
	// Plain RE_DX12_RegisterModel leaves numLods == 0 and lodSlots zeroed.
	int numLods;                    ///< Distinct LODs registered (0 unless RegisterModelAllLODs was used)
	int lodSlots[MD3_MAX_LODS];     ///< Per-LOD data-slot index; –1 when not yet set
} dx12ModelEntry_t;

/** Parallel to dx12ModelNames[] in tr_dx12_main.cpp. */
extern dx12ModelEntry_t dx12ModelData[DX12_MAX_MODELS];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * @brief Pre-build a static index buffer for a skeletal surface (MDS or MDM).
 * The index data is an array of numTriangles*3 int values.
 * Call at model load time; stores IB in ms->indexBuffer and sets ms->numIndices.
 * @return qtrue on success.
 */
qboolean DX12_BuildSkeletalSurfaceIB(dx12ModelSurface_t *ms, const int *triIndexes, int numTriangles);

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
 * @brief Attempt to load a Compressed MD3 (MDC) model from the VFS into GPU buffers.
 *
 * MDC is the RtCW/ET compressed variant of MD3, used for gibs, shells, and
 * weapon effects.  Frame-0 base-frame geometry is decoded and uploaded
 * identically to DX12_LoadMD3 (same GPU draw path).
 *
 * @param slot  Registry slot index (handle - 1).
 * @param name  Game-path of the .mdc file.
 * @return      qtrue when at least one surface was uploaded successfully.
 */
qboolean DX12_LoadMDC(int slot, const char *name);

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
 * @brief Return the BSP submodel index for a brush model handle, or -1.
 *
 * Returns the integer N from a handle that was registered as "*N".
 * Returns -1 for regular 3D model handles, NULL handles, or out-of-range
 * handles.  Used by DX12_RenderScene to route brush model entities through
 * the BSP world-geometry path rather than the per-model VB/IB path.
 */
int DX12_GetBrushSubmodelIdx(qhandle_t hModel);

/**
 * @brief Release all GPU model resources (VBs and IBs).
 *
 * Should be called from R_DX12_Shutdown() before releasing the device.
 */
void DX12_ShutdownModels(void);

/**
 * @brief Interpolate an MD3 tag between two animation frames.
 *
 * Mirrors GL's R_LerpTag for the MD3 (MOD_MESH) case.  The caller provides
 * the refEntity_t so this function can read the frame indices and backlerp.
 *
 * @param[out] tag       Receives the interpolated orientation.
 * @param[in]  refent    Entity whose model and frame data to use.
 * @param[in]  tagName   Tag to find (case-sensitive).
 * @param[in]  startIndex Start search at this tag index (for duplicate names).
 * @return     Tag index found, or -1 on failure.
 */
int DX12_LerpTag(orientation_t *tag, const refEntity_t *refent,
                 const char *tagName, int startIndex);

/**
 * @brief Concatenate a parent world matrix with a tag local matrix to produce
 *        the child model's world matrix.
 *
 * Usage in DX12_DrawEntity (or the scene entity loop) when a child model
 * must be positioned relative to a parent model tag:
 *
 *   // 1. Build parent world matrix (pre-transpose form)
 *   float parentRaw[4][4];
 *   BuildModelMatrix(parentRaw, parentEnt->origin, parentEnt->axis);
 *
 *   // 2. Find the tag on the parent model
 *   orientation_t tagOrient;
 *   DX12_LerpTag(&tagOrient, parentRefEnt, "tag_torso", 0);
 *
 *   // 3. Synthesise an md3Tag_t from the orientation_t
 *   md3Tag_t tag;
 *   VectorCopy(tagOrient.origin, tag.origin);
 *   VectorCopy(tagOrient.axis[0], tag.axis[0]);
 *   VectorCopy(tagOrient.axis[1], tag.axis[1]);
 *   VectorCopy(tagOrient.axis[2], tag.axis[2]);
 *
 *   // 4. Compute child world matrix and transpose into CB
 *   float childRaw[4][4];
 *   DX12_ApplyTagTransform(childRaw, parentRaw, &tag);
 *   Mat4Transpose(childCB.modelMatrix, childRaw);
 *
 * In practice ET's cgame resolves tag attachment before submitting entities
 * to the renderer (trap_R_LerpTag → AxisMultiply / VectorAdd on the client
 * side), so each entity already has pre-baked world-space origin/axis by the
 * time it reaches DX12_DrawEntity.  DX12_ApplyTagTransform is provided for
 * renderer-side assembly (e.g. inline sub-model attachment or future work).
 *
 * @param[out] out     Child world matrix (row-major, pre-transpose).
 * @param[in]  parent  Parent world matrix (row-major, pre-transpose).
 * @param[in]  tag     Tag in parent-local Q3 space.
 */
void DX12_ApplyTagTransform(float out[4][4], const float parent[4][4], const md3Tag_t *tag);

/**
 * @brief Draw one MD3/MDC model surface through the full material stage pipeline
 * with entity lighting (isEntity=1, useLightmap=0).
 *
 * Iterates all active stages of the surface's material applying tcMod chains,
 * rgbGen/alphaGen, blend-mode PSO selection, and per-stage SRV binding.
 * Falls back to a single opaque draw when no material stages are present.
 *
 * Defined in dx12_scene.cpp (needs access to static scene helpers).
 *
 * @param[in] ms             Surface descriptor: VB, IB, materialHandle, texHandle.
 * @param[in] skinTexHandle  Resolved skin texture (0 = use material / embedded).
 * @param[in] cbGpuVA        Per-entity constant-buffer GPU virtual address.
 */
void DX12_DrawEntityModelSurface(const dx12ModelSurface_t *ms,
                                 qhandle_t skinTexHandle,
                                 D3D12_GPU_VIRTUAL_ADDRESS cbGpuVA);

#endif // _WIN32
