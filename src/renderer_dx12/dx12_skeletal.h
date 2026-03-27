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
 * @file dx12_skeletal.h
 * @brief DX12 skeletal model subsystem – MDS / MDM / MDX loading and tag lookup.
 *
 * Skeletal models (MDS / MDM + MDX) are stored as raw heap copies of the
 * on-disk binary data.  The bone animation math is a self-contained port of
 * the GL renderer's tr_animation_mds.c and tr_animation_mdm.c, adapted to
 * run without any GL-renderer-specific globals (backEnd, viewParms, etc.).
 *
 * Only the tag-lookup path (LerpTag) is implemented; GPU skinned rendering
 * is not yet supported and the draw call is silently skipped.
 */

#ifndef DX12_SKELETAL_H
#define DX12_SKELETAL_H

#ifdef _WIN32

#include "tr_dx12_local.h"

extern "C" {
#include "../qcommon/qfiles.h"   // mdsHeader_t, mdmHeader_t, mdxHeader_t
#include "../renderercommon/tr_types.h"  // refEntity_t, orientation_t
}

// ---------------------------------------------------------------------------
// Model type enum (stored in dx12ModelEntry_t::modelType)
// ---------------------------------------------------------------------------

typedef enum
{
	DX12_MOD_UNKNOWN = 0,
	DX12_MOD_MD3,       ///< MD3 rigid mesh (existing support)
	DX12_MOD_MDS,       ///< MDS skeletal mesh+animation (player models)
	DX12_MOD_MDM,       ///< MDM skeletal mesh-only (links to MDX for animation)
	DX12_MOD_MDX,       ///< MDX animation-only companion to MDM
} dx12ModelType_t;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * @brief Load a raw MDS file into memory for tag and bounds queries.
 *
 * The raw binary is heap-allocated and must be freed via DX12_FreeSkeletal().
 *
 * @param name   Game-path of the .mds file.
 * @param outData  Receives a pointer to the heap-allocated mdsHeader_t.
 * @param outSize  Receives the number of bytes allocated.
 * @return qtrue on success.
 */
qboolean DX12_LoadMDS(const char *name, void **outData, int *outSize);

/**
 * @brief Load a raw MDX file into memory for MDM animation queries.
 *
 * @param name   Game-path of the .mdx file.
 * @param outData  Receives a pointer to the heap-allocated mdxHeader_t.
 * @param outSize  Receives the number of bytes allocated.
 * @return qtrue on success.
 */
qboolean DX12_LoadMDX(const char *name, void **outData, int *outSize);

/**
 * @brief Load a raw MDM file into memory for tag queries.
 *
 * @param name   Game-path of the .mdm file.
 * @param outData  Receives a pointer to the heap-allocated mdmHeader_t.
 * @param outSize  Receives the number of bytes allocated.
 * @return qtrue on success.
 */
qboolean DX12_LoadMDM(const char *name, void **outData, int *outSize);

/**
 * @brief Free skeletal raw data previously allocated by DX12_LoadMDS/MDX/MDM.
 * @param data  Pointer returned by the loader (may be NULL).
 */
void DX12_FreeSkeletal(void *data);

/**
 * @brief Interpolate an MDS tag between two animation frames.
 *
 * Mirrors GL's R_GetBoneTag.  Computes the bone chain up to the root and
 * extracts the orientation of the named tag bone.
 *
 * @param[out] outTag    Receives the interpolated orientation.
 * @param[in]  mds       Raw MDS data (from DX12_LoadMDS).
 * @param[in]  refent    Entity to query (frame, oldframe, backlerp, torsoFrame…).
 * @param[in]  tagName   Tag to find (case-sensitive).
 * @param[in]  startTagIndex  Start scan at this tag index.
 * @return Tag index found, or -1 on failure.
 */
int DX12_GetBoneTagMDS(orientation_t *outTag, mdsHeader_t *mds,
                       const refEntity_t *refent, const char *tagName, int startTagIndex);

/**
 * @brief Interpolate an MDM tag using an MDX animation companion.
 *
 * Mirrors GL's R_MDM_GetBoneTag.
 *
 * @param[out] outTag    Receives the interpolated orientation.
 * @param[in]  mdm       Raw MDM data (from DX12_LoadMDM).
 * @param[in]  mdxFrame  MDX data for refent->frameModel.
 * @param[in]  mdxOldFrame  MDX data for refent->oldframeModel.
 * @param[in]  mdxTorso     MDX data for refent->torsoFrameModel.
 * @param[in]  mdxOldTorso  MDX data for refent->oldTorsoFrameModel.
 * @param[in]  refent    Entity to query.
 * @param[in]  tagName   Tag to find (case-sensitive).
 * @param[in]  startTagIndex  Start scan at this tag index.
 * @return Tag index found, or -1 on failure.
 */
int DX12_GetBoneTagMDM(orientation_t *outTag, mdmHeader_t *mdm,
                       mdxHeader_t *mdxFrame, mdxHeader_t *mdxOldFrame,
                       mdxHeader_t *mdxTorso, mdxHeader_t *mdxOldTorso,
                       const refEntity_t *refent, const char *tagName, int startTagIndex);

#endif // _WIN32
#endif // DX12_SKELETAL_H
