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
 * @file dx12_world.h
 * @brief DX12 world (BSP) geometry – types and public API.
 *
 * Declares the data structures used to hold parsed BSP world geometry and
 * the two public entry points: DX12_LoadWorld() and DX12_ShutdownWorld().
 */

#ifndef DX12_WORLD_H
#define DX12_WORLD_H

#ifdef _WIN32

#include "tr_dx12_local.h"

extern "C" {
#include "../qcommon/qfiles.h"   // dnode_t, dleaf_t, dplane_t, LUMP_* defines
}

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/** Maximum simultaneously registered BSP shader entries. */
#define DX12_MAX_WORLD_SHADERS   0x400   ///< MAX_MAP_SHADERS
/** Maximum BSP fog volumes (index 0 is reserved as "no fog"). */
#define DX12_MAX_WORLD_FOGS      0x100   ///< MAX_MAP_FOGS
/** Maximum BSP submodels (entities + worldmodel at index 0). */
#define DX12_MAX_WORLD_MODELS    0x800   ///< MAX_MAP_MODELS
/** Maximum draw surfaces produced from LUMP_SURFACES. */
#define DX12_MAX_DRAW_SURFS      0x20000 ///< MAX_MAP_DRAW_SURFS
/** Maximum simultaneously loaded lightmap textures. */
#define DX12_MAX_LIGHTMAPS       256

// ---------------------------------------------------------------------------
// BSP node/leaf – simplified version for PVS traversal only
// ---------------------------------------------------------------------------

/**
 * @struct dx12BspNode_t
 * @brief Minimal BSP node or leaf used only for PointInLeaf traversal.
 *
 * For nodes: contents == -1, planeIdx >= 0, children[0/1] are valid.
 * For leaves: contents >= 0 (cluster, -1 for opaque), planeIdx == -1.
 */
typedef struct
{
	int contents;    ///< -1 = decision node; >= 0 = leaf (cluster index, or -1)
	int cluster;     ///< Leaf cluster (-1 for opaque/solid leaves)
	int planeIdx;    ///< Index into dx12World.bspPlanes (nodes only)
	int children[2]; ///< Child indices into bspNodes[] (nodes only); negative = -(leaf+1)
} dx12BspNode_t;

// ---------------------------------------------------------------------------
// Vertex layout
// ---------------------------------------------------------------------------

/**
 * @struct dx12WorldVertex_t
 * @brief One vertex of static world geometry uploaded to the GPU.
 *
 * Matches the expected input layout for the world HLSL vertex shader:
 *   POSITION (xyz), TEXCOORD0 (diffuse st), TEXCOORD1 (lightmap st),
 *   NORMAL (xyz), COLOR (rgba normalized).
 */
typedef struct
{
	float xyz[3];      ///< World-space position
	float st[2];       ///< Diffuse texture UV
	float lm[2];       ///< Lightmap UV
	float normal[3];   ///< Vertex normal
	float color[4];    ///< Vertex color (RGBA in [0, 1])
} dx12WorldVertex_t;

// ---------------------------------------------------------------------------
// Draw surface
// ---------------------------------------------------------------------------

/**
 * @struct dx12DrawSurf_t
 * @brief One draw call entry in the world draw-surface list.
 *
 * The list is sorted so opaque surfaces come first (sorted by material),
 * then fog volumes, then sky surfaces, then translucent surfaces.
 */
typedef struct
{
	qhandle_t materialHandle; ///< DX12_RegisterMaterial() handle
	int       fogIndex;       ///< BSP fog-volume index (-1 = none)
	int       lightmapIndex;  ///< Index into dx12World.lightmapHandles[] (-1 = no lightmap)
	int       firstVertex;    ///< First vertex in the world VB
	int       numVertices;    ///< Number of vertices
	int       firstIndex;     ///< First index in the world IB
	int       numIndexes;     ///< Number of indices
	int       surfaceType;    ///< mapSurfaceType_t from BSP (MST_PLANAR etc.)
	qboolean  isSky;          ///< Copied from material surfaceparm sky
	qboolean  isTranslucent;  ///< Copied from material surfaceparm trans
	qboolean  isFog;          ///< Copied from material surfaceparm fog
} dx12DrawSurf_t;

// ---------------------------------------------------------------------------
// Fog volume
// ---------------------------------------------------------------------------

/**
 * @struct dx12WorldFog_t
 * @brief Axis-aligned bounding box for one BSP fog volume.
 *
 * Index 0 is always "no fog" (mins = MAX_WORLD_COORD, maxs = MIN_WORLD_COORD).
 * Indices 1…numFogs correspond to the dfog_t entries in LUMP_FOGS.
 */
typedef struct
{
	float     mins[3];        ///< AABB minimum corner
	float     maxs[3];        ///< AABB maximum corner
	qhandle_t materialHandle; ///< Material for this fog volume (may be 0)
	int       originalBrush;  ///< BSP brush number (-1 = global fog)
} dx12WorldFog_t;

// ---------------------------------------------------------------------------
// Submodel
// ---------------------------------------------------------------------------

/**
 * @struct dx12WorldModel_t
 * @brief One BSP submodel (worldspawn = index 0, inline entities 1…n).
 */
typedef struct
{
	float mins[3];       ///< AABB minimum
	float maxs[3];       ///< AABB maximum
	int   firstSurface;  ///< Index into dx12World.drawSurfs[]
	int   numSurfaces;   ///< Number of draw surfaces in this model
} dx12WorldModel_t;

// ---------------------------------------------------------------------------
// World state
// ---------------------------------------------------------------------------

/**
 * @struct dx12World_t
 * @brief All world geometry state for one loaded BSP map.
 *
 * Cleared by DX12_ShutdownWorld() and populated by DX12_LoadWorld().
 */
typedef struct
{
	char name[MAX_QPATH];     ///< BSP file path (e.g. "maps/et_ice.bsp")

	// Material + lightmap handles (one per BSP shader/lightmap entry)
	qhandle_t shaderHandles[DX12_MAX_WORLD_SHADERS]; ///< Per dshader_t entry
	int       numShaders;                            ///< Count from LUMP_SHADERS

	qhandle_t lightmapHandles[DX12_MAX_LIGHTMAPS]; ///< Per lightmap texture entry
	int       numLightmaps;                        ///< Count from LUMP_LIGHTMAPS

	// GPU geometry buffers
	ID3D12Resource *vertexBuffer; ///< Default-heap VB (dx12WorldVertex_t[])
	ID3D12Resource *indexBuffer;  ///< Default-heap IB (int[])
	UINT            numVertices;  ///< Total vertices uploaded
	UINT            numIndexes;   ///< Total indices uploaded

	// CPU-side draw list (sorted)
	dx12DrawSurf_t drawSurfs[DX12_MAX_DRAW_SURFS]; ///< Sorted draw surface list
	int            numDrawSurfs;                    ///< Active entries

	// Fog volumes (index 0 = no fog)
	dx12WorldFog_t fogs[DX12_MAX_WORLD_FOGS]; ///< Fog volume AABB array
	int            numFogs;                   ///< Count including slot 0

	// Submodels
	dx12WorldModel_t models[DX12_MAX_WORLD_MODELS]; ///< Submodel list
	int              numModels;                     ///< Active entries

	// BSP entity string (LUMP_ENTITIES) – used by RE_DX12_GetEntityToken
	char       *entityString;     ///< Allocated copy of the raw entity lump text
	char *entityParsePoint; ///< Current parse position within entityString

	// BSP node/leaf/visibility data – used by DX12_inPVS
	dx12BspNode_t *bspNodes;      ///< Combined node + leaf array (malloc'd)
	int            numBspNodes;   ///< Total node + leaf count
	int            numDecisionNodes; ///< Number of decision nodes (nodes, not leaves)
	dplane_t      *bspPlanes;     ///< Plane array (malloc'd)
	int            numBspPlanes;  ///< Plane count

	byte *vis;           ///< Raw PVS data (malloc'd; NULL if no vis data)
	byte *novis;         ///< All-0xFF row for "visible" fallback (malloc'd)
	int   numClusters;   ///< Number of PVS clusters
	int   clusterBytes;  ///< Bytes per cluster row in vis[]

	qboolean loaded; ///< qtrue after a successful DX12_LoadWorld() call
} dx12World_t;

extern dx12World_t dx12World;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * @brief DX12_LoadWorld
 * @param[in] name  BSP file path (e.g. "maps/et_ice.bsp").
 *
 * Parses BSP lumps, uploads static geometry to GPU buffers, and builds a
 * sorted draw-surface list.  Calls DX12_ShutdownWorld() first if a world
 * is already loaded.
 */
void DX12_LoadWorld(const char *name);

/**
 * @brief DX12_ShutdownWorld
 *
 * Releases all world GPU resources and clears dx12World.  Safe to call
 * at any time; does nothing if no world is currently loaded.
 */
void DX12_ShutdownWorld(void);

/**
 * @brief DX12_inPVS
 *
 * Returns qtrue if the two world-space points p1 and p2 are in the same or
 * mutually-visible PVS clusters (mirrors GL's R_inPVS).
 *
 * @param[in] p1  First world-space point.
 * @param[in] p2  Second world-space point.
 * @return         qtrue when p2 is potentially visible from p1.
 */
qboolean DX12_inPVS(const vec3_t p1, const vec3_t p2);

#endif // _WIN32
#endif // DX12_WORLD_H
