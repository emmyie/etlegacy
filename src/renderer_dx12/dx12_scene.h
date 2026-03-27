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
 * @file dx12_scene.h
 * @brief DX12 3D scene rendering – types and public API.
 *
 * Declares the structures, constants, and functions used by DX12_RenderScene()
 * to draw world BSP surfaces and ref-entities each frame.
 */

#ifndef DX12_SCENE_H
#define DX12_SCENE_H

#ifdef _WIN32

#include "tr_dx12_local.h"
#include "dx12_world.h"             // dx12WorldVertex_t (used by poly buffer)
#include "../renderercommon/tr_public.h"

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/** Maximum DX12 dynamic lights per frame – matches GL renderer's MAX_DLIGHTS. */
#define DX12_MAX_DLIGHTS MAX_DLIGHTS

/** Maximum DX12 coronas per frame – matches GL renderer's MAX_CORONAS. */
#define DX12_MAX_CORONAS MAX_CORONAS

/**
 * @struct dx12DLight_t
 * @brief Per-frame dynamic light entry (mirrors GL dlight_t without GL internals).
 */
typedef struct
{
	vec3_t origin;               ///< World-space position
	vec3_t transformed;          ///< Origin in local (entity) coordinate system
	vec3_t color;                ///< Normalised RGB [0,1]
	float  radius;               ///< Influence radius in world units
	float  radiusInverseCubed;   ///< 1/(radius^3) – attenuation optimisation, matches GL dlight_t
	float  intensity;            ///< Light strength (1.0 = fullbright)
	int    flags;                ///< REF_*_DLIGHT flags
} dx12DLight_t;

/**
 * @struct dx12Corona_t
 * @brief Per-frame corona / flare entry (mirrors GL corona_t).
 */
typedef struct
{
	vec3_t   origin;    ///< World-space position
	float    color[3];  ///< Normalised RGB [0,1]
	float    scale;     ///< Scale relative to r_flareSize
	int      id;        ///< Unique id for fading continuity
	qboolean visible;   ///< qtrue when the corona is currently visible
} dx12Corona_t;

// ---------------------------------------------------------------------------
// Decal (ProjectDecal / ClearDecals)
// ---------------------------------------------------------------------------

/** Maximum vertices stored per decal polygon (capped to fit memory budget). */
#define DX12_MAX_DECAL_VERTS 8

/** Maximum persistent decals queued for rendering before an explicit ClearDecals. */
#define DX12_MAX_DECALS 128

/**
 * @struct dx12Decal_t
 * @brief One world-space decal entry (simplified first-pass: stores the source
 *        polygon as-is; full BSP-clip projection is a TODO).
 */
typedef struct
{
	qhandle_t  hShader;                       ///< Material handle
	polyVert_t verts[DX12_MAX_DECAL_VERTS];   ///< World-space polygon vertices
	int        numVerts;                       ///< Actual vertex count (≤ DX12_MAX_DECAL_VERTS)
	float      color[4];                       ///< Tint RGBA [0,1]
	int        lifeTime;                       ///< Lifetime hint in ms (0 = infinite)
	int        fadeTime;                       ///< Fade duration in ms
} dx12Decal_t;

/** Maximum ref-entities buffered between ClearScene and RenderScene. */
#define DX12_MAX_SCENE_ENTITIES 1024

/** Maximum distinct poly draw-batches per scene (one per unique shader call). */
#define DX12_MAX_SCENE_POLY_BATCHES 1024

/** Maximum expanded triangle-list vertices stored in the poly buffer. */
#define DX12_MAX_SCENE_POLYVERTS 8192

/** Near clip plane distance (units).  Matches Q3 defaults. */
#define DX12_SCENE_NEAR   4.0f
/** Far clip plane distance (units). */
#define DX12_SCENE_FAR    65536.0f

/**
 * Number of constant-buffer slots allocated per frame:
 *   slot 0        – world / identity model matrix
 *   slots 1..N    – per-entity model matrices (up to DX12_MAX_SCENE_ENTITIES)
 *
 * Allocating a dedicated slot per entity ensures that per-frame command-list
 * recording writes each entity's matrix to a unique memory location so that
 * all recorded draw calls reference the correct matrix when the GPU executes
 * the command list.
 */
#define DX12_MAX_CB_SLOTS_PER_FRAME  (1 + DX12_MAX_SCENE_ENTITIES)

// ---------------------------------------------------------------------------
// Constant-buffer layout  (must match SceneConstants in the world HLSL)
// ---------------------------------------------------------------------------

/**
 * @struct dx12SceneConstants_t
 * @brief Per-frame / per-draw constant buffer uploaded to the GPU.
 *
 * Layout (256-byte aligned as required by D3D12 CBV):
 *   viewProj    – combined view * projection matrix (row-major float4x4)
 *   modelMatrix – per-object model-to-world transform (row-major float4x4)
 *   cameraPos   – world-space camera origin (xyz) + padding (w)
 */
typedef struct
{
	float viewProj[4][4];    ///< View * Projection matrix (row-major)
	float modelMatrix[4][4]; ///< Model matrix (row-major); identity for world
	float cameraPos[4];      ///< World-space camera position (w unused)
} dx12SceneConstants_t;

// ---------------------------------------------------------------------------
// Per-entity scene entry
// ---------------------------------------------------------------------------

/**
 * @struct dx12SceneEntity_t
 * @brief Snapshot of a refEntity_t added to the current scene.
 */
typedef struct
{
	vec3_t  origin;   ///< World-space origin
	vec3_t  axis[3];  ///< Rotation axes: [0]=forward [1]=left [2]=up
	qhandle_t hModel; ///< Model handle (0 = no model)
} dx12SceneEntity_t;

// ---------------------------------------------------------------------------
// World-space polygon (decal/effect) batch
// ---------------------------------------------------------------------------

/**
 * @struct dx12ScenePolyBatch_t
 * @brief Descriptor for one poly draw call: a contiguous slice of the poly
 *        vertex buffer, all sharing the same shader/texture.
 */
typedef struct
{
	qhandle_t shaderHandle; ///< Texture / material handle
	int       firstVert;    ///< Index of first vertex in dx12Scene.polyVerts
	int       numVerts;     ///< Number of consecutive vertices
} dx12ScenePolyBatch_t;

// ---------------------------------------------------------------------------
// Scene state
// ---------------------------------------------------------------------------

/**
 * @struct dx12SceneState_t
 * @brief All 3D scene state for one frame.
 */
typedef struct
{
	// 3D-specific GPU resources
	ID3D12RootSignature  *rootSignature3D; ///< Root sig: CBV b0 + SRV table (t0, t1)
	ID3D12PipelineState  *pso3D;           ///< PSO for dx12WorldVertex_t input – opaque
	ID3D12PipelineState  *pso3DTranslucent; ///< Same layout but with alpha blending + no depth write

	// Per-frame constant buffer (upload heap, persistently mapped, CBV_SIZE aligned)
	ID3D12Resource *constantBuffer;       ///< Holds DX12_FRAME_COUNT * DX12_MAX_CB_SLOTS_PER_FRAME slots
	UINT8          *cbMapped;             ///< Persistently-mapped CPU pointer
	UINT            cbSlotSize;           ///< Size of one aligned CBV slot in bytes

	// Entity list accumulated this frame
	dx12SceneEntity_t entities[DX12_MAX_SCENE_ENTITIES]; ///< Buffered entities
	int               numEntities;                       ///< Active count

	// World-space poly (decal/effect) buffer
	dx12WorldVertex_t  *polyVerts;                         ///< malloc'd staging array (DX12_MAX_SCENE_POLYVERTS)
	int                 numPolyVerts;                      ///< Vertices used this frame
	dx12ScenePolyBatch_t polyBatches[DX12_MAX_SCENE_POLY_BATCHES]; ///< Per-shader draw groups
	int                  numPolyBatches;                   ///< Active batch count
	ID3D12Resource      *polyVertexBuffer;                 ///< GPU upload-heap VB (persistent)
	UINT8               *polyVBMapped;                     ///< Persistently-mapped CPU pointer

	// Per-frame dynamic lights
	dx12DLight_t dlights[DX12_MAX_DLIGHTS]; ///< Dynamic light list
	int          numDLights;                ///< Active dlight count

	// Per-frame coronas / flares
	dx12Corona_t coronas[DX12_MAX_CORONAS]; ///< Corona list
	int          numCoronas;                ///< Active corona count

	// Persistent decal list – cleared only by DX12_ClearDecals(), NOT by DX12_ClearScene()
	dx12Decal_t decals[DX12_MAX_DECALS]; ///< Decal polygon list
	int         numDecals;               ///< Active decal count

	qboolean initialized; ///< qtrue after DX12_SceneInit()
} dx12SceneState_t;

extern dx12SceneState_t dx12Scene;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * @brief DX12_SceneInit  – create 3D PSO, root signature, and constant buffer.
 * @return qtrue on success.
 */
qboolean DX12_SceneInit(void);

/** @brief DX12_SceneShutdown – release all 3D scene GPU resources. */
void DX12_SceneShutdown(void);

/**
 * @brief DX12_AddEntityToScene – buffer a ref-entity for the next RenderScene call.
 * @param[in] re  Entity to add; ignored if the scene list is full.
 */
void DX12_AddEntityToScene(const refEntity_t *re);

/**
 * @brief DX12_ClearScene – reset the per-frame entity and poly lists.
 * Called at the start of each scene (before adding entities / polys).
 */
void DX12_ClearScene(void);

/**
 * @brief DX12_AddScenePoly – buffer a world-space convex polygon for rendering.
 *
 * Expands the polygon from its fan representation to a flat triangle list and
 * appends it to the per-frame poly staging buffer.  The polygon is rendered
 * during the next DX12_RenderScene call, after entities and before translucent
 * world surfaces.
 *
 * @param hShader   Texture / material handle for this polygon.
 * @param numVerts  Number of vertices in @p verts (must be ≥ 3).
 * @param verts     Polygon vertices in world space (polyVert_t format).
 */
void DX12_AddScenePoly(qhandle_t hShader, int numVerts, const polyVert_t *verts);

/**
 * @brief DX12_AddDecalToScene – store a world-space decal polygon for persistent rendering.
 *
 * Decals are NOT cleared by DX12_ClearScene().  Call DX12_ClearDecals() explicitly
 * (e.g. on map load) to remove all active decals.
 *
 * @param hShader   Material handle for the decal.
 * @param numVerts  Source polygon vertex count (clamped to DX12_MAX_DECAL_VERTS).
 * @param points    World-space polygon corners (vec3_t*).
 * @param color     Decal tint colour (RGBA, [0,1]).
 * @param lifeTime  Life hint in ms (0 = infinite).
 * @param fadeTime  Fade duration in ms.
 */
void DX12_AddDecalToScene(qhandle_t hShader, int numVerts, vec3_t *points,
                          const float *color, int lifeTime, int fadeTime);

/**
 * @brief DX12_ClearDecals – remove all persistent decals.
 * Called by RE_DX12_ClearDecals; safe to call at any time.
 */
void DX12_ClearDecals(void);

/**
 * @brief DX12_RenderScene – render a full 3D scene.
 * @param[in] fd  Frame descriptor with view parameters.
 *
 * Renders in order: sky → opaque world → fog surfaces → entities →
 * translucent world surfaces.  Does NOT touch the 2D pipeline.
 */
void DX12_RenderScene(const refdef_t *fd);

#endif // _WIN32
#endif // DX12_SCENE_H
