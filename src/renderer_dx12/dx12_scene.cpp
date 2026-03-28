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
 * @file dx12_scene.cpp
 * @brief DX12 3D scene rendering – view/projection setup and draw dispatch.
 *
 * Implements DX12_RenderScene() which:
 *   1. Builds a view matrix from fd->vieworg / fd->viewaxis.
 *   2. Builds a perspective projection matrix from fd->fov_x / fd->fov_y.
 *   3. Writes the combined viewProj + cameraPos into a per-frame constant buffer.
 *   4. Draws world BSP draw surfaces in order:
 *        sky → opaque → fog-tagged → entities → translucent
 *   5. For each world surface: sets the world VB/IB, binds diffuse + lightmap
 *      SRVs, and issues DrawIndexedInstanced.
 *   6. For each entity: builds a model matrix from origin+axis, updates the
 *      per-object constant buffer slot, and draws if a model is available.
 *
 * The 3D pipeline uses its own root signature and PSO (separate from the 2D
 * pipeline in tr_dx12_backend.cpp) so the two never interfere.
 */

#include "dx12_scene.h"
#include "dx12_world.h"
#include "dx12_shader.h"
#include "dx12_poly.h"
#include "dx12_model.h"

#ifdef _WIN32

#include <math.h>   // tanf, (float)M_PI

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// HLSL source for the 3D world/entity shaders
// ---------------------------------------------------------------------------

/**
 * @brief Embedded HLSL for the 3D world pipeline.
 *
 * Root signature (slot assignments must match DX12_SceneInit):
 *   b0 – SceneConstants (CBV): viewProj, modelMatrix, cameraPos, fog, entity light
 *   b1 – PerSurfaceConstants (root 32-bit constants): uvOffset, alphaTest, isEntity
 *   t0 – diffuse texture  (SRV)
 *   t1 – lightmap texture (SRV)
 *   s0 – linear-wrap sampler (static)
 *
 * Vertex input matches dx12WorldVertex_t:
 *   POSITION  float3   xyz
 *   TEXCOORD0 float2   st   (diffuse UV)
 *   TEXCOORD1 float2   lm   (lightmap UV)
 *   NORMAL    float3   normal
 *   COLOR     float4   color (per-vertex modulate)
 *
 * The pixel shader:
 *   - Applies uvOffset (inline root constants) to the diffuse UV.
 *   - Discards pixels whose alpha fails the alphaTestThreshold test.
 *   - For world surfaces (isEntity == 0): applies lightmap * overbright.
 *   - For entity surfaces (isEntity > 0): applies entity ambient + N.L * directed.
 *   - Applies linear depth fog when fogEnabled > 0.
 */
static const char g_worldShaderSource[] =
	// Main per-frame/per-entity constant buffer
	"cbuffer SceneConstants : register(b0)\n"
	"{\n"
	"    float4x4 viewProj;\n"
	"    float4x4 modelMatrix;\n"
	"    float4   cameraPos;\n"
	"    float4   fogColor;\n"
	"    float    fogStart;\n"
	"    float    fogEnd;\n"
	"    float    fogEnabled;\n"
	"    float    overBrightFactor;\n"
	"    float4   entityAmbient;\n"
	"    float4   entityDirected;\n"
	"    float4   entityLightDir;\n"
	"};\n"
	// Per-draw-surface inline root constants (set via SetGraphicsRoot32BitConstants)
	"cbuffer PerSurface : register(b1)\n"
	"{\n"
	"    float uvOffsetU;\n"
	"    float uvOffsetV;\n"
	"    float alphaTestThreshold;\n"
	"    float isEntity;\n"
	"};\n"
	"\n"
	"Texture2D    g_diffuse  : register(t0);\n"
	"Texture2D    g_lightmap : register(t1);\n"
	"SamplerState g_sampler  : register(s0);\n"
	"\n"
	"struct VSInput\n"
	"{\n"
	"    float3 pos    : POSITION;\n"
	"    float2 uv     : TEXCOORD0;\n"
	"    float2 lm     : TEXCOORD1;\n"
	"    float3 normal : NORMAL;\n"
	"    float4 color  : COLOR;\n"
	"};\n"
	"\n"
	"struct PSInput\n"
	"{\n"
	"    float4 pos      : SV_POSITION;\n"
	"    float2 uv       : TEXCOORD0;\n"
	"    float2 lm       : TEXCOORD1;\n"
	"    float4 color    : COLOR;\n"
	"    float3 worldPos : TEXCOORD2;\n"
	"    float3 normal   : TEXCOORD3;\n"
	"};\n"
	"\n"
	"PSInput VSMain(VSInput input)\n"
	"{\n"
	"    PSInput o;\n"
	"    float4 worldPos4 = mul(modelMatrix, float4(input.pos, 1.0));\n"
	"    o.pos      = mul(viewProj, worldPos4);\n"
	"    o.uv       = input.uv + float2(uvOffsetU, uvOffsetV);\n"
	"    o.lm       = input.lm;\n"
	"    o.color    = input.color;\n"
	"    o.worldPos = worldPos4.xyz;\n"
	"    // Transform normal to world space (rotation part of modelMatrix only)\n"
	"    o.normal   = normalize(mul((float3x3)modelMatrix, input.normal));\n"
	"    return o;\n"
	"}\n"
	"\n"
	"float4 PSMain(PSInput input) : SV_TARGET\n"
	"{\n"
	"    float4 diffuse = g_diffuse.Sample(g_sampler, input.uv);\n"
	"\n"
	"    // Alpha test: positive threshold = GE (keep if alpha >= threshold),\n"
	"    // negative = LT (keep if alpha < |threshold|).\n"
	"    if (alphaTestThreshold > 0.0)\n"
	"    {\n"
	"        clip(diffuse.a - alphaTestThreshold);\n"
	"    }\n"
	"    else if (alphaTestThreshold < 0.0)\n"
	"    {\n"
	"        clip(-alphaTestThreshold - diffuse.a);\n"
	"    }\n"
	"\n"
	"    float4 result;\n"
	"    if (isEntity > 0.0)\n"
	"    {\n"
	"        // Entity shading: ambient + N.L * directed, then overbright.\n"
	"        float  nDotL  = saturate(dot(input.normal, entityLightDir.xyz));\n"
	"        float3 light  = entityAmbient.rgb + nDotL * entityDirected.rgb;\n"
	"        result = diffuse * float4(saturate(light * overBrightFactor), 1.0) * input.color;\n"
	"    }\n"
	"    else\n"
	"    {\n"
	"        // World surface: lightmap overbright (matches GL1 R_ColorShiftLightingBytes).\n"
	"        float4 lightmap = g_lightmap.Sample(g_sampler, input.lm);\n"
	"        result = diffuse * (lightmap * overBrightFactor) * input.color;\n"
	"    }\n"
	"    result = float4(saturate(result.rgb), result.a);\n"
	"\n"
	"    // Linear depth fog\n"
	"    if (fogEnabled > 0.0)\n"
	"    {\n"
	"        float viewDist  = length(input.worldPos - cameraPos.xyz);\n"
	"        float fogFactor = saturate((fogEnd - viewDist) / max(fogEnd - fogStart, 1.0));\n"
	"        result.rgb      = lerp(fogColor.rgb, result.rgb, fogFactor);\n"
	"    }\n"
	"    return result;\n"
	"}\n";

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

dx12SceneState_t dx12Scene;

// ---------------------------------------------------------------------------
// Per-map IB/VB overrun warn counters (Issue #11 fix).
// File-scope so DX12_SceneInit() can reset them between map loads.
// (Previously declared static inside SCN_DrawSurface, which caused them to
// persist across map reloads and suppress warnings on subsequent maps.)
// ---------------------------------------------------------------------------
static int g_ibWarnCount = 0;
static int g_vbWarnCount = 0;

// Current scene time in milliseconds (from refdef_t::time).
// Set at the start of DX12_RenderScene; used by SCN_DrawSurface for
// tcMod scroll UV animation and animMap frame selection.
static int g_sceneTimeMs = 0;

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

/** Identity 4×4 matrix (row-major). */
static void Mat4Identity(float m[4][4])
{
	int i, j;

	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			m[i][j] = (i == j) ? 1.0f : 0.0f;
		}
	}
}

/** Row-major matrix multiply: out = a * b */
static void Mat4Mul(float out[4][4], const float a[4][4], const float b[4][4])
{
	int i, j, k;

	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			float sum = 0.0f;

			for (k = 0; k < 4; k++)
			{
				sum += a[i][k] * b[k][j];
			}
			out[i][j] = sum;
		}
	}
}

/**
 * @brief Build a view matrix from a Q3/ET vieworg and viewaxis.
 *
 * Q3 convention:
 *   viewaxis[0] = forward vector
 *   viewaxis[1] = left vector   (not right!)
 *   viewaxis[2] = up vector
 *
 * D3D convention (row-vector * row-major matrix):
 *   Row 0 = right  (= -viewaxis[1])
 *   Row 1 = up     (= +viewaxis[2])
 *   Row 2 = forward (= +viewaxis[0])
 *   Row 3 = [0, 0, 0, 1]
 *   Translation = -dot(axis_row, eye)
 *
 * @param[out] m      4×4 view matrix (row-major).
 * @param[in]  origin Camera position in world space.
 * @param[in]  axis   3×3 rotation: axis[0]=forward, axis[1]=left, axis[2]=up.
 */

// ---------------------------------------------------------------------------
// HLSL source for the sky shader
// ---------------------------------------------------------------------------

/**
 * @brief Embedded HLSL for the sky surface pipeline.
 *
 * The sky VS strips the translation component from the view matrix so the
 * skybox stays centred on the camera.  The position is emitted with w==z so
 * the depth buffer always sees NDC depth = 1.0 (far plane), ensuring sky is
 * drawn behind everything else.
 *
 * Root signature is the same as the world pipeline; only t0 (diffuse) is used.
 */
static const char g_skyShaderSource[] =
	"cbuffer SceneConstants : register(b0)\n"
	"{\n"
	"    float4x4 viewProj;\n"
	"    float4x4 modelMatrix;\n"
	"    float4   cameraPos;\n"
	"    float4   fogColor;\n"
	"    float    fogStart;\n"
	"    float    fogEnd;\n"
	"    float    fogEnabled;\n"
	"    float    overBrightFactor;\n"
	"    float4   entityAmbient;\n"
	"    float4   entityDirected;\n"
	"    float4   entityLightDir;\n"
	"};\n"
	"cbuffer PerSurfConstants : register(b1)\n"
	"{\n"
	"    float uvOffsetU;\n"
	"    float uvOffsetV;\n"
	"    float alphaTestThreshold;\n"
	"    float isEntity;\n"
	"};\n"
	"Texture2D    gDiffuse  : register(t0);\n"
	"SamplerState gSampler  : register(s0);\n"
	"struct VSIn\n"
	"{\n"
	"    float3 pos   : POSITION;\n"
	"    float2 st    : TEXCOORD0;\n"
	"    float2 lm    : TEXCOORD1;\n"
	"    float3 norm  : NORMAL;\n"
	"    float4 col   : COLOR;\n"
	"};\n"
	"struct VSOut\n"
	"{\n"
	"    float4 pos    : SV_POSITION;\n"
	"    float2 st     : TEXCOORD0;\n"
	"};\n"
	"VSOut VSMain(VSIn vin)\n"
	"{\n"
	"    VSOut vout;\n"
	"    // Remove translation from viewProj so sky stays centred on the camera.\n"
	"    // Build a rotation-only viewProj by zeroing out the translation column.\n"
	"    float4x4 vpNoTrans = viewProj;\n"
	"    vpNoTrans[3][0] = 0.0f;\n"
	"    vpNoTrans[3][1] = 0.0f;\n"
	"    vpNoTrans[3][2] = 0.0f;\n"
	"    float4 clipPos = mul(float4(vin.pos, 1.0f), vpNoTrans);\n"
	"    // Force depth to far plane (NDC depth = 1.0) by setting w = z.\n"
	"    vout.pos = clipPos.xyww;\n"
	"    vout.st  = float2(vin.st.x + uvOffsetU, vin.st.y + uvOffsetV);\n"
	"    return vout;\n"
	"}\n"
	"float4 PSMain(VSOut pin) : SV_TARGET\n"
	"{\n"
	"    float4 col = gDiffuse.Sample(gSampler, pin.st);\n"
	"    if (alphaTestThreshold > 0.0f && col.a < alphaTestThreshold) { discard; }\n"
	"    return col;\n"
	"}\n";

static void BuildViewMatrix(float m[4][4], const vec3_t origin, const vec3_t axis[3])
{
	float rx[3], ry[3], rz[3];
	int   i;

	// Camera right = -left
	rx[0] = -axis[1][0];
	rx[1] = -axis[1][1];
	rx[2] = -axis[1][2];

	// Camera up = viewaxis[2]
	ry[0] = axis[2][0];
	ry[1] = axis[2][1];
	ry[2] = axis[2][2];

	// Camera forward = viewaxis[0]
	rz[0] = axis[0][0];
	rz[1] = axis[0][1];
	rz[2] = axis[0][2];

	// Build row-major view matrix
	for (i = 0; i < 4; i++)
	{
		m[i][0] = m[i][1] = m[i][2] = m[i][3] = 0.0f;
	}

	m[0][0] = rx[0]; m[0][1] = ry[0]; m[0][2] = rz[0]; m[0][3] = 0.0f;
	m[1][0] = rx[1]; m[1][1] = ry[1]; m[1][2] = rz[1]; m[1][3] = 0.0f;
	m[2][0] = rx[2]; m[2][1] = ry[2]; m[2][2] = rz[2]; m[2][3] = 0.0f;

	// Translation: -dot(row, eye)
	m[3][0] = -(rx[0] * origin[0] + rx[1] * origin[1] + rx[2] * origin[2]);
	m[3][1] = -(ry[0] * origin[0] + ry[1] * origin[1] + ry[2] * origin[2]);
	m[3][2] = -(rz[0] * origin[0] + rz[1] * origin[1] + rz[2] * origin[2]);
	m[3][3] = 1.0f;
}

/**
 * @brief Build a D3D left-handed perspective projection matrix.
 *
 * Uses a standard [0, 1] NDC depth range (D3D12 default).
 *
 * @param[out] m        4×4 projection matrix (row-major).
 * @param[in]  fovXDeg  Horizontal field of view in degrees.
 * @param[in]  fovYDeg  Vertical field of view in degrees.
 * @param[in]  zNear    Near clip plane distance.
 * @param[in]  zFar     Far clip plane distance.
 */
static void BuildProjMatrix(float m[4][4],
                             float fovXDeg, float fovYDeg,
                             float zNear, float zFar)
{
	float fovXRad = fovXDeg * (float)(M_PI / 180.0);
	float fovYRad = fovYDeg * (float)(M_PI / 180.0);
	float w       = 1.0f / tanf(fovXRad * 0.5f); // cot(fovX/2)
	float h       = 1.0f / tanf(fovYRad * 0.5f); // cot(fovY/2)
	float q       = zFar / (zFar - zNear);
	int   i, j;

	(void)fovXDeg; // used via fovXRad

	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			m[i][j] = 0.0f;
		}
	}

	// Row-major, left-handed, [0,1] depth
	m[0][0] = w;
	m[1][1] = h;
	m[2][2] = q;
	m[2][3] = 1.0f;
	m[3][2] = -q * zNear;
}

/**
 * @brief Build a model matrix from a Q3/ET entity origin + axis.
 *
 * @param[out] m    4×4 model matrix (row-major).
 * @param[in]  org  Entity world-space origin.
 * @param[in]  axis Entity axes: [0]=forward [1]=left [2]=up.
 */
static void BuildModelMatrix(float m[4][4], const vec3_t org, const vec3_t axis[3])
{
	// Column vectors of the rotation part are the world-space axes.
	// row-major layout: m[row][col]
	// First three rows are the basis vectors (transposed for row-major).
	m[0][0] = axis[0][0]; m[0][1] = axis[1][0]; m[0][2] = axis[2][0]; m[0][3] = 0.0f;
	m[1][0] = axis[0][1]; m[1][1] = axis[1][1]; m[1][2] = axis[2][1]; m[1][3] = 0.0f;
	m[2][0] = axis[0][2]; m[2][1] = axis[1][2]; m[2][2] = axis[2][2]; m[2][3] = 0.0f;
	m[3][0] = org[0];     m[3][1] = org[1];     m[3][2] = org[2];     m[3][3] = 1.0f;
}

// ---------------------------------------------------------------------------
// Constant-buffer helpers
// ---------------------------------------------------------------------------

/** Round up to the nearest 256-byte boundary (D3D12 CBV requirement). */
static UINT Align256(UINT size)
{
	return (size + 255U) & ~255U;
}

/**
 * @brief Write dx12SceneConstants_t into the constant buffer at the given slot.
 *
 * @param[in] slot  Absolute slot index into the CB allocation.  For frame f:
 *                    world slot = f * DX12_MAX_CB_SLOTS_PER_FRAME + 0
 *                    entity[i] = f * DX12_MAX_CB_SLOTS_PER_FRAME + 1 + i
 * @param[in] cb    Constant data to write.
 */
static void SCN_UpdateCB(UINT slot, const dx12SceneConstants_t *cb)
{
	UINT8 *dst = dx12Scene.cbMapped + (size_t)slot * dx12Scene.cbSlotSize;

	memcpy(dst, cb, sizeof(dx12SceneConstants_t));
}

// ---------------------------------------------------------------------------
// Surface draw helper
// ---------------------------------------------------------------------------

/**
 * @brief Bind SRVs, per-surface root constants, and issue a DrawIndexedInstanced.
 *
 * Per-surface variables (uvOffset, alphaTestThreshold, isEntity) are passed as
 * inline root 32-bit constants (DX12_SCENE_ROOT_PARAM_PERSURF) recorded inline
 * in the command list.  This is the correct DX12 approach for per-draw-call
 * variation: unlike CB writes to a shared upload-heap slot, root constants are
 * embedded in the command stream and are guaranteed correct for each draw call
 * even under deferred GPU execution.
 *
 * Also handles:
 *   - animMap: selects the current animation frame via g_sceneTimeMs.
 *   - tcMod scroll: accumulates UV offsets from SCROLL entries.
 *   - alphaFunc: passes the stage's alphaTestThreshold as a root constant.
 *
 * @param[in] ds      World draw surface descriptor.
 * @param[in] cbGpuVA GPU virtual address of the (per-frame or per-entity) CB slot.
 */
static void SCN_DrawSurface(const dx12DrawSurf_t *ds, D3D12_GPU_VIRTUAL_ADDRESS cbGpuVA)
{
	dx12Material_t         *mat      = NULL;
	dx12Texture_t          *diffTex  = NULL;
	dx12Texture_t          *lmTex    = NULL;
	qhandle_t               diffHandle = 0;
	dx12PerSurfConstants_t  psc      = {};

	if (ds->numIndexes <= 0 || ds->numVertices <= 0)
	{
		return;
	}

	// Guard against out-of-range draw calls.
	// g_ibWarnCount / g_vbWarnCount are file-scope and reset by DX12_SceneInit()
	// (Issue #11 fix: previously declared static inside this function, causing
	// them to persist across map loads and suppress warnings on subsequent maps.)
	if (dx12World.numIndexes > 0
	    && ((UINT)ds->firstIndex >= dx12World.numIndexes
	        || (UINT)ds->firstIndex + (UINT)ds->numIndexes > dx12World.numIndexes))
	{
		if (g_ibWarnCount < 5)
		{
			g_ibWarnCount++;
			dx12.ri.Printf(PRINT_DEVELOPER,
			               "SCN_DrawSurface: IB overrun – firstIdx %d numIdx %d totalIdx %u (skipped)\n",
			               ds->firstIndex, ds->numIndexes, dx12World.numIndexes);
		}

		return;
	}

	if (dx12World.numVertices > 0
	    && ((UINT)ds->firstVertex >= dx12World.numVertices
	        || (UINT)ds->firstVertex + (UINT)ds->numVertices > dx12World.numVertices))
	{
		if (g_vbWarnCount < 5)
		{
			g_vbWarnCount++;
			dx12.ri.Printf(PRINT_DEVELOPER,
			               "SCN_DrawSurface: VB overrun – firstVtx %d numVtx %d totalVtx %u (skipped)\n",
			               ds->firstVertex, ds->numVertices, dx12World.numVertices);
		}

		return;
	}

	// --- Resolve textures; build per-surface root constants ---
	mat = DX12_GetMaterial(ds->materialHandle);

	if (mat && mat->numStages > 0 && mat->stages[0].active)
	{
		const dx12MaterialStage_t *st = &mat->stages[0];
		int t;

		// animMap: select the animation frame for the current scene time
		if (st->animNumFrames > 0 && st->animFps > 0.0f)
		{
			int frameIdx = (int)((float)g_sceneTimeMs * st->animFps / 1000.0f);

			frameIdx = ((frameIdx % st->animNumFrames) + st->animNumFrames) % st->animNumFrames;
			diffHandle = st->animFrames[frameIdx];
		}
		else
		{
			diffHandle = st->texHandle;
		}

		// tcMod scroll: accumulate UV offset from SCROLL entries
		for (t = 0; t < st->numTcMods; t++)
		{
			if (st->tcMods[t].type == DX12_TMOD_SCROLL)
			{
				float timeSec = (float)g_sceneTimeMs / 1000.0f;

				psc.uvOffsetU += st->tcMods[t].scroll[0] * timeSec;
				psc.uvOffsetV += st->tcMods[t].scroll[1] * timeSec;
			}
		}

		// alphaFunc → inline root constant (correct per draw call)
		psc.alphaTestThreshold = st->alphaTestThreshold;
	}

	// isEntity = 0 for world surfaces; entity draws override this before calling
	psc.isEntity = 0.0f;

	diffTex = DX12_GetTexture(diffHandle);

	// --- Bind constant buffer and per-surface root constants ---
	dx12.commandList->SetGraphicsRootConstantBufferView(DX12_SCENE_ROOT_PARAM_CB, cbGpuVA);
	dx12.commandList->SetGraphicsRoot32BitConstants(DX12_SCENE_ROOT_PARAM_PERSURF,
	                                                4, &psc, 0);

	// --- Bind SRV descriptor tables ---
	{
		D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = {};

		if (diffTex && diffTex->resource)
		{
			srvHandle = diffTex->gpuHandle;
		}
		else
		{
			srvHandle = dx12.srvHeap->GetGPUDescriptorHandleForHeapStart();
		}

		dx12.commandList->SetGraphicsRootDescriptorTable(DX12_SCENE_ROOT_PARAM_DIFFUSE,
		                                                 srvHandle);

		// Lightmap: BSP slot or white fallback for vertex-lit surfaces
		D3D12_GPU_DESCRIPTOR_HANDLE lmHandle = dx12.srvHeap->GetGPUDescriptorHandleForHeapStart();

		if (ds->lightmapIndex >= 0 && ds->lightmapIndex < dx12World.numLightmaps)
		{
			qhandle_t lmH = dx12World.lightmapHandles[ds->lightmapIndex];

			lmTex = DX12_GetTexture(lmH);
			if (lmTex && lmTex->resource)
			{
				lmHandle = lmTex->gpuHandle;
			}
		}

		dx12.commandList->SetGraphicsRootDescriptorTable(DX12_SCENE_ROOT_PARAM_LIGHTMAP,
		                                                 lmHandle);
	}

	// --- Issue draw call ---
	// Indices are absolute (rebased in dx12_world.cpp loader), so BaseVertexLocation = 0.
	dx12.commandList->DrawIndexedInstanced(
		(UINT)ds->numIndexes,
		1,
		(UINT)ds->firstIndex,
		0,
		0
		);
}

// ---------------------------------------------------------------------------
// DX12_DrawSkySurface
// ---------------------------------------------------------------------------

/**
 * @brief Draw a single MST_SKY world surface using the sky PSO.
 *
 * Uses the sky PSO (pso3DSky) which strips view translation so the skybox
 * stays centred on the camera, and forces NDC depth = 1.0 so sky is always
 * behind world geometry.  Falls back to the opaque PSO when pso3DSky is NULL.
 *
 * @param[in] ds      World draw surface descriptor.
 * @param[in] cbGpuVA GPU virtual address of the per-frame CB slot.
 */
static void DX12_DrawSkySurface(const dx12DrawSurf_t *ds, D3D12_GPU_VIRTUAL_ADDRESS cbGpuVA)
{
	dx12Material_t         *mat     = NULL;
	dx12Texture_t          *diffTex = NULL;
	qhandle_t               diffHandle = 0;
	dx12PerSurfConstants_t  psc     = {};

	if (ds->numIndexes <= 0 || ds->numVertices <= 0)
	{
		return;
	}

	if (dx12World.numIndexes > 0
	    && ((UINT)ds->firstIndex >= dx12World.numIndexes
	        || (UINT)ds->firstIndex + (UINT)ds->numIndexes > dx12World.numIndexes))
	{
		return;
	}

	mat = DX12_GetMaterial(ds->materialHandle);

	if (mat && mat->numStages > 0 && mat->stages[0].active)
	{
		const dx12MaterialStage_t *st = &mat->stages[0];
		int t;

		if (st->animNumFrames > 0 && st->animFps > 0.0f)
		{
			int frameIdx = (int)((float)g_sceneTimeMs * st->animFps / 1000.0f);

			frameIdx = ((frameIdx % st->animNumFrames) + st->animNumFrames) % st->animNumFrames;
			diffHandle = st->animFrames[frameIdx];
		}
		else
		{
			diffHandle = st->texHandle;
		}

		for (t = 0; t < st->numTcMods; t++)
		{
			if (st->tcMods[t].type == DX12_TMOD_SCROLL)
			{
				float timeSec = (float)g_sceneTimeMs / 1000.0f;

				psc.uvOffsetU += st->tcMods[t].scroll[0] * timeSec;
				psc.uvOffsetV += st->tcMods[t].scroll[1] * timeSec;
			}
		}

		psc.alphaTestThreshold = st->alphaTestThreshold;
	}

	psc.isEntity = 0.0f;

	diffTex = DX12_GetTexture(diffHandle);

	// Switch to sky PSO (no depth write, view-translation stripped)
	if (dx12Scene.pso3DSky)
	{
		dx12.commandList->SetPipelineState(dx12Scene.pso3DSky);
	}

	dx12.commandList->SetGraphicsRootConstantBufferView(DX12_SCENE_ROOT_PARAM_CB, cbGpuVA);
	dx12.commandList->SetGraphicsRoot32BitConstants(DX12_SCENE_ROOT_PARAM_PERSURF, 4, &psc, 0);

	{
		D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = dx12.srvHeap->GetGPUDescriptorHandleForHeapStart();

		if (diffTex && diffTex->resource)
		{
			srvHandle = diffTex->gpuHandle;
		}

		dx12.commandList->SetGraphicsRootDescriptorTable(DX12_SCENE_ROOT_PARAM_DIFFUSE, srvHandle);
		// Sky does not use a lightmap; bind the heap start as a dummy
		dx12.commandList->SetGraphicsRootDescriptorTable(DX12_SCENE_ROOT_PARAM_LIGHTMAP,
		                                                 dx12.srvHeap->GetGPUDescriptorHandleForHeapStart());
	}

	dx12.commandList->DrawIndexedInstanced(
		(UINT)ds->numIndexes,
		1,
		(UINT)ds->firstIndex,
		0,
		0
		);

	// Restore opaque PSO for subsequent draws
	dx12.commandList->SetPipelineState(dx12Scene.pso3D);
}

// ---------------------------------------------------------------------------
// DX12_SceneInit
// ---------------------------------------------------------------------------

/**
 * @brief DX12_SceneInit
 *
 * Creates the 3D-specific root signature, PSO, and per-frame constant buffer.
 */
qboolean DX12_SceneInit(void)
{
	HRESULT hr;
	UINT    cbTotalSize;

	if (!dx12.initialized)
	{
		return qfalse;
	}

	Com_Memset(&dx12Scene, 0, sizeof(dx12Scene));

	// Reset per-map warn counters (Issue #11 fix)
	g_ibWarnCount = 0;
	g_vbWarnCount = 0;
	g_sceneTimeMs = 0;

	// ----------------------------------------------------------------
	// Root Signature for 3D rendering
	//   Param 0 (DX12_SCENE_ROOT_PARAM_CB):       Root CBV at b0  (VS + PS)
	//   Param 1 (DX12_SCENE_ROOT_PARAM_DIFFUSE):  Descriptor table – 1 SRV at t0
	//   Param 2 (DX12_SCENE_ROOT_PARAM_LIGHTMAP): Descriptor table – 1 SRV at t1
	//   Param 3 (DX12_SCENE_ROOT_PARAM_PERSURF):  32-bit constants at b1 (4 DWORDs)
	//     uvOffsetU, uvOffsetV, alphaTestThreshold, isEntity
	//   Static sampler at s0: linear-wrap
	// ----------------------------------------------------------------
	{
		D3D12_DESCRIPTOR_RANGE srvRange0 = {};
		srvRange0.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		srvRange0.NumDescriptors                    = 1;
		srvRange0.BaseShaderRegister                = 0; // t0
		srvRange0.RegisterSpace                     = 0;
		srvRange0.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

		D3D12_DESCRIPTOR_RANGE srvRange1 = {};
		srvRange1.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		srvRange1.NumDescriptors                    = 1;
		srvRange1.BaseShaderRegister                = 1; // t1
		srvRange1.RegisterSpace                     = 0;
		srvRange1.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

		D3D12_ROOT_PARAMETER params[4] = {};

		// Param 0: root CBV (b0) – scene + entity constants
		params[DX12_SCENE_ROOT_PARAM_CB].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
		params[DX12_SCENE_ROOT_PARAM_CB].Descriptor.ShaderRegister = 0; // b0
		params[DX12_SCENE_ROOT_PARAM_CB].Descriptor.RegisterSpace  = 0;
		params[DX12_SCENE_ROOT_PARAM_CB].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

		// Param 1: diffuse SRV table (t0)
		params[DX12_SCENE_ROOT_PARAM_DIFFUSE].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		params[DX12_SCENE_ROOT_PARAM_DIFFUSE].DescriptorTable.NumDescriptorRanges = 1;
		params[DX12_SCENE_ROOT_PARAM_DIFFUSE].DescriptorTable.pDescriptorRanges   = &srvRange0;
		params[DX12_SCENE_ROOT_PARAM_DIFFUSE].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_PIXEL;

		// Param 2: lightmap SRV table (t1)
		params[DX12_SCENE_ROOT_PARAM_LIGHTMAP].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		params[DX12_SCENE_ROOT_PARAM_LIGHTMAP].DescriptorTable.NumDescriptorRanges = 1;
		params[DX12_SCENE_ROOT_PARAM_LIGHTMAP].DescriptorTable.pDescriptorRanges   = &srvRange1;
		params[DX12_SCENE_ROOT_PARAM_LIGHTMAP].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_PIXEL;

		// Param 3: per-surface inline root constants (b1) – 4 DWORDs
		// uvOffsetU, uvOffsetV, alphaTestThreshold, isEntity
		// Recorded inline in the command list – correct per draw call.
		params[DX12_SCENE_ROOT_PARAM_PERSURF].ParameterType                         = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
		params[DX12_SCENE_ROOT_PARAM_PERSURF].Constants.ShaderRegister              = 1; // b1
		params[DX12_SCENE_ROOT_PARAM_PERSURF].Constants.RegisterSpace               = 0;
		params[DX12_SCENE_ROOT_PARAM_PERSURF].Constants.Num32BitValues              = 4;
		params[DX12_SCENE_ROOT_PARAM_PERSURF].ShaderVisibility                      = D3D12_SHADER_VISIBILITY_ALL;

		D3D12_STATIC_SAMPLER_DESC sampler = {};
		sampler.Filter           = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
		sampler.AddressU         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
		sampler.AddressV         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
		sampler.AddressW         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
		sampler.MipLODBias       = 0.0f;
		sampler.MaxAnisotropy    = 1;
		sampler.ComparisonFunc   = D3D12_COMPARISON_FUNC_NEVER;
		sampler.BorderColor      = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
		sampler.MinLOD           = 0.0f;
		sampler.MaxLOD           = D3D12_FLOAT32_MAX;
		sampler.ShaderRegister   = 0; // s0
		sampler.RegisterSpace    = 0;
		sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

		D3D12_ROOT_SIGNATURE_DESC rsd = {};
		rsd.NumParameters     = 4;
		rsd.pParameters       = params;
		rsd.NumStaticSamplers = 1;
		rsd.pStaticSamplers   = &sampler;
		rsd.Flags             = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

		ID3DBlob *sigBlob  = NULL;
		ID3DBlob *errBlob  = NULL;

		hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob);
		if (FAILED(hr))
		{
			if (errBlob)
			{
				dx12.ri.Printf(PRINT_WARNING, "DX12_SceneInit: root sig error: %s\n",
				               (const char *)errBlob->GetBufferPointer());
				errBlob->Release();
			}
			dx12.ri.Printf(PRINT_WARNING,
			               "DX12_SceneInit: D3D12SerializeRootSignature failed (0x%08lx)\n", hr);
			return qfalse;
		}
		if (errBlob)
		{
			errBlob->Release();
		}

		hr = dx12.device->CreateRootSignature(0, sigBlob->GetBufferPointer(),
		                                      sigBlob->GetBufferSize(),
		                                      IID_PPV_ARGS(&dx12Scene.rootSignature3D));
		sigBlob->Release();

		if (FAILED(hr))
		{
			dx12.ri.Printf(PRINT_WARNING,
			               "DX12_SceneInit: CreateRootSignature failed (0x%08lx)\n", hr);
			return qfalse;
		}
	}

	// ----------------------------------------------------------------
	// Compile 3D shaders and create PSO
	// ----------------------------------------------------------------
	{
		ID3DBlob *vs       = NULL;
		ID3DBlob *ps       = NULL;
		ID3DBlob *errBlob  = NULL;
		UINT      flags    = 0;

#ifdef _DEBUG
		flags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

		SIZE_T srcLen = strlen(g_worldShaderSource);

		hr = D3DCompile(g_worldShaderSource, srcLen, "dx12_world_shader", NULL, NULL,
		                "VSMain", "vs_5_0", flags, 0, &vs, &errBlob);
		if (FAILED(hr))
		{
			if (errBlob)
			{
				dx12.ri.Printf(PRINT_WARNING, "DX12_SceneInit: VS compile error: %s\n",
				               (const char *)errBlob->GetBufferPointer());
				errBlob->Release();
			}
			dx12.ri.Printf(PRINT_WARNING,
			               "DX12_SceneInit: vertex shader compile failed (0x%08lx)\n", hr);
			dx12Scene.rootSignature3D->Release();
			dx12Scene.rootSignature3D = NULL;
			return qfalse;
		}
		if (errBlob) { errBlob->Release(); errBlob = NULL; }

		hr = D3DCompile(g_worldShaderSource, srcLen, "dx12_world_shader", NULL, NULL,
		                "PSMain", "ps_5_0", flags, 0, &ps, &errBlob);
		if (FAILED(hr))
		{
			if (errBlob)
			{
				dx12.ri.Printf(PRINT_WARNING, "DX12_SceneInit: PS compile error: %s\n",
				               (const char *)errBlob->GetBufferPointer());
				errBlob->Release();
			}
			dx12.ri.Printf(PRINT_WARNING,
			               "DX12_SceneInit: pixel shader compile failed (0x%08lx)\n", hr);
			vs->Release();
			dx12Scene.rootSignature3D->Release();
			dx12Scene.rootSignature3D = NULL;
			return qfalse;
		}
		if (errBlob) { errBlob->Release(); errBlob = NULL; }

		// Input layout matching dx12WorldVertex_t
		D3D12_INPUT_ELEMENT_DESC elems[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0,  0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT,       0, 20, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 28, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 40, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		};

		D3D12_GRAPHICS_PIPELINE_STATE_DESC pso = {};
		pso.InputLayout.pInputElementDescs = elems;
		pso.InputLayout.NumElements        = 5;
		pso.pRootSignature                 = dx12Scene.rootSignature3D;
		pso.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
		pso.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };

		// Rasterizer
		pso.RasterizerState.FillMode              = D3D12_FILL_MODE_SOLID;
		pso.RasterizerState.CullMode              = D3D12_CULL_MODE_BACK;
		pso.RasterizerState.FrontCounterClockwise = TRUE;  // BSP/MD3 geometry uses CCW winding (OpenGL convention)
		pso.RasterizerState.DepthBias             = D3D12_DEFAULT_DEPTH_BIAS;
		pso.RasterizerState.DepthBiasClamp        = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
		pso.RasterizerState.SlopeScaledDepthBias  = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
		pso.RasterizerState.DepthClipEnable       = TRUE;
		pso.RasterizerState.MultisampleEnable      = FALSE;
		pso.RasterizerState.AntialiasedLineEnable  = FALSE;
		pso.RasterizerState.ForcedSampleCount      = 0;
		pso.RasterizerState.ConservativeRaster     = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

		// Blend: opaque
		pso.BlendState.AlphaToCoverageEnable  = FALSE;
		pso.BlendState.IndependentBlendEnable = FALSE;
		{
			D3D12_RENDER_TARGET_BLEND_DESC &rt = pso.BlendState.RenderTarget[0];
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

		pso.DepthStencilState.DepthEnable    = TRUE;
		pso.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
		pso.DepthStencilState.DepthFunc      = D3D12_COMPARISON_FUNC_LESS_EQUAL;
		pso.DepthStencilState.StencilEnable  = FALSE;
		pso.SampleMask                       = UINT_MAX;
		pso.PrimitiveTopologyType            = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		pso.NumRenderTargets                 = 1;
		pso.RTVFormats[0]                    = DXGI_FORMAT_R8G8B8A8_UNORM;
		pso.DSVFormat                        = DXGI_FORMAT_D32_FLOAT;
		pso.SampleDesc.Count                 = 1;

		hr = dx12.device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&dx12Scene.pso3D));
		vs->Release();
		ps->Release();

		if (FAILED(hr))
		{
			dx12.ri.Printf(PRINT_WARNING,
			               "DX12_SceneInit: CreateGraphicsPipelineState failed (0x%08lx)\n", hr);
			dx12Scene.rootSignature3D->Release();
			dx12Scene.rootSignature3D = NULL;
			return qfalse;
		}
	}

	// ----------------------------------------------------------------
	// Translucent 3D PSO – identical to pso3D except:
	//   • alpha blending enabled (SRC_ALPHA / INV_SRC_ALPHA)
	//   • depth writes disabled so translucent surfaces don't occlude
	//     geometry behind them
	// ----------------------------------------------------------------
	{
		ID3D12PipelineState *opaqueTemp = dx12Scene.pso3D;

		// Retrieve the opaque PSO description by recompiling; D3D12 does not
		// expose GetDesc() for PSOs, so we build the translucent PSO
		// by repeating the shader compilation and altering the blend/DS state.
		ID3DBlob *vs       = NULL;
		ID3DBlob *ps       = NULL;
		ID3DBlob *errBlob  = NULL;
		UINT      flags    = 0;

		(void)opaqueTemp; // used only as reminder

#ifdef _DEBUG
		flags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

		SIZE_T srcLen2 = strlen(g_worldShaderSource);

		hr = D3DCompile(g_worldShaderSource, srcLen2, "dx12_world_shader_translucent", NULL, NULL,
		                "VSMain", "vs_5_0", flags, 0, &vs, &errBlob);
		if (SUCCEEDED(hr))
		{
			if (errBlob) { errBlob->Release(); errBlob = NULL; }
			hr = D3DCompile(g_worldShaderSource, srcLen2, "dx12_world_shader_translucent", NULL, NULL,
			                "PSMain", "ps_5_0", flags, 0, &ps, &errBlob);
		}

		if (FAILED(hr) || !vs || !ps)
		{
			if (errBlob) { errBlob->Release(); }
			if (vs) { vs->Release(); }
			if (ps) { ps->Release(); }
			// Non-fatal: fall back to the opaque PSO for translucent surfaces
			dx12Scene.pso3DTranslucent = NULL;
			dx12.ri.Printf(PRINT_DEVELOPER, "DX12_SceneInit: translucent PSO shader compile failed, "
			               "translucent surfaces will render opaque\n");
		}
		else
		{
			if (errBlob) { errBlob->Release(); errBlob = NULL; }

			// Reuse the same input layout as the opaque PSO
			D3D12_INPUT_ELEMENT_DESC elems2[] =
			{
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0,  0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT,       0, 20, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 28, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 40, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			};

			D3D12_GRAPHICS_PIPELINE_STATE_DESC psoT = {};
			psoT.InputLayout.pInputElementDescs = elems2;
			psoT.InputLayout.NumElements        = 5;
			psoT.pRootSignature                 = dx12Scene.rootSignature3D;
			psoT.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
			psoT.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };

			psoT.RasterizerState.FillMode              = D3D12_FILL_MODE_SOLID;
			psoT.RasterizerState.CullMode              = D3D12_CULL_MODE_NONE; // no back-face cull for translucent
			psoT.RasterizerState.FrontCounterClockwise = TRUE;
			psoT.RasterizerState.DepthBias             = D3D12_DEFAULT_DEPTH_BIAS;
			psoT.RasterizerState.DepthBiasClamp        = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
			psoT.RasterizerState.SlopeScaledDepthBias  = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
			psoT.RasterizerState.DepthClipEnable       = TRUE;

			// Alpha blending: SRC_ALPHA * src + (1 – SRC_ALPHA) * dst
			psoT.BlendState.AlphaToCoverageEnable  = FALSE;
			psoT.BlendState.IndependentBlendEnable = FALSE;
			{
				D3D12_RENDER_TARGET_BLEND_DESC &rt = psoT.BlendState.RenderTarget[0];
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

			// Read depth but don't write (translucent surfaces shouldn't occlude)
			psoT.DepthStencilState.DepthEnable    = TRUE;
			psoT.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
			psoT.DepthStencilState.DepthFunc      = D3D12_COMPARISON_FUNC_LESS_EQUAL;
			psoT.DepthStencilState.StencilEnable  = FALSE;

			psoT.SampleMask              = UINT_MAX;
			psoT.PrimitiveTopologyType   = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			psoT.NumRenderTargets        = 1;
			psoT.RTVFormats[0]           = DXGI_FORMAT_R8G8B8A8_UNORM;
			psoT.DSVFormat               = DXGI_FORMAT_D32_FLOAT;
			psoT.SampleDesc.Count        = 1;

			hr = dx12.device->CreateGraphicsPipelineState(&psoT, IID_PPV_ARGS(&dx12Scene.pso3DTranslucent));
			vs->Release();
			ps->Release();

			if (FAILED(hr))
			{
				dx12.ri.Printf(PRINT_WARNING,
				               "DX12_SceneInit: CreateGraphicsPipelineState (translucent) failed (0x%08lx)\n", hr);
				dx12Scene.pso3DTranslucent = NULL;
				// Non-fatal: translucent surfaces will use the opaque PSO
			}
		}
	}

	// ----------------------------------------------------------------
	// Sky 3D PSO – renders sky surfaces at the far plane.
	//   • depth test uses LESS_EQUAL (sky VS forces w=z → NDC depth 1.0)
	//   • depth writes disabled (sky must not occlude world geometry)
	//   • back-face culling disabled
	//   • no blending
	// Uses the dedicated g_skyShaderSource which strips view translation.
	// ----------------------------------------------------------------
	{
		ID3DBlob *vs       = NULL;
		ID3DBlob *ps       = NULL;
		ID3DBlob *errBlob  = NULL;
		UINT      flags    = 0;

#ifdef _DEBUG
		flags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

		SIZE_T skyLen = strlen(g_skyShaderSource);

		hr = D3DCompile(g_skyShaderSource, skyLen, "dx12_sky_shader", NULL, NULL,
		                "VSMain", "vs_5_0", flags, 0, &vs, &errBlob);
		if (SUCCEEDED(hr))
		{
			if (errBlob) { errBlob->Release(); errBlob = NULL; }
			hr = D3DCompile(g_skyShaderSource, skyLen, "dx12_sky_shader", NULL, NULL,
			                "PSMain", "ps_5_0", flags, 0, &ps, &errBlob);
		}

		if (FAILED(hr) || !vs || !ps)
		{
			if (errBlob) { errBlob->Release(); }
			if (vs)      { vs->Release(); }
			if (ps)      { ps->Release(); }
			dx12Scene.pso3DSky = NULL;
			dx12.ri.Printf(PRINT_DEVELOPER, "DX12_SceneInit: sky PSO shader compile failed, "
			               "sky surfaces will be skipped\n");
		}
		else
		{
			if (errBlob) { errBlob->Release(); errBlob = NULL; }

			D3D12_INPUT_ELEMENT_DESC skyElems[] =
			{
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0,  0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT,       0, 20, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 28, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 40, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			};

			D3D12_GRAPHICS_PIPELINE_STATE_DESC psoSky = {};
			psoSky.InputLayout.pInputElementDescs = skyElems;
			psoSky.InputLayout.NumElements        = 5;
			psoSky.pRootSignature                 = dx12Scene.rootSignature3D;
			psoSky.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
			psoSky.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };

			psoSky.RasterizerState.FillMode              = D3D12_FILL_MODE_SOLID;
			psoSky.RasterizerState.CullMode              = D3D12_CULL_MODE_NONE;
			psoSky.RasterizerState.FrontCounterClockwise = TRUE;
			psoSky.RasterizerState.DepthBias             = D3D12_DEFAULT_DEPTH_BIAS;
			psoSky.RasterizerState.DepthBiasClamp        = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
			psoSky.RasterizerState.SlopeScaledDepthBias  = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
			psoSky.RasterizerState.DepthClipEnable       = TRUE;

			// No blending for sky
			psoSky.BlendState.AlphaToCoverageEnable  = FALSE;
			psoSky.BlendState.IndependentBlendEnable = FALSE;
			{
				D3D12_RENDER_TARGET_BLEND_DESC &rt = psoSky.BlendState.RenderTarget[0];
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

			// Depth: test but no write – sky renders at far plane (depth = 1.0)
			psoSky.DepthStencilState.DepthEnable    = TRUE;
			psoSky.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
			psoSky.DepthStencilState.DepthFunc      = D3D12_COMPARISON_FUNC_LESS_EQUAL;
			psoSky.DepthStencilState.StencilEnable  = FALSE;

			psoSky.SampleMask               = UINT_MAX;
			psoSky.PrimitiveTopologyType    = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			psoSky.NumRenderTargets         = 1;
			psoSky.RTVFormats[0]            = DXGI_FORMAT_R8G8B8A8_UNORM;
			psoSky.DSVFormat                = DXGI_FORMAT_D32_FLOAT;
			psoSky.SampleDesc.Count         = 1;

			hr = dx12.device->CreateGraphicsPipelineState(&psoSky, IID_PPV_ARGS(&dx12Scene.pso3DSky));
			vs->Release();
			ps->Release();

			if (FAILED(hr))
			{
				dx12.ri.Printf(PRINT_WARNING,
				               "DX12_SceneInit: CreateGraphicsPipelineState (sky) failed (0x%08lx)\n", hr);
				dx12Scene.pso3DSky = NULL;
			}
		}
	}

	// ----------------------------------------------------------------
	// Per-frame constant buffer
	// Each frame reserves DX12_MAX_CB_SLOTS_PER_FRAME slots:
	//   slot 0        – world / identity model matrix
	//   slots 1..N    – per-entity model matrices
	// This ensures every entity draw command references its own memory
	// location, so the GPU sees the correct matrix for each draw.
	// ----------------------------------------------------------------
	{
		D3D12_HEAP_PROPERTIES heapProps = {};
		D3D12_RESOURCE_DESC   resDesc   = {};

		dx12Scene.cbSlotSize = Align256((UINT)sizeof(dx12SceneConstants_t));
		cbTotalSize          = dx12Scene.cbSlotSize
		                       * DX12_FRAME_COUNT
		                       * DX12_MAX_CB_SLOTS_PER_FRAME;

		heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

		resDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
		resDesc.Width            = cbTotalSize;
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
			IID_PPV_ARGS(&dx12Scene.constantBuffer));

		if (FAILED(hr))
		{
			dx12.ri.Printf(PRINT_WARNING,
			               "DX12_SceneInit: CreateCommittedResource (CB) failed (0x%08lx)\n", hr);
			dx12Scene.pso3D->Release();          dx12Scene.pso3D          = NULL;
			dx12Scene.rootSignature3D->Release(); dx12Scene.rootSignature3D = NULL;
			return qfalse;
		}

		D3D12_RANGE readRange = { 0, 0 };
		hr = dx12Scene.constantBuffer->Map(0, &readRange, (void **)&dx12Scene.cbMapped);
		if (FAILED(hr))
		{
			dx12.ri.Printf(PRINT_WARNING,
			               "DX12_SceneInit: constant buffer Map failed (0x%08lx)\n", hr);
			dx12Scene.constantBuffer->Release();  dx12Scene.constantBuffer  = NULL;
			dx12Scene.pso3D->Release();           dx12Scene.pso3D           = NULL;
			dx12Scene.rootSignature3D->Release(); dx12Scene.rootSignature3D = NULL;
			return qfalse;
		}
	}

	// ----------------------------------------------------------------
	// Per-frame poly vertex upload buffer (world-space decals / effects)
	// ----------------------------------------------------------------
	{
		D3D12_HEAP_PROPERTIES heapProps = {};
		D3D12_RESOURCE_DESC   resDesc   = {};
		D3D12_RANGE           readRange = { 0, 0 };
		UINT64                polyVBSize;

		polyVBSize = (UINT64)DX12_MAX_SCENE_POLYVERTS * sizeof(dx12WorldVertex_t);

		heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

		resDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
		resDesc.Width            = polyVBSize;
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
			IID_PPV_ARGS(&dx12Scene.polyVertexBuffer));

		if (FAILED(hr))
		{
			dx12.ri.Printf(PRINT_WARNING,
			               "DX12_SceneInit: poly VB creation failed (0x%08lx); polys disabled\n", hr);
			// Non-fatal: poly rendering simply won't work
			dx12Scene.polyVertexBuffer = NULL;
			dx12Scene.polyVBMapped     = NULL;
		}
		else
		{
			hr = dx12Scene.polyVertexBuffer->Map(0, &readRange, (void **)&dx12Scene.polyVBMapped);
			if (FAILED(hr))
			{
				dx12Scene.polyVertexBuffer->Release();
				dx12Scene.polyVertexBuffer = NULL;
				dx12Scene.polyVBMapped     = NULL;
			}
		}

		// CPU staging array (always allocate so AddScenePoly can accumulate)
		dx12Scene.polyVerts = (dx12WorldVertex_t *)dx12.ri.Z_Malloc(
			DX12_MAX_SCENE_POLYVERTS * sizeof(dx12WorldVertex_t));
		// If malloc fails the poly pass will be silently skipped
	}

	dx12Scene.initialized = qtrue;
	dx12.ri.Printf(PRINT_ALL, "DX12_SceneInit: 3D scene pipeline ready\n");
	return qtrue;
}

// ---------------------------------------------------------------------------
// DX12_SceneShutdown
// ---------------------------------------------------------------------------

/**
 * @brief DX12_SceneShutdown – release all 3D scene GPU resources.
 */
void DX12_SceneShutdown(void)
{
	if (!dx12Scene.initialized)
	{
		return;
	}

	if (dx12Scene.constantBuffer)
	{
		dx12Scene.constantBuffer->Unmap(0, NULL);
		dx12Scene.constantBuffer->Release();
		dx12Scene.constantBuffer = NULL;
	}

	if (dx12Scene.pso3D)
	{
		dx12Scene.pso3D->Release();
		dx12Scene.pso3D = NULL;
	}

	if (dx12Scene.pso3DTranslucent)
	{
		dx12Scene.pso3DTranslucent->Release();
		dx12Scene.pso3DTranslucent = NULL;
	}

	if (dx12Scene.pso3DSky)
	{
		dx12Scene.pso3DSky->Release();
		dx12Scene.pso3DSky = NULL;
	}

	if (dx12Scene.rootSignature3D)
	{
		dx12Scene.rootSignature3D->Release();
		dx12Scene.rootSignature3D = NULL;
	}

	if (dx12Scene.polyVertexBuffer)
	{
		dx12Scene.polyVertexBuffer->Unmap(0, NULL);
		dx12Scene.polyVertexBuffer->Release();
		dx12Scene.polyVertexBuffer = NULL;
		dx12Scene.polyVBMapped     = NULL;
	}

	if (dx12Scene.polyVerts)
	{
		dx12.ri.Free(dx12Scene.polyVerts);
		dx12Scene.polyVerts = NULL;
	}

	Com_Memset(&dx12Scene, 0, sizeof(dx12Scene));
}

// ---------------------------------------------------------------------------
// DX12_ClearScene / DX12_AddEntityToScene
// ---------------------------------------------------------------------------

/**
 * @brief DX12_ClearScene – reset the per-frame entity and poly lists.
 */
void DX12_ClearScene(void)
{
	dx12Scene.numEntities    = 0;
	dx12Scene.numPolyVerts   = 0;
	dx12Scene.numPolyBatches = 0;
	dx12Scene.numDLights     = 0;
	dx12Scene.numCoronas     = 0;
}

/**
 * @brief DX12_AddEntityToScene – add a ref-entity to the scene list.
 * @param[in] re  Entity to add; ignored if list is full.
 */
void DX12_AddEntityToScene(const refEntity_t *re)
{
	dx12SceneEntity_t *ent;

	if (!re || dx12Scene.numEntities >= DX12_MAX_SCENE_ENTITIES)
	{
		return;
	}

	ent           = &dx12Scene.entities[dx12Scene.numEntities++];
	ent->hModel   = re->hModel;
	ent->origin[0] = re->origin[0];
	ent->origin[1] = re->origin[1];
	ent->origin[2] = re->origin[2];
	ent->axis[0][0] = re->axis[0][0]; ent->axis[0][1] = re->axis[0][1]; ent->axis[0][2] = re->axis[0][2];
	ent->axis[1][0] = re->axis[1][0]; ent->axis[1][1] = re->axis[1][1]; ent->axis[1][2] = re->axis[1][2];
	ent->axis[2][0] = re->axis[2][0]; ent->axis[2][1] = re->axis[2][1]; ent->axis[2][2] = re->axis[2][2];
}

// ---------------------------------------------------------------------------
// DX12_AddScenePoly
// ---------------------------------------------------------------------------

/**
 * @brief DX12_AddScenePoly – buffer a world-space convex polygon.
 * @param[in] hShader   Texture / material handle for this polygon.
 * @param[in] numVerts  Number of polygon vertices (must be ≥ 3).
 * @param[in] verts     Vertices in world space; xyz, st, and byte modulate.
 *
 * Expands the polygon from a triangle fan to an explicit triangle list and
 * appends the result to dx12Scene.polyVerts[].  Consecutive calls that share
 * the same @p hShader are merged into a single batch to minimise draw calls.
 */
void DX12_AddScenePoly(qhandle_t hShader, int numVerts, const polyVert_t *verts)
{
	int numTris;
	int expandedVerts;
	int i;

	if (numVerts < 3 || !verts || !dx12Scene.polyVerts)
	{
		return;
	}

	// Fan has (numVerts - 2) triangles, each needing 3 vertices
	numTris      = numVerts - 2;
	expandedVerts = numTris * 3;

	if (dx12Scene.numPolyVerts + expandedVerts > DX12_MAX_SCENE_POLYVERTS)
	{
		dx12.ri.Printf(PRINT_DEVELOPER, "DX12_AddScenePoly: poly vertex buffer full\n");
		return;
	}

	if (dx12Scene.numPolyBatches >= DX12_MAX_SCENE_POLY_BATCHES)
	{
		dx12.ri.Printf(PRINT_DEVELOPER, "DX12_AddScenePoly: poly batch buffer full\n");
		return;
	}

	// Merge with the current batch if the shader is identical
	if (dx12Scene.numPolyBatches > 0 &&
	    dx12Scene.polyBatches[dx12Scene.numPolyBatches - 1].shaderHandle == hShader)
	{
		dx12Scene.polyBatches[dx12Scene.numPolyBatches - 1].numVerts += expandedVerts;
	}
	else
	{
		dx12ScenePolyBatch_t *batch = &dx12Scene.polyBatches[dx12Scene.numPolyBatches++];
		batch->shaderHandle = hShader;
		batch->firstVert    = dx12Scene.numPolyVerts;
		batch->numVerts     = expandedVerts;
	}

	// Expand fan → triangle list and write into the CPU staging buffer
	for (i = 1; i <= numTris; i++)
	{
		const polyVert_t  *v0  = &verts[0];
		const polyVert_t  *v1  = &verts[i];
		const polyVert_t  *v2  = &verts[i + 1];
		dx12WorldVertex_t *dst = &dx12Scene.polyVerts[dx12Scene.numPolyVerts];
		int                k;
		const polyVert_t  *src[3] = { v0, v1, v2 };

		for (k = 0; k < 3; k++)
		{
			dst[k].xyz[0]    = src[k]->xyz[0];
			dst[k].xyz[1]    = src[k]->xyz[1];
			dst[k].xyz[2]    = src[k]->xyz[2];
			dst[k].st[0]     = src[k]->st[0];
			dst[k].st[1]     = src[k]->st[1];
			dst[k].lm[0]     = 0.0f;
			dst[k].lm[1]     = 0.0f;
			dst[k].normal[0] = 0.0f;
			dst[k].normal[1] = 0.0f;
			dst[k].normal[2] = 1.0f;
			dst[k].color[0]  = src[k]->modulate[0] / 255.0f;
			dst[k].color[1]  = src[k]->modulate[1] / 255.0f;
			dst[k].color[2]  = src[k]->modulate[2] / 255.0f;
			dst[k].color[3]  = src[k]->modulate[3] / 255.0f;
		}

		dx12Scene.numPolyVerts += 3;
	}
}

// ---------------------------------------------------------------------------
// DX12_RenderScene
// ---------------------------------------------------------------------------

/**
 * @brief DX12_RenderScene – render a full 3D scene from a refdef_t.
 * @param[in] fd  Frame parameters: view origin/axis, FOV, and entity list.
 *
 * Render order: sky → opaque world → fog-tagged → entities (sorted back-to-front)
 *               → translucent world surfaces.
 *
 * Improvements over the original pass:
 *   - Issue #11 fix: ibWarnCount/vbWarnCount are file-scope and reset on world load.
 *   - Issue #15 fix: entities sorted by decreasing squared distance (painter's
 *     algorithm for translucent entities; depth test handles opaque ones).
 *   - Issue #10 fix: entity ambient and directed light sampled from the BSP light
 *     grid and passed to the per-entity CB; the HLSL uses them when isEntity=1.
 *   - Issue #6  fix: dynamic lights contribute to the entity ambient light.
 *   - Issue #3  fix: animMap frame and tcMod scroll UV offset passed as per-surface
 *     root constants (DX12_SCENE_ROOT_PARAM_PERSURF) and applied in the shader.
 *   - Issue #4  fix: alphaTestThreshold passed as per-surface root constant and
 *     used in the PS to clip() fragments below the cutoff.
 */
void DX12_RenderScene(const refdef_t *fd)
{
	dx12SceneConstants_t cb;
	float view[4][4], proj[4][4], viewProj[4][4];
	D3D12_GPU_VIRTUAL_ADDRESS cbBaseGpuVA; // GPU VA of slot 0 in this frame
	UINT  cbBaseSlot;                      // Index of the world CB slot

	if (!dx12Scene.initialized || !dx12.frameOpen)
	{
		return;
	}

	if (!fd)
	{
		return;
	}

	// ----------------------------------------------------------------
	// 0a. Store scene time for animMap / tcMod scroll animation.
	// ----------------------------------------------------------------
	g_sceneTimeMs = fd->time;

	// ----------------------------------------------------------------
	// 0b. Tick global fog transition (matches GL tr_main.c ~line 622)
	// ----------------------------------------------------------------
	if (dx12World.loaded && dx12World.globalFogTransEndTime > 0)
	{
		int now = dx12.ri.Milliseconds();

		if (now < dx12World.globalFogTransEndTime)
		{
			int   fadeTime = dx12World.globalFogTransEndTime - dx12World.globalFogTransStartTime;
			float lerpPos  = (float)(now - dx12World.globalFogTransStartTime) / (float)fadeTime;
			int   i;

			for (i = 0; i < 3; i++)
			{
				dx12World.globalFogColor[i] = dx12World.globalFogTransStartFog[i]
				                              + (dx12World.globalFogTransEndFog[i] - dx12World.globalFogTransStartFog[i]) * lerpPos;
			}
			dx12World.globalFogDepth = dx12World.globalFogTransStartFog[3]
			                           + (dx12World.globalFogTransEndFog[3] - dx12World.globalFogTransStartFog[3]) * lerpPos;
			dx12World.globalFogActive = qtrue;
		}
		else
		{
			// Transition complete – snap to end values
			dx12World.globalFogColor[0]       = dx12World.globalFogTransEndFog[0];
			dx12World.globalFogColor[1]       = dx12World.globalFogTransEndFog[1];
			dx12World.globalFogColor[2]       = dx12World.globalFogTransEndFog[2];
			dx12World.globalFogDepth          = dx12World.globalFogTransEndFog[3];
			dx12World.globalFogActive         = (dx12World.globalFogDepth > 0.0f) ? qtrue : qfalse;
			dx12World.globalFogTransEndTime   = 0;
			dx12World.globalFogTransStartTime = 0;
		}
	}

	// ----------------------------------------------------------------
	// 1.  Build view and projection matrices
	// ----------------------------------------------------------------
	BuildViewMatrix(view, fd->vieworg, (const vec3_t *)fd->viewaxis);
	BuildProjMatrix(proj, fd->fov_x, fd->fov_y, DX12_SCENE_NEAR, DX12_SCENE_FAR);
	Mat4Mul(viewProj, view, proj);

	// ----------------------------------------------------------------
	// 2.  Update the world/identity constant-buffer slot
	//     cbBaseSlot == slot for world geometry (identity model matrix)
	//     cbBaseSlot + 1 + i == dedicated slot for entity[i]
	// ----------------------------------------------------------------
	cbBaseSlot  = (UINT)dx12.frameIndex * (UINT)DX12_MAX_CB_SLOTS_PER_FRAME;
	cbBaseGpuVA = dx12Scene.constantBuffer->GetGPUVirtualAddress()
	              + (D3D12_GPU_VIRTUAL_ADDRESS)cbBaseSlot * dx12Scene.cbSlotSize;

	// Identity model matrix for world geometry
	Mat4Identity(cb.modelMatrix);

	// Copy viewProj
	{
		int i, j;

		for (i = 0; i < 4; i++)
		{
			for (j = 0; j < 4; j++)
			{
				cb.viewProj[i][j] = viewProj[i][j];
			}
		}
	}

	cb.cameraPos[0] = fd->vieworg[0];
	cb.cameraPos[1] = fd->vieworg[1];
	cb.cameraPos[2] = fd->vieworg[2];
	cb.cameraPos[3] = 1.0f;

	// Populate fog fields from global-fog override or clear them
	if (dx12World.globalFogActive && dx12World.globalFogDepth > 0.0f)
	{
		cb.fogColor[0] = dx12World.globalFogColor[0];
		cb.fogColor[1] = dx12World.globalFogColor[1];
		cb.fogColor[2] = dx12World.globalFogColor[2];
		cb.fogColor[3] = 1.0f;
		cb.fogStart    = 0.0f;
		cb.fogEnd      = dx12World.globalFogDepth;
		cb.fogEnabled  = 1.0f;
	}
	else
	{
		cb.fogColor[0] = 0.0f;
		cb.fogColor[1] = 0.0f;
		cb.fogColor[2] = 0.0f;
		cb.fogColor[3] = 1.0f;
		cb.fogStart    = 0.0f;
		cb.fogEnd      = 0.0f;
		cb.fogEnabled  = 0.0f;
	}
	// ----------------------------------------------------------------
	// Compute overbright factor matching GL renderer1 behaviour:
	//   factor = 2^(r_mapOverBrightBits - r_overBrightBits)
	// Default: 2^(2-0) = 4.  Clamped to [1, 8].
	// ----------------------------------------------------------------
	{
		cvar_t *r_mapOB = dx12.ri.Cvar_Get("r_mapOverBrightBits", "2", 0);
		cvar_t *r_ob    = dx12.ri.Cvar_Get("r_overBrightBits",    "0", 0);
		int     shift   = (r_mapOB ? r_mapOB->integer : 2) - (r_ob ? r_ob->integer : 0);

		if (shift < 0) { shift = 0; }
		if (shift > 3) { shift = 3; }
		cb.overBrightFactor = (float)(1 << shift);
	}

	// World CB has no entity light (entity light fields are 0 for world surfaces)
	cb.entityAmbient[0]  = 0.0f; cb.entityAmbient[1]  = 0.0f;
	cb.entityAmbient[2]  = 0.0f; cb.entityAmbient[3]  = 0.0f;
	cb.entityDirected[0] = 0.0f; cb.entityDirected[1] = 0.0f;
	cb.entityDirected[2] = 0.0f; cb.entityDirected[3] = 0.0f;
	cb.entityLightDir[0] = 0.0f; cb.entityLightDir[1] = 0.0f;
	cb.entityLightDir[2] = 1.0f; cb.entityLightDir[3] = 0.0f;

	SCN_UpdateCB(cbBaseSlot, &cb);

	// ----------------------------------------------------------------
	// 2b. Pre-compute per-entity light grid samples and distance sort
	//     (Issues #10, #15)
	//
	//     We sort entity indices by decreasing squared distance from the
	//     camera.  This is the painter's-algorithm order for translucent
	//     entities; depth test ensures opaque entities are unaffected.
	//
	//     Light grid sampling (DX12_SampleLightGrid) is done here so
	//     the sampled values are available when building the entity CB.
	//     Dynamic lights (dlights) contribute to entity ambient light
	//     with a simple 1/r² attenuation (Issue #6 partial fix).
	// ----------------------------------------------------------------
	{
		int i, j;

		// Build entity distance array for sorting
		for (i = 0; i < dx12Scene.numEntities; i++)
		{
			dx12SceneEntity_t *ent = &dx12Scene.entities[i];
			float dx, dy, dz;

			dx = ent->origin[0] - fd->vieworg[0];
			dy = ent->origin[1] - fd->vieworg[1];
			dz = ent->origin[2] - fd->vieworg[2];
			ent->distSq = dx * dx + dy * dy + dz * dz;

			// Sample light grid for this entity's world position
			DX12_SampleLightGrid(ent->origin, ent->ambientLight,
			                     ent->directedLight, ent->lightDir);

			// Apply dynamic light contributions to entity ambient (Issue #6)
			for (j = 0; j < dx12Scene.numDLights; j++)
			{
				const dx12DLight_t *dl = &dx12Scene.dlights[j];
				float dlDx, dlDy, dlDz, distSqDl, rSq, atten;

				dlDx    = ent->origin[0] - dl->origin[0];
				dlDy    = ent->origin[1] - dl->origin[1];
				dlDz    = ent->origin[2] - dl->origin[2];
				distSqDl = dlDx * dlDx + dlDy * dlDy + dlDz * dlDz;
				rSq      = dl->radius * dl->radius;

				if (distSqDl >= rSq)
				{
					continue; // outside light radius
				}

				atten = (1.0f - distSqDl / rSq) * dl->intensity;
				ent->ambientLight[0] += dl->color[0] * atten;
				ent->ambientLight[1] += dl->color[1] * atten;
				ent->ambientLight[2] += dl->color[2] * atten;
			}
		}

		// Insertion sort: entities ordered by decreasing distSq (furthest first)
		for (i = 1; i < dx12Scene.numEntities; i++)
		{
			dx12SceneEntity_t tmp = dx12Scene.entities[i];

			j = i - 1;
			while (j >= 0 && dx12Scene.entities[j].distSq < tmp.distSq)
			{
				dx12Scene.entities[j + 1] = dx12Scene.entities[j];
				j--;
			}
			dx12Scene.entities[j + 1] = tmp;
		}
	}

	// ----------------------------------------------------------------
	// 3.  Switch to the 3D pipeline
	// ----------------------------------------------------------------
	// Flush any pending 2D batch before switching pipelines
	DX12_Flush2D();

	dx12.commandList->SetGraphicsRootSignature(dx12Scene.rootSignature3D);
	dx12.commandList->SetPipelineState(dx12Scene.pso3D);

	// Re-bind the SRV heap (already bound from BeginFrame, but be explicit)
	{
		ID3D12DescriptorHeap *heaps[1] = { dx12.srvHeap };
		dx12.commandList->SetDescriptorHeaps(1, heaps);
	}

	// Set primitive topology for world geometry
	dx12.commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	// Restore full-screen viewport and scissor for the 3D pass
	dx12.commandList->RSSetViewports(1, &dx12.viewport);
	dx12.commandList->RSSetScissorRects(1, &dx12.scissorRect);

	// Initialise per-surface root constants to safe defaults (world mode)
	{
		dx12PerSurfConstants_t pscDefault = {};

		dx12.commandList->SetGraphicsRoot32BitConstants(DX12_SCENE_ROOT_PARAM_PERSURF,
		                                                4, &pscDefault, 0);
	}

	// ----------------------------------------------------------------
	// 4.  Bind world vertex + index buffers (if world is loaded)
	// ----------------------------------------------------------------
	if (dx12World.loaded && dx12World.vertexBuffer && dx12World.indexBuffer
	    && dx12World.numVertices > 0 && dx12World.numIndexes > 0)
	{
		D3D12_VERTEX_BUFFER_VIEW vbv = {};
		D3D12_INDEX_BUFFER_VIEW  ibv = {};

		vbv.BufferLocation = dx12World.vertexBuffer->GetGPUVirtualAddress();
		vbv.SizeInBytes    = dx12World.numVertices * (UINT)sizeof(dx12WorldVertex_t);
		vbv.StrideInBytes  = (UINT)sizeof(dx12WorldVertex_t);

		ibv.BufferLocation = dx12World.indexBuffer->GetGPUVirtualAddress();
		ibv.SizeInBytes    = dx12World.numIndexes * (UINT)sizeof(int);
		ibv.Format         = DXGI_FORMAT_R32_UINT;

		dx12.commandList->IASetVertexBuffers(0, 1, &vbv);
		dx12.commandList->IASetIndexBuffer(&ibv);

		// ----------------------------------------------------------------
		// 5a. Sky surfaces
		//
		// TODO: Implement proper skybox / sky-portal rendering.  Until then
		// sky surfaces are intentionally skipped so the background clear
		// colour shows through — this is preferable to a pure-white blob.
		// ----------------------------------------------------------------
		/* sky surfaces intentionally not drawn here */
		// ----------------------------------------------------------------
		// 5a. Sky surfaces – rendered at far plane using sky PSO.
		// ----------------------------------------------------------------
		if (dx12Scene.pso3DSky)
		{
			int i;

			for (i = 0; i < dx12World.numDrawSurfs; i++)
			{
				const dx12DrawSurf_t *ds  = &dx12World.drawSurfs[i];
				const dx12Material_t *mat = DX12_GetMaterial(ds->materialHandle);

				if (mat && mat->isSky)
				{
					DX12_DrawSkySurface(ds, cbBaseGpuVA);
				}
			}
		}

		// ----------------------------------------------------------------
		// 5b. Opaque world surfaces (not sky, not translucent, not fog)
		// ----------------------------------------------------------------
		{
			int i;

			for (i = 0; i < dx12World.numDrawSurfs; i++)
			{
				const dx12DrawSurf_t  *ds  = &dx12World.drawSurfs[i];
				const dx12Material_t  *mat = DX12_GetMaterial(ds->materialHandle);

				if (!mat || (!mat->isSky && !mat->isTranslucent && !mat->isFog))
				{
					SCN_DrawSurface(ds, cbBaseGpuVA);
				}
			}
		}

		// ----------------------------------------------------------------
		// 5c. Fog-tagged surfaces
		// ----------------------------------------------------------------
		{
			int i;

			for (i = 0; i < dx12World.numDrawSurfs; i++)
			{
				const dx12DrawSurf_t  *ds  = &dx12World.drawSurfs[i];
				const dx12Material_t  *mat = DX12_GetMaterial(ds->materialHandle);

				if (mat && mat->isFog)
				{
					SCN_DrawSurface(ds, cbBaseGpuVA);
				}
			}
		}
	}

	// ----------------------------------------------------------------
	// 6.  Entities (sorted back-to-front by distance; Issue #15 fix)
	//     Each entity uses its own dedicated CB slot so all recorded
	//     draw commands reference unique CB memory.
	//     Entity ambient + directed light from light grid (Issue #10).
	// ----------------------------------------------------------------
	{
		int i;

		for (i = 0; i < dx12Scene.numEntities; i++)
		{
			dx12SceneEntity_t    *ent = &dx12Scene.entities[i];
			dx12SceneConstants_t  entCB;
			dx12PerSurfConstants_t entPsc = {};
			UINT                  entSlot;
			D3D12_GPU_VIRTUAL_ADDRESS entCBGpuVA;

			// Clamp to the allocated range to guard against overflow
			if (i >= DX12_MAX_SCENE_ENTITIES)
			{
				break;
			}

			// Each entity gets its own unique slot within this frame
			entSlot    = cbBaseSlot + 1u + (UINT)i;
			entCBGpuVA = dx12Scene.constantBuffer->GetGPUVirtualAddress()
			             + (D3D12_GPU_VIRTUAL_ADDRESS)entSlot * dx12Scene.cbSlotSize;

			// Build per-entity model matrix
			BuildModelMatrix(entCB.modelMatrix, ent->origin,
			                 (const vec3_t *)ent->axis);

			// Copy the shared viewProj and cameraPos
			{
				int r, c;

				for (r = 0; r < 4; r++)
				{
					for (c = 0; c < 4; c++)
					{
						entCB.viewProj[r][c] = cb.viewProj[r][c];
					}
				}
			}

			entCB.cameraPos[0] = cb.cameraPos[0];
			entCB.cameraPos[1] = cb.cameraPos[1];
			entCB.cameraPos[2] = cb.cameraPos[2];
			entCB.cameraPos[3] = cb.cameraPos[3];

			// Copy fog parameters so entities are also fogged
			entCB.fogColor[0] = cb.fogColor[0];
			entCB.fogColor[1] = cb.fogColor[1];
			entCB.fogColor[2] = cb.fogColor[2];
			entCB.fogColor[3] = cb.fogColor[3];
			entCB.fogStart         = cb.fogStart;
			entCB.fogEnd           = cb.fogEnd;
			entCB.fogEnabled       = cb.fogEnabled;
			entCB.overBrightFactor = cb.overBrightFactor;

			// Entity light from grid (Issue #10 fix)
			entCB.entityAmbient[0]  = ent->ambientLight[0];
			entCB.entityAmbient[1]  = ent->ambientLight[1];
			entCB.entityAmbient[2]  = ent->ambientLight[2];
			entCB.entityAmbient[3]  = 0.0f;
			entCB.entityDirected[0] = ent->directedLight[0];
			entCB.entityDirected[1] = ent->directedLight[1];
			entCB.entityDirected[2] = ent->directedLight[2];
			entCB.entityDirected[3] = 0.0f;
			entCB.entityLightDir[0] = ent->lightDir[0];
			entCB.entityLightDir[1] = ent->lightDir[1];
			entCB.entityLightDir[2] = ent->lightDir[2];
			entCB.entityLightDir[3] = 0.0f;

			// Write to the entity's dedicated slot
			SCN_UpdateCB(entSlot, &entCB);

			// Per-surface root constants for entity: isEntity=1, no UV offset,
			// no alpha test (entity models rarely use alphaFunc)
			entPsc.uvOffsetU          = 0.0f;
			entPsc.uvOffsetV          = 0.0f;
			entPsc.alphaTestThreshold = 0.0f;
			entPsc.isEntity           = 1.0f;

			dx12.commandList->SetGraphicsRoot32BitConstants(DX12_SCENE_ROOT_PARAM_PERSURF,
			                                                4, &entPsc, 0);

			// Draw the entity's model surfaces
			DX12_DrawEntity(ent, entCBGpuVA);
		}

		// Reset per-surface root constants back to world defaults after entity pass
		{
			dx12PerSurfConstants_t pscWorld = {};

			dx12.commandList->SetGraphicsRoot32BitConstants(DX12_SCENE_ROOT_PARAM_PERSURF,
			                                                4, &pscWorld, 0);
		}
	}

	// ----------------------------------------------------------------
	// 7a. Persistent decals – re-submitted every frame with time-based fade.
	// ----------------------------------------------------------------
	{
		int i, p;

		for (i = 0; i < dx12Scene.numDecals; i++)
		{
			dx12Decal_t *dec = &dx12Scene.decals[i];
			float        fadeAlpha = 1.0f;

			// Check expiry (fadeEndTime==0 means permanent)
			if (dec->fadeEndTime > 0)
			{
				if (fd->time >= dec->fadeEndTime)
				{
					continue; // expired – skip
				}
				if (fd->time >= dec->fadeStartTime && dec->fadeEndTime > dec->fadeStartTime)
				{
					int elapsed  = fd->time - dec->fadeStartTime;
					int duration = dec->fadeEndTime - dec->fadeStartTime;

					fadeAlpha = 1.0f - (float)elapsed / (float)duration;
					if (fadeAlpha < 0.0f)
					{
						fadeAlpha = 0.0f;
					}
				}
			}

			if (dec->numVerts < 3)
			{
				continue;
			}

			// Build a local copy with alpha scaled by fade factor
			if (fadeAlpha < 0.9999f)
			{
				polyVert_t fadedVerts[DX12_MAX_DECAL_VERTS];

				for (p = 0; p < dec->numVerts; p++)
				{
					fadedVerts[p]              = dec->verts[p];
					fadedVerts[p].modulate[3]  = (byte)((float)dec->verts[p].modulate[3] * fadeAlpha);
				}
				DX12_AddScenePoly(dec->hShader, dec->numVerts, fadedVerts);
			}
			else
			{
				DX12_AddScenePoly(dec->hShader, dec->numVerts, dec->verts);
			}
		}
	}

	// ----------------------------------------------------------------
	// 7b. World-space polys (decals, effects)
	//     Rendered after entities, before translucent world surfaces.
	//     Uses the world CB slot (identity model matrix).
	// ----------------------------------------------------------------
	if (dx12Scene.numPolyVerts > 0 && dx12Scene.polyVerts &&
	    dx12Scene.polyVertexBuffer && dx12Scene.polyVBMapped)
	{
		int b;
		D3D12_VERTEX_BUFFER_VIEW pvbv = {};

		// Upload staging buffer to the GPU upload-heap VB
		memcpy(dx12Scene.polyVBMapped, dx12Scene.polyVerts,
		       (size_t)dx12Scene.numPolyVerts * sizeof(dx12WorldVertex_t));

		pvbv.BufferLocation = dx12Scene.polyVertexBuffer->GetGPUVirtualAddress();
		pvbv.SizeInBytes    = (UINT)dx12Scene.numPolyVerts * (UINT)sizeof(dx12WorldVertex_t);
		pvbv.StrideInBytes  = (UINT)sizeof(dx12WorldVertex_t);

		dx12.commandList->IASetVertexBuffers(0, 1, &pvbv);

		for (b = 0; b < dx12Scene.numPolyBatches; b++)
		{
			const dx12ScenePolyBatch_t *batch   = &dx12Scene.polyBatches[b];
			dx12Texture_t              *polyTex;
			D3D12_GPU_DESCRIPTOR_HANDLE srvPoly;
			dx12PerSurfConstants_t      polyPsc = {};

			if (batch->numVerts <= 0)
			{
				continue;
			}

			// Resolve poly texture; fall back to white if missing
			polyTex = DX12_GetTexture(batch->shaderHandle);
			if (polyTex && polyTex->resource)
			{
				srvPoly = polyTex->gpuHandle;
			}
			else
			{
				srvPoly = dx12.srvHeap->GetGPUDescriptorHandleForHeapStart();
			}

			dx12.commandList->SetGraphicsRootConstantBufferView(DX12_SCENE_ROOT_PARAM_CB,
			                                                    cbBaseGpuVA);
			dx12.commandList->SetGraphicsRoot32BitConstants(DX12_SCENE_ROOT_PARAM_PERSURF,
			                                                4, &polyPsc, 0);
			dx12.commandList->SetGraphicsRootDescriptorTable(DX12_SCENE_ROOT_PARAM_DIFFUSE,
			                                                 srvPoly);
			dx12.commandList->SetGraphicsRootDescriptorTable(DX12_SCENE_ROOT_PARAM_LIGHTMAP,
			                                                 srvPoly);

			dx12.commandList->DrawInstanced((UINT)batch->numVerts, 1,
			                                (UINT)batch->firstVert, 0);
		}

		// Restore world VB/IB for the translucent pass below
		if (dx12World.loaded && dx12World.vertexBuffer && dx12World.indexBuffer)
		{
			D3D12_VERTEX_BUFFER_VIEW wvbv = {};
			D3D12_INDEX_BUFFER_VIEW  wibv = {};

			wvbv.BufferLocation = dx12World.vertexBuffer->GetGPUVirtualAddress();
			wvbv.SizeInBytes    = dx12World.numVertices * (UINT)sizeof(dx12WorldVertex_t);
			wvbv.StrideInBytes  = (UINT)sizeof(dx12WorldVertex_t);

			wibv.BufferLocation = dx12World.indexBuffer->GetGPUVirtualAddress();
			wibv.SizeInBytes    = dx12World.numIndexes * (UINT)sizeof(int);
			wibv.Format         = DXGI_FORMAT_R32_UINT;

			dx12.commandList->IASetVertexBuffers(0, 1, &wvbv);
			dx12.commandList->IASetIndexBuffer(&wibv);
		}
	}

	// ----------------------------------------------------------------
	// 7c. Translucent / alpha world surfaces (back-to-front order)
	//     Switch to the translucent PSO (alpha-blend, no depth-write).
	// ----------------------------------------------------------------
	if (dx12World.loaded && dx12World.vertexBuffer && dx12World.indexBuffer
	    && dx12World.numVertices > 0 && dx12World.numIndexes > 0)
	{
		int i;

		// Rebind world VB/IB – entity and poly draws above may have changed
		// the active input assembler buffers.
		{
			D3D12_VERTEX_BUFFER_VIEW tvbv = {};
			D3D12_INDEX_BUFFER_VIEW  tibv = {};

			tvbv.BufferLocation = dx12World.vertexBuffer->GetGPUVirtualAddress();
			tvbv.SizeInBytes    = dx12World.numVertices * (UINT)sizeof(dx12WorldVertex_t);
			tvbv.StrideInBytes  = (UINT)sizeof(dx12WorldVertex_t);

			tibv.BufferLocation = dx12World.indexBuffer->GetGPUVirtualAddress();
			tibv.SizeInBytes    = dx12World.numIndexes * (UINT)sizeof(int);
			tibv.Format         = DXGI_FORMAT_R32_UINT;

			dx12.commandList->IASetVertexBuffers(0, 1, &tvbv);
			dx12.commandList->IASetIndexBuffer(&tibv);
		}

		// Switch to translucent PSO if available
		if (dx12Scene.pso3DTranslucent)
		{
			dx12.commandList->SetPipelineState(dx12Scene.pso3DTranslucent);
		}

		for (i = dx12World.numDrawSurfs - 1; i >= 0; i--)
		{
			const dx12DrawSurf_t  *ds  = &dx12World.drawSurfs[i];
			const dx12Material_t  *mat = DX12_GetMaterial(ds->materialHandle);

			if (mat && mat->isTranslucent)
			{
				SCN_DrawSurface(ds, cbBaseGpuVA);
			}
		}

		// Restore opaque PSO for subsequent passes
		dx12.commandList->SetPipelineState(dx12Scene.pso3D);
	}

	// ----------------------------------------------------------------
	// 8.  Restore 2D pipeline for subsequent UI rendering
	// ----------------------------------------------------------------
	dx12.commandList->SetGraphicsRootSignature(dx12.rootSignature);
	dx12.commandList->SetPipelineState(dx12.pipelineState);
	dx12.commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
}

// ---------------------------------------------------------------------------
// DX12_AddDecalToScene / DX12_ClearDecals
// ---------------------------------------------------------------------------

/**
 * @brief DX12_AddDecalToScene – store an already-projected world-space decal polygon.
 *
 * @p verts must have xyz, st, and base modulate already computed (by
 * RE_DX12_ProjectDecal).  The decal is re-submitted to the poly system on
 * every DX12_RenderScene call, with the alpha channel scaled by the
 * time-based fade factor, until it expires or DX12_ClearDecals() is called.
 */
void DX12_AddDecalToScene(qhandle_t hShader, int numVerts, const polyVert_t *verts,
                          int fadeStartTime, int fadeEndTime)
{
	dx12Decal_t *dec;
	int          i;
	int          clampedVerts;

	if (!verts || numVerts < 3)
	{
		return;
	}

	if (dx12Scene.numDecals >= DX12_MAX_DECALS)
	{
		dx12.ri.Printf(PRINT_DEVELOPER,
		               "DX12_AddDecalToScene: dropping decal, reached MAX_DECALS (%d)\n",
		               DX12_MAX_DECALS);
		return;
	}

	clampedVerts = numVerts < DX12_MAX_DECAL_VERTS ? numVerts : DX12_MAX_DECAL_VERTS;

	dec                = &dx12Scene.decals[dx12Scene.numDecals++];
	dec->hShader       = hShader;
	dec->numVerts      = clampedVerts;
	dec->fadeStartTime = fadeStartTime;
	dec->fadeEndTime   = fadeEndTime;

	for (i = 0; i < clampedVerts; i++)
	{
		dec->verts[i] = verts[i];
	}
}

/**
 * @brief DX12_ClearDecals – remove all persistent decals (called by RE_DX12_ClearDecals).
 */
void DX12_ClearDecals(void)
{
	dx12Scene.numDecals = 0;
}

#endif // _WIN32
