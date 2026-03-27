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
 *   b0 – SceneConstants (CBV): viewProj, modelMatrix, cameraPos
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
 */
static const char g_worldShaderSource[] =
	"cbuffer SceneConstants : register(b0)\n"
	"{\n"
	"    float4x4 viewProj;\n"
	"    float4x4 modelMatrix;\n"
	"    float4   cameraPos;\n"
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
	"    float4 pos    : SV_POSITION;\n"
	"    float2 uv     : TEXCOORD0;\n"
	"    float2 lm     : TEXCOORD1;\n"
	"    float4 color  : COLOR;\n"
	"};\n"
	"\n"
	"PSInput VSMain(VSInput input)\n"
	"{\n"
	"    PSInput o;\n"
	"    float4 worldPos = mul(modelMatrix, float4(input.pos, 1.0));\n"
	"    o.pos   = mul(viewProj, worldPos);\n"
	"    o.uv    = input.uv;\n"
	"    o.lm    = input.lm;\n"
	"    o.color = input.color;\n"
	"    return o;\n"
	"}\n"
	"\n"
	"float4 PSMain(PSInput input) : SV_TARGET\n"
	"{\n"
	"    float4 diffuse  = g_diffuse.Sample(g_sampler, input.uv);\n"
	"    float4 lightmap = g_lightmap.Sample(g_sampler, input.lm);\n"
	"    return diffuse * lightmap * input.color;\n"
	"}\n";

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

dx12SceneState_t dx12Scene;

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
 * @param[in] slot  Frame slot index (0 … DX12_FRAME_COUNT-1).
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
 * @brief Bind SRVs for one draw surface and issue a DrawIndexedInstanced call.
 *
 * @param[in] ds      World draw surface descriptor.
 * @param[in] cbGpuVA GPU virtual address of the constant buffer slot.
 */
static void SCN_DrawSurface(const dx12DrawSurf_t *ds, D3D12_GPU_VIRTUAL_ADDRESS cbGpuVA)
{
	dx12Material_t *mat      = NULL;
	dx12Texture_t  *diffTex  = NULL;
	dx12Texture_t  *lmTex    = NULL;
	D3D12_GPU_DESCRIPTOR_HANDLE srvBase;

	if (ds->numIndexes <= 0 || ds->numVertices <= 0)
	{
		return;
	}

	// --- Bind constant buffer (root CBV at slot 0) ---
	dx12.commandList->SetGraphicsRootConstantBufferView(0, cbGpuVA);

	// --- Resolve textures ---
	mat = DX12_GetMaterial(ds->materialHandle);

	if (mat && mat->numStages > 0 && mat->stages[0].active)
	{
		diffTex = DX12_GetTexture(mat->stages[0].texHandle);
	}

	// Lightmap handle is stored in dx12World per surface-shader entry.
	// For now, use the material's stage-0 texture as diffuse and fall back
	// to the same texture for the lightmap if no dedicated lightmap is wired.
	// (Full lightmap binding requires model-specific data beyond current scope.)

	// --- Bind SRV descriptor table (slot 1) ---
	{
		D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = {};

		if (diffTex && diffTex->resource)
		{
			srvHandle = diffTex->gpuHandle;
		}
		else
		{
			// Fallback: first SRV in heap (should be a white/default texture)
			srvHandle = dx12.srvHeap->GetGPUDescriptorHandleForHeapStart();
		}

		// Bind diffuse (t0) via descriptor table root param 1
		dx12.commandList->SetGraphicsRootDescriptorTable(1, srvHandle);

		// Bind lightmap (t1) – use diffuse as fallback if no dedicated lm
		if (ds->fogIndex > 0 && ds->fogIndex < dx12World.numFogs)
		{
			qhandle_t lmH = dx12World.fogs[ds->fogIndex].materialHandle;

			lmTex = DX12_GetTexture(lmH);
		}

		srvBase = lmTex ? lmTex->gpuHandle : srvHandle;
		dx12.commandList->SetGraphicsRootDescriptorTable(2, srvBase);
	}

	// --- Issue draw call ---
	dx12.commandList->DrawIndexedInstanced(
		(UINT)ds->numIndexes,   // IndexCountPerInstance
		1,                      // InstanceCount
		(UINT)ds->firstIndex,   // StartIndexLocation
		ds->firstVertex,        // BaseVertexLocation
		0                       // StartInstanceLocation
		);
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

	// ----------------------------------------------------------------
	// Root Signature for 3D rendering
	//   Param 0: Root CBV at b0  (VS + PS visible)
	//   Param 1: Descriptor table – 1 SRV at t0 (diffuse)
	//   Param 2: Descriptor table – 1 SRV at t1 (lightmap)
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

		D3D12_ROOT_PARAMETER params[3] = {};

		// Param 0: root CBV
		params[0].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
		params[0].Descriptor.ShaderRegister = 0; // b0
		params[0].Descriptor.RegisterSpace  = 0;
		params[0].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

		// Param 1: diffuse SRV table
		params[1].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		params[1].DescriptorTable.NumDescriptorRanges = 1;
		params[1].DescriptorTable.pDescriptorRanges   = &srvRange0;
		params[1].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_PIXEL;

		// Param 2: lightmap SRV table
		params[2].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		params[2].DescriptorTable.NumDescriptorRanges = 1;
		params[2].DescriptorTable.pDescriptorRanges   = &srvRange1;
		params[2].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_PIXEL;

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
		rsd.NumParameters     = 3;
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
	// Per-frame constant buffer (DX12_FRAME_COUNT slots, 256-byte aligned)
	// ----------------------------------------------------------------
	{
		D3D12_HEAP_PROPERTIES heapProps = {};
		D3D12_RESOURCE_DESC   resDesc   = {};

		dx12Scene.cbSlotSize = Align256((UINT)sizeof(dx12SceneConstants_t));
		cbTotalSize          = dx12Scene.cbSlotSize * DX12_FRAME_COUNT;

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
		dx12Scene.polyVerts = (dx12WorldVertex_t *)malloc(
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
		free(dx12Scene.polyVerts);
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
 * Render order: sky → opaque world → fog-tagged → entities → translucent.
 */
void DX12_RenderScene(const refdef_t *fd)
{
	dx12SceneConstants_t cb;
	float view[4][4], proj[4][4], viewProj[4][4];
	D3D12_GPU_VIRTUAL_ADDRESS cbGpuVA;
	UINT  cbSlot;

	if (!dx12Scene.initialized || !dx12.frameOpen)
	{
		return;
	}

	if (!fd)
	{
		return;
	}

	// ----------------------------------------------------------------
	// 1.  Build view and projection matrices
	// ----------------------------------------------------------------
	BuildViewMatrix(view, fd->vieworg, (const vec3_t *)fd->viewaxis);
	BuildProjMatrix(proj, fd->fov_x, fd->fov_y, DX12_SCENE_NEAR, DX12_SCENE_FAR);
	Mat4Mul(viewProj, view, proj);

	// ----------------------------------------------------------------
	// 2.  Update per-frame constant buffer
	// ----------------------------------------------------------------
	cbSlot = dx12.frameIndex;
	cbGpuVA = dx12Scene.constantBuffer->GetGPUVirtualAddress()
	          + (D3D12_GPU_VIRTUAL_ADDRESS)cbSlot * dx12Scene.cbSlotSize;

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

	SCN_UpdateCB(cbSlot, &cb);

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
		// ----------------------------------------------------------------
		{
			int i;

			for (i = 0; i < dx12World.numDrawSurfs; i++)
			{
				const dx12DrawSurf_t *ds = &dx12World.drawSurfs[i];

				if (ds->isSky)
				{
					SCN_DrawSurface(ds, cbGpuVA);
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
				const dx12DrawSurf_t *ds = &dx12World.drawSurfs[i];

				if (!ds->isSky && !ds->isTranslucent && !ds->isFog)
				{
					SCN_DrawSurface(ds, cbGpuVA);
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
				const dx12DrawSurf_t *ds = &dx12World.drawSurfs[i];

				if (ds->isFog)
				{
					SCN_DrawSurface(ds, cbGpuVA);
				}
			}
		}
	}

	// ----------------------------------------------------------------
	// 6.  Entities
	// ----------------------------------------------------------------
	{
		int i;

		for (i = 0; i < dx12Scene.numEntities; i++)
		{
			dx12SceneEntity_t *ent = &dx12Scene.entities[i];
			dx12SceneConstants_t entCB;
			D3D12_GPU_VIRTUAL_ADDRESS entCBGpuVA;

			// Build per-entity model matrix
			BuildModelMatrix(entCB.modelMatrix, ent->origin,
			                 (const vec3_t *)ent->axis);

			// Copy the same viewProj
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

			// Reuse slot (same frame slot; entities overwrite world CB
			// but world is already drawn above)
			SCN_UpdateCB(cbSlot, &entCB);
			entCBGpuVA = cbGpuVA;

			// Draw the entity's MD3 surfaces (no-op if model has no geometry)
			DX12_DrawEntity(ent, entCBGpuVA);
		}
	}

	// ----------------------------------------------------------------
	// 7a. World-space polys (decals, effects)
	//     Rendered after entities, before translucent world surfaces.
	//     Uses the same 3D pipeline with identity model matrix.
	// ----------------------------------------------------------------
	if (dx12Scene.numPolyVerts > 0 && dx12Scene.polyVerts &&
	    dx12Scene.polyVertexBuffer && dx12Scene.polyVBMapped)
	{
		int b;
		D3D12_VERTEX_BUFFER_VIEW pvbv = {};

		// Restore identity model matrix for world-space poly verts
		SCN_UpdateCB(cbSlot, &cb);

		// Upload staging buffer to the GPU upload-heap VB
		memcpy(dx12Scene.polyVBMapped, dx12Scene.polyVerts,
		       (size_t)dx12Scene.numPolyVerts * sizeof(dx12WorldVertex_t));

		pvbv.BufferLocation = dx12Scene.polyVertexBuffer->GetGPUVirtualAddress();
		pvbv.SizeInBytes    = (UINT)dx12Scene.numPolyVerts * (UINT)sizeof(dx12WorldVertex_t);
		pvbv.StrideInBytes  = (UINT)sizeof(dx12WorldVertex_t);

		dx12.commandList->IASetVertexBuffers(0, 1, &pvbv);

		for (b = 0; b < dx12Scene.numPolyBatches; b++)
		{
			const dx12ScenePolyBatch_t *batch = &dx12Scene.polyBatches[b];
			dx12Texture_t              *polyTex;
			D3D12_GPU_DESCRIPTOR_HANDLE srvPoly;

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

			dx12.commandList->SetGraphicsRootDescriptorTable(1, srvPoly);
			dx12.commandList->SetGraphicsRootDescriptorTable(2, srvPoly);

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
	// 7.  Translucent / alpha world surfaces (back-to-front order)
	//     The draw list is already sorted by the loader, so iterate
	//     in reverse for correct alpha blending.
	// ----------------------------------------------------------------
	if (dx12World.loaded && dx12World.vertexBuffer && dx12World.indexBuffer)
	{
		// Restore world CB (entity loop may have modified it)
		SCN_UpdateCB(cbSlot, &cb);

		{
			int i;

			for (i = dx12World.numDrawSurfs - 1; i >= 0; i--)
			{
				const dx12DrawSurf_t *ds = &dx12World.drawSurfs[i];

				if (ds->isTranslucent)
				{
					SCN_DrawSurface(ds, cbGpuVA);
				}
			}
		}
	}

	// ----------------------------------------------------------------
	// 8.  Restore 2D pipeline for subsequent UI rendering
	// ----------------------------------------------------------------
	dx12.commandList->SetGraphicsRootSignature(dx12.rootSignature);
	dx12.commandList->SetPipelineState(dx12.pipelineState);
	dx12.commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
}

#endif // _WIN32
