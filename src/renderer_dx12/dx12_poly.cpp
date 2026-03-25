/**
 * @file dx12_poly.cpp
 * @brief DX12 2D drawing – DrawStretchPic, DrawStretchPicGradient, Add2dPolys.
 *
 * All draws append vertices to the per-frame ring-buffer and issue immediate
 * draw calls into the open command list.  No batching is performed; each call
 * produces one DrawInstanced.
 */

#include "dx12_poly.h"
#include "dx12_shader.h"

#ifdef _WIN32

#include <string.h>   // memcpy

// ---------------------------------------------------------------------------
// Coordinate-conversion helpers
// ---------------------------------------------------------------------------

/** Convert screen-pixel X  [0, vidWidth]  →  NDC  [-1, +1] */
static inline float NDC_X(float x)
{
	return (x / (float)dx12.vidWidth) * 2.0f - 1.0f;
}

/** Convert screen-pixel Y  [0, vidHeight]  →  NDC  [+1, -1]  (Y flipped) */
static inline float NDC_Y(float y)
{
	return 1.0f - (y / (float)dx12.vidHeight) * 2.0f;
}

// ---------------------------------------------------------------------------
// Internal: write verts into ring-buffer and issue one draw call
// ---------------------------------------------------------------------------

/**
 * @brief Append vertices to the frame ring-buffer and draw them.
 * @param[in] verts     Array of vertices to write.
 * @param[in] numVerts  Number of vertices (4 for a quad strip, 3*n for fans).
 * @param[in] topology  D3D primitive topology.
 * @param[in] tex       Texture to bind via SRV descriptor table.
 */
static void DX12_AppendAndDraw(const dx12QuadVertex_t *verts, int numVerts,
                                D3D12_PRIMITIVE_TOPOLOGY topology, dx12Texture_t *tex)
{
	D3D12_VERTEX_BUFFER_VIEW vbv;
	dx12QuadVertex_t        *dst;

	if (!dx12.frameOpen || !tex || !dx12.quadVBMapped)
	{
		return;
	}

	if (dx12.quadVBOffset + (UINT)numVerts > DX12_MAX_2D_VERTS)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12: 2D vertex ring-buffer full; dropping draw\n");
		return;
	}

	// Write into mapped upload buffer
	dst = (dx12QuadVertex_t *)dx12.quadVBMapped + dx12.quadVBOffset;
	memcpy(dst, verts, (size_t)numVerts * sizeof(dx12QuadVertex_t));

	// Build a per-draw VBV that points at the slice we just wrote
	vbv.BufferLocation = dx12.quadVertexBuffer->GetGPUVirtualAddress()
	                     + dx12.quadVBOffset * sizeof(dx12QuadVertex_t);
	vbv.StrideInBytes  = (UINT)sizeof(dx12QuadVertex_t);
	vbv.SizeInBytes    = (UINT)(numVerts * (int)sizeof(dx12QuadVertex_t));

	// Bind the texture and draw
	dx12.commandList->SetGraphicsRootDescriptorTable(0, tex->gpuHandle);
	dx12.commandList->IASetPrimitiveTopology(topology);
	dx12.commandList->IASetVertexBuffers(0, 1, &vbv);
	dx12.commandList->DrawInstanced((UINT)numVerts, 1, 0, 0);

	dx12.quadVBOffset += (UINT)numVerts;
}

// ---------------------------------------------------------------------------
// DX12_DrawStretchPic
// ---------------------------------------------------------------------------

/**
 * @brief DX12_DrawStretchPic
 *
 * Builds a TRIANGLESTRIP quad (4 vertices) from the given screen-pixel
 * rectangle and UV range, tinted by the current dx12.color2D value.
 */
void DX12_DrawStretchPic(float x, float y, float w, float h,
                         float s1, float t1, float s2, float t2,
                         qhandle_t hShader)
{
	dx12Texture_t   *tex;
	float            r, g, b, a;
	float            nx1, ny1, nx2, ny2;
	dx12QuadVertex_t verts[4];

	if (!dx12.frameOpen)
	{
		return;
	}

	tex = DX12_GetTexture(hShader);
	if (!tex)
	{
		return;
	}

	r   = dx12.color2D[0];
	g   = dx12.color2D[1];
	b   = dx12.color2D[2];
	a   = dx12.color2D[3];
	nx1 = NDC_X(x);
	ny1 = NDC_Y(y);
	nx2 = NDC_X(x + w);
	ny2 = NDC_Y(y + h);

	// TRIANGLESTRIP order: TL, TR, BL, BR
	verts[0].pos[0]   = nx1; verts[0].pos[1]   = ny1;
	verts[0].uv[0]    = s1;  verts[0].uv[1]    = t1;
	verts[0].color[0] = r;   verts[0].color[1] = g;
	verts[0].color[2] = b;   verts[0].color[3] = a;

	verts[1].pos[0]   = nx2; verts[1].pos[1]   = ny1;
	verts[1].uv[0]    = s2;  verts[1].uv[1]    = t1;
	verts[1].color[0] = r;   verts[1].color[1] = g;
	verts[1].color[2] = b;   verts[1].color[3] = a;

	verts[2].pos[0]   = nx1; verts[2].pos[1]   = ny2;
	verts[2].uv[0]    = s1;  verts[2].uv[1]    = t2;
	verts[2].color[0] = r;   verts[2].color[1] = g;
	verts[2].color[2] = b;   verts[2].color[3] = a;

	verts[3].pos[0]   = nx2; verts[3].pos[1]   = ny2;
	verts[3].uv[0]    = s2;  verts[3].uv[1]    = t2;
	verts[3].color[0] = r;   verts[3].color[1] = g;
	verts[3].color[2] = b;   verts[3].color[3] = a;

	DX12_AppendAndDraw(verts, 4, D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, tex);
}

// ---------------------------------------------------------------------------
// DX12_DrawStretchPicGradient
// ---------------------------------------------------------------------------

/**
 * @brief DX12_DrawStretchPicGradient
 *
 * Like DX12_DrawStretchPic but the two "far" corners receive the gradient
 * color.  gradientType 0 = left→right; anything else = top→bottom.
 */
void DX12_DrawStretchPicGradient(float x, float y, float w, float h,
                                 float s1, float t1, float s2, float t2,
                                 qhandle_t hShader, const float *gradientColor,
                                 int gradientType)
{
	dx12Texture_t   *tex;
	float            r, g, b, a;
	float            gr, gg, gb, ga;
	float            nx1, ny1, nx2, ny2;
	dx12QuadVertex_t verts[4];

	if (!dx12.frameOpen)
	{
		return;
	}

	tex = DX12_GetTexture(hShader);
	if (!tex)
	{
		return;
	}

	r  = dx12.color2D[0];
	g  = dx12.color2D[1];
	b  = dx12.color2D[2];
	a  = dx12.color2D[3];
	gr = gradientColor ? gradientColor[0] : r;
	gg = gradientColor ? gradientColor[1] : g;
	gb = gradientColor ? gradientColor[2] : b;
	ga = gradientColor ? gradientColor[3] : a;

	nx1 = NDC_X(x);
	ny1 = NDC_Y(y);
	nx2 = NDC_X(x + w);
	ny2 = NDC_Y(y + h);

	if (gradientType == 0)
	{
		// Left edge → base color; right edge → gradient color
		verts[0].pos[0]   = nx1; verts[0].pos[1]   = ny1;
		verts[0].uv[0]    = s1;  verts[0].uv[1]    = t1;
		verts[0].color[0] = r;   verts[0].color[1] = g;
		verts[0].color[2] = b;   verts[0].color[3] = a;

		verts[1].pos[0]   = nx2; verts[1].pos[1]   = ny1;
		verts[1].uv[0]    = s2;  verts[1].uv[1]    = t1;
		verts[1].color[0] = gr;  verts[1].color[1] = gg;
		verts[1].color[2] = gb;  verts[1].color[3] = ga;

		verts[2].pos[0]   = nx1; verts[2].pos[1]   = ny2;
		verts[2].uv[0]    = s1;  verts[2].uv[1]    = t2;
		verts[2].color[0] = r;   verts[2].color[1] = g;
		verts[2].color[2] = b;   verts[2].color[3] = a;

		verts[3].pos[0]   = nx2; verts[3].pos[1]   = ny2;
		verts[3].uv[0]    = s2;  verts[3].uv[1]    = t2;
		verts[3].color[0] = gr;  verts[3].color[1] = gg;
		verts[3].color[2] = gb;  verts[3].color[3] = ga;
	}
	else
	{
		// Top edge → base color; bottom edge → gradient color
		verts[0].pos[0]   = nx1; verts[0].pos[1]   = ny1;
		verts[0].uv[0]    = s1;  verts[0].uv[1]    = t1;
		verts[0].color[0] = r;   verts[0].color[1] = g;
		verts[0].color[2] = b;   verts[0].color[3] = a;

		verts[1].pos[0]   = nx2; verts[1].pos[1]   = ny1;
		verts[1].uv[0]    = s2;  verts[1].uv[1]    = t1;
		verts[1].color[0] = r;   verts[1].color[1] = g;
		verts[1].color[2] = b;   verts[1].color[3] = a;

		verts[2].pos[0]   = nx1; verts[2].pos[1]   = ny2;
		verts[2].uv[0]    = s1;  verts[2].uv[1]    = t2;
		verts[2].color[0] = gr;  verts[2].color[1] = gg;
		verts[2].color[2] = gb;  verts[2].color[3] = ga;

		verts[3].pos[0]   = nx2; verts[3].pos[1]   = ny2;
		verts[3].uv[0]    = s2;  verts[3].uv[1]    = t2;
		verts[3].color[0] = gr;  verts[3].color[1] = gg;
		verts[3].color[2] = gb;  verts[3].color[3] = ga;
	}

	DX12_AppendAndDraw(verts, 4, D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, tex);
}

// ---------------------------------------------------------------------------
// DX12_Add2dPolys
// ---------------------------------------------------------------------------

/**
 * @brief DX12_Add2dPolys
 *
 * Converts polyVert_t screen-space vertices to NDC and draws them as a
 * triangle fan (indices: 0,1,2 ; 0,2,3 ; 0,3,4 ; ...) expanded inline into
 * a TRIANGLELIST vertex buffer.
 *
 * @param polys     Array of polyVert_t.  xyz[0]/xyz[1] are screen-pixel X/Y.
 * @param numverts  Total vertex count (must be ≥ 3).
 * @param hShader   Texture handle.
 */
void DX12_Add2dPolys(polyVert_t *polys, int numverts, qhandle_t hShader)
{
	int            numTris;
	int            numTriVerts;
	dx12Texture_t *tex;
	dx12QuadVertex_t *dst;
	int            i;

	if (!dx12.frameOpen || numverts < 3 || !polys)
	{
		return;
	}

	tex = DX12_GetTexture(hShader);
	if (!tex)
	{
		return;
	}

	// Fan triangulation: (numverts - 2) triangles, each 3 verts
	numTris    = numverts - 2;
	numTriVerts = numTris * 3;

	if (dx12.quadVBOffset + (UINT)numTriVerts > DX12_MAX_2D_VERTS)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_Add2dPolys: 2D vertex ring-buffer full\n");
		return;
	}

	dst = (dx12QuadVertex_t *)dx12.quadVBMapped + dx12.quadVBOffset;

	for (i = 0; i < numTris; i++)
	{
		// Triangle: fan vertex 0, fan vertex (i+1), fan vertex (i+2)
		int j;

		for (j = 0; j < 3; j++)
		{
			const polyVert_t *pv;
			int               vi;

			vi = (j == 0) ? 0 : (i + j);
			pv = &polys[vi];

			dst->pos[0]   = NDC_X(pv->xyz[0]);
			dst->pos[1]   = NDC_Y(pv->xyz[1]);
			dst->uv[0]    = pv->st[0];
			dst->uv[1]    = pv->st[1];
			dst->color[0] = pv->modulate[0] / 255.0f;
			dst->color[1] = pv->modulate[1] / 255.0f;
			dst->color[2] = pv->modulate[2] / 255.0f;
			dst->color[3] = pv->modulate[3] / 255.0f;
			dst++;
		}
	}

	{
		D3D12_VERTEX_BUFFER_VIEW vbv;

		vbv.BufferLocation = dx12.quadVertexBuffer->GetGPUVirtualAddress()
		                     + dx12.quadVBOffset * sizeof(dx12QuadVertex_t);
		vbv.StrideInBytes  = (UINT)sizeof(dx12QuadVertex_t);
		vbv.SizeInBytes    = (UINT)(numTriVerts * (int)sizeof(dx12QuadVertex_t));

		dx12.commandList->SetGraphicsRootDescriptorTable(0, tex->gpuHandle);
		dx12.commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		dx12.commandList->IASetVertexBuffers(0, 1, &vbv);
		dx12.commandList->DrawInstanced((UINT)numTriVerts, 1, 0, 0);
	}

	dx12.quadVBOffset += (UINT)numTriVerts;
}

#endif // _WIN32
