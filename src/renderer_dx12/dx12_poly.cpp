/**
 * @file dx12_poly.cpp
 * @brief DX12 2D drawing – DrawStretchPic, DrawStretchPicGradient, Add2dPolys,
 *        Flush2D, SetScissor, and DrawString.
 *
 * All draws append vertices to the per-frame ring-buffer and defer the actual
 * GPU draw call until either the texture/topology/scissor changes or
 * DX12_Flush2D() is called explicitly.  This keeps the number of
 * SetGraphicsRootDescriptorTable + DrawInstanced invocations low.
 *
 * All quads are expanded to 6-vertex TRIANGLELIST so they can be freely
 * concatenated in the same batch regardless of origin (DrawStretchPic vs
 * Add2dPolys vs DrawString).
 */

#include "dx12_poly.h"
#include "dx12_shader.h"

#ifdef _WIN32

#include <string.h>   // memcpy
#include <math.h>     // cosf, sinf, sqrtf

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
// Internal helpers
// ---------------------------------------------------------------------------

/** Compare two D3D12_RECT values for equality. */
static qboolean DX12_RectsEqual(const D3D12_RECT *a, const D3D12_RECT *b)
{
	return (a->left == b->left && a->top == b->top
	        && a->right == b->right && a->bottom == b->bottom)
	       ? qtrue : qfalse;
}

// ---------------------------------------------------------------------------
// DX12_Begin2DBatch
// ---------------------------------------------------------------------------

/**
 * @brief DX12_Begin2DBatch
 *
 * Starts an explicit 2D batch for the given GPU texture descriptor handle and
 * primitive topology.  Any pending batch is flushed first.  Subsequent calls
 * to DX12_AddQuadToBatch() will accumulate vertices into this batch until
 * DX12_Flush2DBatch() is called.
 *
 * @param texHandle  GPU descriptor handle of the texture to bind.
 * @param topology   D3D12 primitive topology (TRIANGLELIST or TRIANGLESTRIP).
 */
void DX12_Begin2DBatch(D3D12_GPU_DESCRIPTOR_HANDLE texHandle, D3D12_PRIMITIVE_TOPOLOGY topology)
{
	if (!dx12.frameOpen || !dx12.quadVBMapped)
	{
		return;
	}

	if (dx12.batch2DCount > 0)
	{
		DX12_Flush2DBatch();
	}

	dx12.batch2DTexHandle = texHandle;
	dx12.batch2DTopology  = topology;
	dx12.batch2DStart     = dx12.quadVBOffset;
	dx12.batch2DCount     = 0;
	dx12.batch2DScissor   = dx12.currentScissor;
}

// ---------------------------------------------------------------------------
// DX12_AddQuadToBatch
// ---------------------------------------------------------------------------

/**
 * @brief DX12_AddQuadToBatch
 *
 * Appends one quad (four corner vertices) to the current batch.  For
 * D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST the corners are expanded to six
 * vertices (two triangles); for D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP they
 * are written as four vertices in strip order.
 *
 * Corner layout:
 *   corners[0] = top-left     corners[1] = top-right
 *   corners[2] = bottom-left  corners[3] = bottom-right
 *
 * No-op when the ring-buffer has no room.  Call DX12_Begin2DBatch() before
 * the first AddQuadToBatch and DX12_Flush2DBatch() when done.
 *
 * @param corners  Array of exactly four dx12QuadVertex_t corner vertices.
 */
void DX12_AddQuadToBatch(const dx12QuadVertex_t corners[4])
{
	dx12QuadVertex_t *dst;
	UINT              numVerts;

	if (!dx12.frameOpen || !dx12.quadVBMapped)
	{
		return;
	}

	numVerts = (dx12.batch2DTopology == D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST) ? 6u : 4u;

	if (dx12.quadVBOffset + numVerts > DX12_MAX_2D_VERTS)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_AddQuadToBatch: 2D vertex ring-buffer full; dropping quad\n");
		return;
	}

	dst = (dx12QuadVertex_t *)dx12.quadVBMapped + dx12.quadVBOffset;

	if (dx12.batch2DTopology == D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST)
	{
		// Triangle 1: TL, TR, BL
		dst[0] = corners[0];
		dst[1] = corners[1];
		dst[2] = corners[2];
		// Triangle 2: TR, BR, BL
		dst[3] = corners[1];
		dst[4] = corners[3];
		dst[5] = corners[2];
	}
	else
	{
		// TRIANGLESTRIP: TL, TR, BL, BR
		dst[0] = corners[0];
		dst[1] = corners[1];
		dst[2] = corners[2];
		dst[3] = corners[3];
	}

	dx12.quadVBOffset += numVerts;
	dx12.batch2DCount += numVerts;
}

// ---------------------------------------------------------------------------
// DX12_Flush2DBatch
// ---------------------------------------------------------------------------

/**
 * @brief DX12_Flush2DBatch
 *
 * Self-contained flush: re-binds the root signature, PSO, and SRV descriptor
 * heap before issuing the draw call.  Use this when callers cannot assume
 * the per-frame pipeline state is already current (e.g. after a render-target
 * switch or when mixing with other draw paths).
 *
 * After returning, batch2DCount is 0 and batch2DStart equals quadVBOffset.
 * No-op when there is nothing pending.
 */
void DX12_Flush2DBatch(void)
{
	D3D12_VERTEX_BUFFER_VIEW vbv;
	ID3D12DescriptorHeap    *heaps[1];

	if (!dx12.frameOpen || dx12.batch2DCount == 0)
	{
		return;
	}

	vbv.BufferLocation = dx12.quadVertexBuffer->GetGPUVirtualAddress()
	                     + dx12.batch2DStart * sizeof(dx12QuadVertex_t);
	vbv.StrideInBytes  = (UINT)sizeof(dx12QuadVertex_t);
	vbv.SizeInBytes    = dx12.batch2DCount * (UINT)sizeof(dx12QuadVertex_t);

	heaps[0] = dx12.srvHeap;

	dx12.commandList->SetGraphicsRootSignature(dx12.rootSignature);
	dx12.commandList->SetPipelineState(dx12.pipelineState);
	dx12.commandList->SetDescriptorHeaps(1, heaps);
	dx12.commandList->RSSetScissorRects(1, &dx12.batch2DScissor);
	dx12.commandList->SetGraphicsRootDescriptorTable(0, dx12.batch2DTexHandle);
	dx12.commandList->IASetPrimitiveTopology(dx12.batch2DTopology);
	dx12.commandList->IASetVertexBuffers(0, 1, &vbv);
	dx12.commandList->DrawInstanced(dx12.batch2DCount, 1, 0, 0);

	dx12.batch2DCount = 0;
	dx12.batch2DStart = dx12.quadVBOffset;
}

// ---------------------------------------------------------------------------
// DX12_Flush2D
// ---------------------------------------------------------------------------

/**
 * @brief DX12_Flush2D
 *
 * Submits the currently accumulated 2D batch as a single DrawInstanced call.
 * After returning, batch2DCount is 0 and batch2DStart equals quadVBOffset.
 * No-op when there is nothing pending.
 *
 * Unlike DX12_Flush2DBatch(), this function assumes the root signature, PSO,
 * and SRV descriptor heap are already bound (i.e. set during DX12_BeginFrame).
 */
void DX12_Flush2D(void)
{
	D3D12_VERTEX_BUFFER_VIEW vbv;

	if (!dx12.frameOpen || dx12.batch2DCount == 0)
	{
		return;
	}

	vbv.BufferLocation = dx12.quadVertexBuffer->GetGPUVirtualAddress()
	                     + dx12.batch2DStart * sizeof(dx12QuadVertex_t);
	vbv.StrideInBytes  = (UINT)sizeof(dx12QuadVertex_t);
	vbv.SizeInBytes    = dx12.batch2DCount * (UINT)sizeof(dx12QuadVertex_t);

	dx12.commandList->RSSetScissorRects(1, &dx12.batch2DScissor);
	dx12.commandList->SetGraphicsRootDescriptorTable(0, dx12.batch2DTexHandle);
	dx12.commandList->IASetPrimitiveTopology(dx12.batch2DTopology);
	dx12.commandList->IASetVertexBuffers(0, 1, &vbv);
	dx12.commandList->DrawInstanced(dx12.batch2DCount, 1, 0, 0);

	dx12.batch2DCount = 0;
	dx12.batch2DStart = dx12.quadVBOffset;
}

// ---------------------------------------------------------------------------
// DX12_AppendToBatch
// ---------------------------------------------------------------------------

/**
 * @brief Append vertices to the current 2D batch, flushing first if needed.
 *
 * Flushes the pending batch whenever the texture, topology, or scissor rect
 * changes.  Also flushes when the ring-buffer does not have enough room.
 *
 * @param verts     Source vertex array.
 * @param numVerts  Number of vertices to append.
 * @param topology  D3D primitive topology (must be TRIANGLELIST for correct batching).
 * @param tex       Target texture; compared by GPU handle pointer.
 */
static void DX12_AppendToBatch(const dx12QuadVertex_t *verts, int numVerts,
                                D3D12_PRIMITIVE_TOPOLOGY topology, dx12Texture_t *tex)
{
	dx12QuadVertex_t *dst;

	if (!dx12.frameOpen || !tex || !dx12.quadVBMapped)
	{
		return;
	}

	// If the ring-buffer is full, flush what is pending and check again
	if (dx12.quadVBOffset + (UINT)numVerts > DX12_MAX_2D_VERTS)
	{
		DX12_Flush2D();
		if (dx12.quadVBOffset + (UINT)numVerts > DX12_MAX_2D_VERTS)
		{
			dx12.ri.Printf(PRINT_WARNING, "DX12: 2D vertex ring-buffer full; dropping draw\n");
			return;
		}
	}

	// Flush the current batch if any draw state has changed
	if (dx12.batch2DCount > 0)
	{
		qboolean flushNeeded = qfalse;

		if (tex->gpuHandle.ptr != dx12.batch2DTexHandle.ptr)
		{
			flushNeeded = qtrue;
		}
		else if (topology != dx12.batch2DTopology)
		{
			flushNeeded = qtrue;
		}
		else if (!DX12_RectsEqual(&dx12.currentScissor, &dx12.batch2DScissor))
		{
			flushNeeded = qtrue;
		}

		if (flushNeeded)
		{
			DX12_Flush2D();
		}
	}

	// Begin a new batch if the current one is empty
	if (dx12.batch2DCount == 0)
	{
		dx12.batch2DStart     = dx12.quadVBOffset;
		dx12.batch2DTexHandle = tex->gpuHandle;
		dx12.batch2DTopology  = topology;
		dx12.batch2DScissor   = dx12.currentScissor;
	}

	// Write vertices into the ring-buffer
	dst = (dx12QuadVertex_t *)dx12.quadVBMapped + dx12.quadVBOffset;
	memcpy(dst, verts, (size_t)numVerts * sizeof(dx12QuadVertex_t));
	dx12.quadVBOffset += (UINT)numVerts;
	dx12.batch2DCount += (UINT)numVerts;
}

// ---------------------------------------------------------------------------
// DX12_SetScissor
// ---------------------------------------------------------------------------

/**
 * @brief DX12_SetScissor
 *
 * Stores a new scissor rectangle for subsequent draw calls.  If the active
 * batch uses a different rectangle it is flushed first so the batch and its
 * matching scissor stay consistent.
 *
 * @param x,y   Top-left corner in screen pixels.
 * @param w,h   Width and height in screen pixels.
 */
void DX12_SetScissor(int x, int y, int w, int h)
{
	D3D12_RECT r;

	r.left   = (LONG)x;
	r.top    = (LONG)y;
	r.right  = (LONG)(x + w);
	r.bottom = (LONG)(y + h);

	// Clamp to the physical swap-chain viewport
	if (r.left   < 0)                            { r.left   = 0; }
	if (r.top    < 0)                            { r.top    = 0; }
	if (r.right  > dx12.scissorRect.right)       { r.right  = dx12.scissorRect.right; }
	if (r.bottom > dx12.scissorRect.bottom)      { r.bottom = dx12.scissorRect.bottom; }

	// If scissor changes, flush the in-flight batch before updating
	if (dx12.batch2DCount > 0 && !DX12_RectsEqual(&r, &dx12.currentScissor))
	{
		DX12_Flush2D();
	}

	dx12.currentScissor = r;
}

// ---------------------------------------------------------------------------
// DX12_DrawStretchPic
// ---------------------------------------------------------------------------

/**
 * @brief DX12_DrawStretchPic
 *
 * Appends two triangles (6 vertices, TRIANGLELIST) covering the given
 * screen-pixel rectangle, tinted by the current dx12.color2D value.
 *
 * Vertex layout (quad corners):
 *   TL = top-left,   TR = top-right
 *   BL = bottom-left, BR = bottom-right
 *
 * Triangle 1: TL, TR, BL
 * Triangle 2: TR, BR, BL
 */
void DX12_DrawStretchPic(float x, float y, float w, float h,
                         float s1, float t1, float s2, float t2,
                         qhandle_t hShader)
{
	dx12Texture_t   *tex;
	float            r, g, b, a;
	float            nx1, ny1, nx2, ny2;
	dx12QuadVertex_t verts[6];

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

	// Triangle 1: TL, TR, BL
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

	// Triangle 2: TR, BR, BL
	verts[3].pos[0]   = nx2; verts[3].pos[1]   = ny1;
	verts[3].uv[0]    = s2;  verts[3].uv[1]    = t1;
	verts[3].color[0] = r;   verts[3].color[1] = g;
	verts[3].color[2] = b;   verts[3].color[3] = a;

	verts[4].pos[0]   = nx2; verts[4].pos[1]   = ny2;
	verts[4].uv[0]    = s2;  verts[4].uv[1]    = t2;
	verts[4].color[0] = r;   verts[4].color[1] = g;
	verts[4].color[2] = b;   verts[4].color[3] = a;

	verts[5].pos[0]   = nx1; verts[5].pos[1]   = ny2;
	verts[5].uv[0]    = s1;  verts[5].uv[1]    = t2;
	verts[5].color[0] = r;   verts[5].color[1] = g;
	verts[5].color[2] = b;   verts[5].color[3] = a;

	DX12_AppendToBatch(verts, 6, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, tex);
}

// ---------------------------------------------------------------------------
// DX12_DrawRotatedPic
// ---------------------------------------------------------------------------

#ifndef M_PI_DX12
#define M_PI_DX12 3.14159265358979323846f
#endif

/**
 * @brief DX12_DrawRotatedPic
 *
 * Draws a textured quad rotated about its own center.
 * Matches the ET:Legacy GL renderer's RE_RotatedPic / RB_RotatedPic behaviour:
 *   - cx = x + w/2, cy = y + h/2 (center in screen pixels)
 *   - r  = sqrt((w/2)^2 + (h/2)^2)  (radius to corners)
 *   - Four corners at angles: A, A+π/2, A+π, A+3π/2  where A = angle * 2π
 *
 * @param x,y     Top-left of the bounding box, in screen pixels.
 * @param w,h     Width and height in screen pixels.
 * @param s1,t1   UV for corner 0 (angle A).
 * @param s2,t2   UV for corner 2 (angle A+π).
 * @param hShader Texture handle.
 * @param angle   Rotation amount in [0, 1] (fraction of a full CCW turn).
 */
void DX12_DrawRotatedPic(float x, float y, float w, float h,
                         float s1, float t1, float s2, float t2,
                         qhandle_t hShader, float angle)
{
	dx12Texture_t   *tex;
	float            r, g, b, a;
	float            cx, cy, rad;
	float            a0, a1, a2, a3;
	float            px[4], py[4];
	dx12QuadVertex_t verts[6];
	float            uvs[4][2];
	float            clr[4];
	int              i;

	if (!dx12.frameOpen)
	{
		return;
	}

	tex = DX12_GetTexture(hShader);
	if (!tex)
	{
		return;
	}

	r = dx12.color2D[0];
	g = dx12.color2D[1];
	b = dx12.color2D[2];
	a = dx12.color2D[3];

	// Compute center and corner radius
	cx  = x + w * 0.5f;
	cy  = y + h * 0.5f;
	rad = sqrtf((w * 0.5f) * (w * 0.5f) + (h * 0.5f) * (h * 0.5f));

	// Base angle in radians (angle is [0, 1] fraction of a full turn)
	a0 = angle * 2.0f * M_PI_DX12;
	a1 = a0 + 0.5f * M_PI_DX12;
	a2 = a0 + 1.0f * M_PI_DX12;
	a3 = a0 + 1.5f * M_PI_DX12;

	// Corner positions in screen pixels
	px[0] = cx + cosf(a0) * rad;  py[0] = cy + sinf(a0) * rad;
	px[1] = cx + cosf(a1) * rad;  py[1] = cy + sinf(a1) * rad;
	px[2] = cx + cosf(a2) * rad;  py[2] = cy + sinf(a2) * rad;
	px[3] = cx + cosf(a3) * rad;  py[3] = cy + sinf(a3) * rad;

	// UV assignment (GL renderer convention)
	uvs[0][0] = s1; uvs[0][1] = t1; // corner 0: A
	uvs[1][0] = s2; uvs[1][1] = t1; // corner 1: A + π/2
	uvs[2][0] = s2; uvs[2][1] = t2; // corner 2: A + π
	uvs[3][0] = s1; uvs[3][1] = t2; // corner 3: A + 3π/2

	clr[0] = r; clr[1] = g; clr[2] = b; clr[3] = a;

	// Expand to 6 vertices (two triangles): (0,1,3) and (1,2,3)
	{
		static const int idx[6] = { 0, 1, 3, 1, 2, 3 };

		for (i = 0; i < 6; i++)
		{
			int ci = idx[i];

			verts[i].pos[0]   = NDC_X(px[ci]);
			verts[i].pos[1]   = NDC_Y(py[ci]);
			verts[i].uv[0]    = uvs[ci][0];
			verts[i].uv[1]    = uvs[ci][1];
			verts[i].color[0] = clr[0];
			verts[i].color[1] = clr[1];
			verts[i].color[2] = clr[2];
			verts[i].color[3] = clr[3];
		}
	}

	DX12_AppendToBatch(verts, 6, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, tex);
}

// ---------------------------------------------------------------------------
// DX12_DrawStretchPicGradient
// ---------------------------------------------------------------------------

/**
 * @brief DX12_DrawStretchPicGradient
 *
 * Like DX12_DrawStretchPic, but the two "far" corners receive a different
 * color (gradientColor) while the "near" corners keep dx12.color2D.
 *
 * gradientType == 0  →  left(near) → right(far)  horizontal gradient
 * any other value    →  top(near)  → bottom(far) vertical gradient
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
	float            tl_r, tl_g, tl_b, tl_a;  // top-left  color
	float            tr_r, tr_g, tr_b, tr_a;  // top-right color
	float            bl_r, bl_g, bl_b, bl_a;  // bottom-left  color
	float            br_r, br_g, br_b, br_a;  // bottom-right color
	dx12QuadVertex_t verts[6];

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
		// Horizontal: left = base, right = gradient
		tl_r = r;  tl_g = g;  tl_b = b;  tl_a = a;
		tr_r = gr; tr_g = gg; tr_b = gb; tr_a = ga;
		bl_r = r;  bl_g = g;  bl_b = b;  bl_a = a;
		br_r = gr; br_g = gg; br_b = gb; br_a = ga;
	}
	else
	{
		// Vertical: top = base, bottom = gradient
		tl_r = r;  tl_g = g;  tl_b = b;  tl_a = a;
		tr_r = r;  tr_g = g;  tr_b = b;  tr_a = a;
		bl_r = gr; bl_g = gg; bl_b = gb; bl_a = ga;
		br_r = gr; br_g = gg; br_b = gb; br_a = ga;
	}

	// Triangle 1: TL, TR, BL
	verts[0].pos[0]   = nx1; verts[0].pos[1]   = ny1;
	verts[0].uv[0]    = s1;  verts[0].uv[1]    = t1;
	verts[0].color[0] = tl_r; verts[0].color[1] = tl_g;
	verts[0].color[2] = tl_b; verts[0].color[3] = tl_a;

	verts[1].pos[0]   = nx2; verts[1].pos[1]   = ny1;
	verts[1].uv[0]    = s2;  verts[1].uv[1]    = t1;
	verts[1].color[0] = tr_r; verts[1].color[1] = tr_g;
	verts[1].color[2] = tr_b; verts[1].color[3] = tr_a;

	verts[2].pos[0]   = nx1; verts[2].pos[1]   = ny2;
	verts[2].uv[0]    = s1;  verts[2].uv[1]    = t2;
	verts[2].color[0] = bl_r; verts[2].color[1] = bl_g;
	verts[2].color[2] = bl_b; verts[2].color[3] = bl_a;

	// Triangle 2: TR, BR, BL
	verts[3].pos[0]   = nx2; verts[3].pos[1]   = ny1;
	verts[3].uv[0]    = s2;  verts[3].uv[1]    = t1;
	verts[3].color[0] = tr_r; verts[3].color[1] = tr_g;
	verts[3].color[2] = tr_b; verts[3].color[3] = tr_a;

	verts[4].pos[0]   = nx2; verts[4].pos[1]   = ny2;
	verts[4].uv[0]    = s2;  verts[4].uv[1]    = t2;
	verts[4].color[0] = br_r; verts[4].color[1] = br_g;
	verts[4].color[2] = br_b; verts[4].color[3] = br_a;

	verts[5].pos[0]   = nx1; verts[5].pos[1]   = ny2;
	verts[5].uv[0]    = s1;  verts[5].uv[1]    = t2;
	verts[5].color[0] = bl_r; verts[5].color[1] = bl_g;
	verts[5].color[2] = bl_b; verts[5].color[3] = bl_a;

	DX12_AppendToBatch(verts, 6, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, tex);
}

// ---------------------------------------------------------------------------
// DX12_Add2dPolys
// ---------------------------------------------------------------------------

/**
 * @brief DX12_Add2dPolys
 *
 * Fan-expands a polyVert_t array into TRIANGLELIST vertices and appends them
 * to the 2D batch via DX12_AppendToBatch.  Per-vertex colours come from the
 * polyVert_t modulate field.
 *
 * @param polys     Array of polyVert_t (xyz[0]/xyz[1] = screen-pixel X/Y).
 * @param numverts  Total vertex count (≥ 3).
 * @param hShader   Texture handle.
 */
void DX12_Add2dPolys(polyVert_t *polys, int numverts, qhandle_t hShader)
{
	int               numTris;
	int               numTriVerts;
	dx12Texture_t    *tex;
	dx12QuadVertex_t *expanded;
	int               i;

	if (!dx12.frameOpen || !dx12.quadVBMapped || numverts < 3 || !polys)
	{
		return;
	}

	tex = DX12_GetTexture(hShader);
	if (!tex)
	{
		return;
	}

	// Fan triangulation: (numverts - 2) triangles, each 3 verts
	numTris     = numverts - 2;
	numTriVerts = numTris * 3;

	if ((UINT)numTriVerts > DX12_MAX_2D_VERTS)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_Add2dPolys: polygon too large (%d verts), dropping\n", numverts);
		return;
	}

	expanded = new dx12QuadVertex_t[numTriVerts];

	for (i = 0; i < numTris; i++)
	{
		int j;

		for (j = 0; j < 3; j++)
		{
			const polyVert_t *pv;
			int               vi  = (j == 0) ? 0 : (i + j);
			int               out = i * 3 + j;

			pv = &polys[vi];

			expanded[out].pos[0]   = NDC_X(pv->xyz[0]);
			expanded[out].pos[1]   = NDC_Y(pv->xyz[1]);
			expanded[out].uv[0]    = pv->st[0];
			expanded[out].uv[1]    = pv->st[1];
			expanded[out].color[0] = pv->modulate[0] / 255.0f;
			expanded[out].color[1] = pv->modulate[1] / 255.0f;
			expanded[out].color[2] = pv->modulate[2] / 255.0f;
			expanded[out].color[3] = pv->modulate[3] / 255.0f;
		}
	}

	DX12_AppendToBatch(expanded, numTriVerts, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, tex);

	delete[] expanded;
}

// ---------------------------------------------------------------------------
// DX12_DrawString
// ---------------------------------------------------------------------------

/**
 * @brief DX12_DrawString
 *
 * Iterates through each ASCII character of @p text, looks up the glyph in
 * @p font, and appends a two-triangle TRIANGLELIST quad to the batch.
 *
 * The glyph's @c glyph handle must have been registered via
 * DX12_RegisterTexture (which RE_DX12_RegisterFont does automatically).
 * Skips characters whose glyph handle is zero or whose image dimensions
 * are degenerate.
 */
void DX12_DrawString(float x, float y, float scale,
                     const char *text, const fontInfo_t *font)
{
	float       curX;
	const char *p;

	if (!dx12.frameOpen || !text || !font)
	{
		return;
	}

	curX = x;

	for (p = text; *p; p++)
	{
		const glyphInfo_t *glyph;
		dx12Texture_t     *tex;
		dx12QuadVertex_t   verts[6];
		float              gx, gy, gw, gh;
		float              nx1, ny1, nx2, ny2;
		float              r, g, b, a;
		int                ch;

		ch = (unsigned char)*p;

		if (ch < GLYPH_START || ch > GLYPH_ASCII_END)
		{
			continue;
		}

		glyph = &font->glyphs[ch];

		if (!glyph->imageWidth || !glyph->imageHeight || !glyph->glyph)
		{
			curX += (float)glyph->xSkip * scale;
			continue;
		}

		tex = DX12_GetTexture(glyph->glyph);
		if (!tex)
		{
			curX += (float)glyph->xSkip * scale;
			continue;
		}

		// Glyph quad: x advances by pitch (bearing); y is offset by top
		gx = curX + (float)glyph->pitch * scale;
		gy = y - (float)glyph->top * scale;
		gw = (float)glyph->imageWidth  * scale;
		gh = (float)glyph->imageHeight * scale;

		nx1 = NDC_X(gx);
		ny1 = NDC_Y(gy);
		nx2 = NDC_X(gx + gw);
		ny2 = NDC_Y(gy + gh);

		r = dx12.color2D[0];
		g = dx12.color2D[1];
		b = dx12.color2D[2];
		a = dx12.color2D[3];

		// Triangle 1: TL, TR, BL
		verts[0].pos[0]   = nx1;        verts[0].pos[1]   = ny1;
		verts[0].uv[0]    = glyph->s;   verts[0].uv[1]    = glyph->t;
		verts[0].color[0] = r;          verts[0].color[1] = g;
		verts[0].color[2] = b;          verts[0].color[3] = a;

		verts[1].pos[0]   = nx2;        verts[1].pos[1]   = ny1;
		verts[1].uv[0]    = glyph->s2;  verts[1].uv[1]    = glyph->t;
		verts[1].color[0] = r;          verts[1].color[1] = g;
		verts[1].color[2] = b;          verts[1].color[3] = a;

		verts[2].pos[0]   = nx1;        verts[2].pos[1]   = ny2;
		verts[2].uv[0]    = glyph->s;   verts[2].uv[1]    = glyph->t2;
		verts[2].color[0] = r;          verts[2].color[1] = g;
		verts[2].color[2] = b;          verts[2].color[3] = a;

		// Triangle 2: TR, BR, BL
		verts[3].pos[0]   = nx2;        verts[3].pos[1]   = ny1;
		verts[3].uv[0]    = glyph->s2;  verts[3].uv[1]    = glyph->t;
		verts[3].color[0] = r;          verts[3].color[1] = g;
		verts[3].color[2] = b;          verts[3].color[3] = a;

		verts[4].pos[0]   = nx2;        verts[4].pos[1]   = ny2;
		verts[4].uv[0]    = glyph->s2;  verts[4].uv[1]    = glyph->t2;
		verts[4].color[0] = r;          verts[4].color[1] = g;
		verts[4].color[2] = b;          verts[4].color[3] = a;

		verts[5].pos[0]   = nx1;        verts[5].pos[1]   = ny2;
		verts[5].uv[0]    = glyph->s;   verts[5].uv[1]    = glyph->t2;
		verts[5].color[0] = r;          verts[5].color[1] = g;
		verts[5].color[2] = b;          verts[5].color[3] = a;

		DX12_AppendToBatch(verts, 6, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, tex);

		curX += (float)glyph->xSkip * scale;
	}
}

#endif // _WIN32
