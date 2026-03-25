/**
 * @file dx12_poly.h
 * @brief DX12 2D drawing helpers – no GL, no renderer_common dependency.
 *
 * All functions write vertices into the per-frame ring-buffer
 * (dx12.quadVBMapped / dx12.quadVBOffset) and issue draw calls against the
 * already-open command list.  They must only be called between
 * DX12_BeginFrameRender() and R_DX12_SwapBuffers().
 */

#ifndef DX12_POLY_H
#define DX12_POLY_H

#ifdef _WIN32

#include "tr_dx12_local.h"

/**
 * @brief Draw a textured quad with per-corner modulate color.
 *
 * @param x,y       Top-left corner in screen pixels (0→vidWidth, 0→vidHeight).
 * @param w,h       Width and height in screen pixels.
 * @param s1,t1     UV for the top-left corner.
 * @param s2,t2     UV for the bottom-right corner.
 * @param hShader   Texture handle from DX12_RegisterTexture().
 */
void DX12_DrawStretchPic(float x, float y, float w, float h,
                         float s1, float t1, float s2, float t2,
                         qhandle_t hShader);

/**
 * @brief Draw a textured quad with a gradient modulate across two corners.
 *
 * @param gradientColor  RGBA [0,1] color applied to the gradient side.
 * @param gradientType   0 = left-right gradient; any other value = top-bottom.
 */
void DX12_DrawStretchPicGradient(float x, float y, float w, float h,
                                 float s1, float t1, float s2, float t2,
                                 qhandle_t hShader, const float *gradientColor,
                                 int gradientType);

/**
 * @brief Draw an array of 2D polygon vertices as a triangle fan.
 *
 * @param polys     Array of numverts polyVert_t vertices (screen-space XY,
 *                  UV, and byte modulate color).
 * @param numverts  Total vertex count (≥ 3).
 * @param hShader   Texture handle.
 */
void DX12_Add2dPolys(polyVert_t *polys, int numverts, qhandle_t hShader);

/**
 * @brief Flush any pending batched draw calls to the GPU.
 *
 * Must be called before R_DX12_SwapBuffers() and whenever a state change
 * (e.g. render target switch) requires the batch to be committed.
 */
void DX12_Flush2D(void);

/**
 * @brief Set the 2D scissor rectangle for subsequent draw calls.
 *
 * All draws issued after this call will be clipped to the given rectangle.
 * If the active batch uses a different scissor rect it is flushed first.
 * Coordinates are in screen pixels; (0,0) is the top-left corner.
 *
 * @param x,y   Top-left corner of the scissor region.
 * @param w,h   Width and height of the scissor region.
 */
void DX12_SetScissor(int x, int y, int w, int h);

/**
 * @brief Draw a UTF-8 string using the supplied font.
 *
 * Each printable ASCII glyph is appended to the batched vertex ring-buffer
 * as a two-triangle (6-vertex) TRIANGLELIST quad.  The current dx12.color2D
 * value is used as the modulate color.
 *
 * @param x      Left edge of the first character, in screen pixels.
 * @param y      Baseline Y position, in screen pixels.
 * @param scale  Uniform scale factor applied to glyph dimensions and advance.
 * @param text   Null-terminated ASCII string to draw.
 * @param font   Pre-registered fontInfo_t (from RE_DX12_RegisterFont).
 */
void DX12_DrawString(float x, float y, float scale,
                     const char *text, const fontInfo_t *font);

#endif // _WIN32
#endif // DX12_POLY_H
