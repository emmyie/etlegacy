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
 * @file dx12_skeletal.cpp
 * @brief DX12 skeletal model subsystem – MDS / MDM / MDX loading and tag lookup.
 *
 * This is a self-contained port of the GL renderer's tr_animation_mds.c and
 * tr_animation_mdm.c, adapted to work without any GL-specific globals
 * (backEnd, viewParms, tess, etc.).  Only the bone-calculation and tag-lookup
 * paths are implemented; GPU surface rendering is not yet supported.
 *
 * Reference files (GL renderer):
 *   src/renderer/tr_animation_mds.c  – MDS bone calculator + tag lookup
 *   src/renderer/tr_animation_mdm.c  – MDM bone calculator + tag lookup
 *   src/renderer/tr_model.c          – R_LoadMDS / R_LoadMDM / R_LoadMDX
 */

#ifdef _WIN32

#include "dx12_skeletal.h"
#include "tr_dx12_local.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

extern "C" {
#include "../qcommon/q_shared.h"
#include "../qcommon/q_math.h"
#include "../qcommon/qfiles.h"
}

// ---------------------------------------------------------------------------
// Shared inline math helpers
// (Ported verbatim from tr_animation_mds.c; identical in tr_animation_mdm.c)
// ---------------------------------------------------------------------------

static ID_INLINE void SKL_LocalMatrixTransformVector(vec3_t in, vec3_t mat[3], vec3_t out)
{
	out[0] = in[0] * mat[0][0] + in[1] * mat[0][1] + in[2] * mat[0][2];
	out[1] = in[0] * mat[1][0] + in[1] * mat[1][1] + in[2] * mat[1][2];
	out[2] = in[0] * mat[2][0] + in[1] * mat[2][1] + in[2] * mat[2][2];
}

static ID_INLINE void SKL_LocalAddScaledMatrixTransformVectorTranslate(vec3_t in, float s, vec3_t mat[3], vec3_t tr, vec3_t out)
{
	out[0] += s * (in[0] * mat[0][0] + in[1] * mat[0][1] + in[2] * mat[0][2] + tr[0]);
	out[1] += s * (in[0] * mat[1][0] + in[1] * mat[1][1] + in[2] * mat[1][2] + tr[1]);
	out[2] += s * (in[0] * mat[2][0] + in[1] * mat[2][1] + in[2] * mat[2][2] + tr[2]);
}

static ID_INLINE void SKL_LocalScaledMatrixTransformVector(vec3_t in, float s, vec3_t mat[3], vec3_t out)
{
	out[0] = (1.0f - s) * in[0] + s * (in[0] * mat[0][0] + in[1] * mat[0][1] + in[2] * mat[0][2]);
	out[1] = (1.0f - s) * in[1] + s * (in[0] * mat[1][0] + in[1] * mat[1][1] + in[2] * mat[1][2]);
	out[2] = (1.0f - s) * in[2] + s * (in[0] * mat[2][0] + in[1] * mat[2][1] + in[2] * mat[2][2]);
}

static ID_INLINE void SKL_LocalVectorMA(vec3_t org, float dist, vec3_t vec, vec3_t out)
{
	out[0] = org[0] + dist * vec[0];
	out[1] = org[1] + dist * vec[1];
	out[2] = org[2] + dist * vec[2];
}

static ID_INLINE void SKL_SLerp_Normal(vec3_t from, vec3_t to, float tt, vec3_t out)
{
	float ft = 1.0f - tt;

	out[0] = from[0] * ft + to[0] * tt;
	out[1] = from[1] * ft + to[1] * tt;
	out[2] = from[2] * ft + to[2] * tt;

	VectorNormalize(out);
}

/** Decompress a pair of 16-bit pitch/yaw values into a unit direction vector. */
static ID_INLINE void SKL_LocalAngleVector(float pitch_deg, float yaw_deg, vec3_t out)
{
	static const float DEG_TO_RAD = M_TAU_F / 360.0f;
	float sp = (float)sin(pitch_deg * DEG_TO_RAD);
	float cp = (float)cos(pitch_deg * DEG_TO_RAD);
	float sy = (float)sin(yaw_deg  * DEG_TO_RAD);
	float cy = (float)cos(yaw_deg  * DEG_TO_RAD);

	out[0] = cp * cy;
	out[1] = cp * sy;
	out[2] = -sp;
}

static ID_INLINE void SKL_Matrix4FromAxisPlusTranslation(vec3_t axis[3], const vec3_t t, vec4_t dst[4])
{
	int i, j;

	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			dst[i][j] = axis[i][j];
		}
		dst[3][i] = 0;
		dst[i][3] = t[i];
	}
	dst[3][3] = 1;
}

static ID_INLINE void SKL_Matrix4FromScaledAxisPlusTranslation(vec3_t axis[3], const float scale, const vec3_t t, vec4_t dst[4])
{
	int i, j;

	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			dst[i][j] = scale * axis[i][j];
			if (i == j)
			{
				dst[i][j] += 1.0f - scale;
			}
		}
		dst[3][i] = 0;
		dst[i][3] = t[i];
	}
	dst[3][3] = 1;
}

static ID_INLINE void SKL_Matrix4MultiplyInto3x3AndTranslation(vec4_t a[4], vec4_t b[4], vec3_t dst[3], vec3_t t)
{
	dst[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0] + a[0][3] * b[3][0];
	dst[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1] + a[0][3] * b[3][1];
	dst[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2] + a[0][3] * b[3][2];
	t[0]      = a[0][0] * b[0][3] + a[0][1] * b[1][3] + a[0][2] * b[2][3] + a[0][3] * b[3][3];

	dst[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0] + a[1][3] * b[3][0];
	dst[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1] + a[1][3] * b[3][1];
	dst[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2] + a[1][3] * b[3][2];
	t[1]      = a[1][0] * b[0][3] + a[1][1] * b[1][3] + a[1][2] * b[2][3] + a[1][3] * b[3][3];

	dst[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0] + a[2][3] * b[3][0];
	dst[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1] + a[2][3] * b[3][1];
	dst[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2] + a[2][3] * b[3][2];
	t[2]      = a[2][0] * b[0][3] + a[2][1] * b[1][3] + a[2][2] * b[2][3] + a[2][3] * b[3][3];
}

static ID_INLINE void SKL_Matrix3Transpose(const vec3_t matrix[3], vec3_t transpose[3])
{
	int i, j;

	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			transpose[i][j] = matrix[j][i];
		}
	}
}

#define SKL_ANGLES_SHORT_TO_FLOAT(pf, sh) \
	do { *(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = SHORT2ANGLE(*(sh++)); } while (0)

// ===========================================================================
// MDS Bone Calculator
// (Port of tr_animation_mds.c R_CalcBone / R_CalcBoneLerp / R_CalcBones)
// ===========================================================================

// Static context for the MDS bone calculator.
// Mirrors the file-scope statics in tr_animation_mds.c.
// Only valid during a single DX12_GetBoneTagMDS call.
static mdsBoneFrame_t           mds_bones[MDS_MAX_BONES];
static mdsBoneFrame_t           mds_rawBones[MDS_MAX_BONES];
static mdsBoneFrame_t           mds_oldBones[MDS_MAX_BONES];
static char                     mds_validBones[MDS_MAX_BONES];
static char                     mds_newBones[MDS_MAX_BONES];
static refEntity_t              mds_lastEnt;

static float                    mds_frontlerp, mds_backlerp;
static float                    mds_torsoFrontlerp, mds_torsoBacklerp;

static mdsBoneFrame_t           *mds_bonePtr, *mds_parentBone;
static mdsBoneFrameCompressed_t *mds_cBonePtr, *mds_cTBonePtr;
static mdsBoneFrameCompressed_t *mds_cOldBonePtr, *mds_cOldTBonePtr;
static mdsBoneFrameCompressed_t *mds_cBoneList, *mds_cOldBoneList;
static mdsBoneFrameCompressed_t *mds_cBoneListTorso, *mds_cOldBoneListTorso;
static mdsBoneInfo_t            *mds_boneInfo, *mds_thisBoneInfo;
static mdsFrame_t               *mds_frame, *mds_torsoFrame;
static mdsFrame_t               *mds_oldFrame, *mds_oldTorsoFrame;
static int                      mds_frameSize;

static vec3_t   mds_angles, mds_tangles;
static vec3_t   mds_torsoParentOffset, mds_torsoAxis[3], mds_tmpAxis[3];
static vec3_t   mds_vec, mds_v2;
static float    mds_diff;
static qboolean mds_isTorso, mds_fullTorso;

/**
 * Compute the world-space transform of a single MDS bone (no interpolation).
 * Mirrors GL's R_CalcBone (tr_animation_mds.c).
 */
static void MDS_CalcBone(mdsHeader_t *header, int boneNum)
{
	short *sh;
	float *pf;
	int    j;

	mds_thisBoneInfo = &mds_boneInfo[boneNum];

	if (mds_thisBoneInfo->torsoWeight != 0.f)
	{
		mds_cTBonePtr = &mds_cBoneListTorso[boneNum];
		mds_isTorso   = qtrue;
		mds_fullTorso = (mds_thisBoneInfo->torsoWeight == 1.0f) ? qtrue : qfalse;
	}
	else
	{
		mds_isTorso   = qfalse;
		mds_fullTorso = qfalse;
	}
	mds_cBonePtr = &mds_cBoneList[boneNum];
	mds_bonePtr  = &mds_bones[boneNum];

	if (mds_thisBoneInfo->parent >= 0)
	{
		mds_parentBone = &mds_bones[mds_thisBoneInfo->parent];
	}
	else
	{
		mds_parentBone = NULL;
	}

	// rotation
	if (mds_fullTorso)
	{
		sh = (short *)mds_cTBonePtr->angles;
		pf = mds_angles;
		SKL_ANGLES_SHORT_TO_FLOAT(pf, sh);
	}
	else
	{
		sh = (short *)mds_cBonePtr->angles;
		pf = mds_angles;
		SKL_ANGLES_SHORT_TO_FLOAT(pf, sh);
		if (mds_isTorso)
		{
			sh = (short *)mds_cTBonePtr->angles;
			pf = mds_tangles;
			SKL_ANGLES_SHORT_TO_FLOAT(pf, sh);
			for (j = 0; j < 3; j++)
			{
				mds_diff = mds_tangles[j] - mds_angles[j];
				if (Q_fabs(mds_diff) > 180)
				{
					mds_diff = AngleNormalize180(mds_diff);
				}
				mds_angles[j] = mds_angles[j] + mds_thisBoneInfo->torsoWeight * mds_diff;
			}
		}
	}
	AnglesToAxis(mds_angles, mds_bonePtr->matrix);

	// translation
	if (mds_parentBone)
	{
		if (mds_fullTorso)
		{
			sh          = (short *)mds_cTBonePtr->ofsAngles; pf = mds_angles;
			*(pf++)     = SHORT2ANGLE(*(sh++)); *(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = 0;
			SKL_LocalAngleVector(mds_angles[PITCH], mds_angles[YAW], mds_vec);
			SKL_LocalVectorMA(mds_parentBone->translation, mds_thisBoneInfo->parentDist, mds_vec, mds_bonePtr->translation);
		}
		else
		{
			sh          = (short *)mds_cBonePtr->ofsAngles; pf = mds_angles;
			*(pf++)     = SHORT2ANGLE(*(sh++)); *(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = 0;
			SKL_LocalAngleVector(mds_angles[PITCH], mds_angles[YAW], mds_vec);

			if (mds_isTorso)
			{
				sh          = (short *)mds_cTBonePtr->ofsAngles; pf = mds_tangles;
				*(pf++)     = SHORT2ANGLE(*(sh++)); *(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = 0;
				SKL_LocalAngleVector(mds_tangles[PITCH], mds_tangles[YAW], mds_v2);
				SKL_SLerp_Normal(mds_vec, mds_v2, mds_thisBoneInfo->torsoWeight, mds_vec);
				SKL_LocalVectorMA(mds_parentBone->translation, mds_thisBoneInfo->parentDist, mds_vec, mds_bonePtr->translation);
			}
			else
			{
				SKL_LocalVectorMA(mds_parentBone->translation, mds_thisBoneInfo->parentDist, mds_vec, mds_bonePtr->translation);
			}
		}
	}
	else
	{
		// root bone: use the frame's parent offset
		VectorCopy(mds_frame->parentOffset, mds_bonePtr->translation);
	}

	if (boneNum == header->torsoParent)
	{
		VectorCopy(mds_bonePtr->translation, mds_torsoParentOffset);
	}

	mds_validBones[boneNum] = 1;
	mds_rawBones[boneNum]   = *mds_bonePtr;
	mds_newBones[boneNum]   = 1;
}

/**
 * Compute the world-space transform of a single MDS bone (with interpolation).
 * Mirrors GL's R_CalcBoneLerp (tr_animation_mds.c).
 */
static void MDS_CalcBoneLerp(mdsHeader_t *header, int boneNum)
{
	short *sh, *sh2;
	float *pf;
	float  a1, a2;
	int    j;

	if (boneNum < 0 || boneNum >= MDS_MAX_BONES)
	{
		return;
	}

	mds_thisBoneInfo = &mds_boneInfo[boneNum];

	if (mds_thisBoneInfo->parent >= 0)
	{
		mds_parentBone = &mds_bones[mds_thisBoneInfo->parent];
	}
	else
	{
		mds_parentBone = NULL;
	}

	if (mds_thisBoneInfo->torsoWeight != 0.f)
	{
		mds_cTBonePtr    = &mds_cBoneListTorso[boneNum];
		mds_cOldTBonePtr = &mds_cOldBoneListTorso[boneNum];
		mds_isTorso      = qtrue;
		mds_fullTorso    = (mds_thisBoneInfo->torsoWeight == 1.0f) ? qtrue : qfalse;
	}
	else
	{
		mds_isTorso   = qfalse;
		mds_fullTorso = qfalse;
	}
	mds_cBonePtr    = &mds_cBoneList[boneNum];
	mds_cOldBonePtr = &mds_cOldBoneList[boneNum];
	mds_bonePtr     = &mds_bones[boneNum];
	mds_newBones[boneNum] = 1;

	// rotation (shortest-path lerp)
	if (mds_fullTorso)
	{
		sh  = (short *)mds_cTBonePtr->angles;
		sh2 = (short *)mds_cOldTBonePtr->angles;
		pf  = mds_angles;

		a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
		mds_diff = AngleNormalize180(a1 - a2);
		*(pf++) = a1 - mds_torsoBacklerp * mds_diff;
		a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
		mds_diff = AngleNormalize180(a1 - a2);
		*(pf++) = a1 - mds_torsoBacklerp * mds_diff;
		a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
		mds_diff = AngleNormalize180(a1 - a2);
		*(pf++) = a1 - mds_torsoBacklerp * mds_diff;
	}
	else
	{
		sh  = (short *)mds_cBonePtr->angles;
		sh2 = (short *)mds_cOldBonePtr->angles;
		pf  = mds_angles;

		a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
		mds_diff = AngleNormalize180(a1 - a2);
		*(pf++) = a1 - mds_backlerp * mds_diff;
		a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
		mds_diff = AngleNormalize180(a1 - a2);
		*(pf++) = a1 - mds_backlerp * mds_diff;
		a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
		mds_diff = AngleNormalize180(a1 - a2);
		*(pf++) = a1 - mds_backlerp * mds_diff;

		if (mds_isTorso)
		{
			sh  = (short *)mds_cTBonePtr->angles;
			sh2 = (short *)mds_cOldTBonePtr->angles;
			pf  = mds_tangles;

			a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
			mds_diff = AngleNormalize180(a1 - a2);
			*(pf++) = a1 - mds_torsoBacklerp * mds_diff;
			a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
			mds_diff = AngleNormalize180(a1 - a2);
			*(pf++) = a1 - mds_torsoBacklerp * mds_diff;
			a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
			mds_diff = AngleNormalize180(a1 - a2);
			*(pf++) = a1 - mds_torsoBacklerp * mds_diff;

			// blend the angles together
			for (j = 0; j < 3; j++)
			{
				mds_diff = mds_tangles[j] - mds_angles[j];
				if (Q_fabs(mds_diff) > 180)
				{
					mds_diff = AngleNormalize180(mds_diff);
				}
				mds_angles[j] = mds_angles[j] + mds_thisBoneInfo->torsoWeight * mds_diff;
			}
		}
	}
	AnglesToAxis(mds_angles, mds_bonePtr->matrix);

	// translation
	if (mds_parentBone)
	{
		if (mds_fullTorso)
		{
			sh  = (short *)mds_cTBonePtr->ofsAngles;
			sh2 = (short *)mds_cOldTBonePtr->ofsAngles;
		}
		else
		{
			sh  = (short *)mds_cBonePtr->ofsAngles;
			sh2 = (short *)mds_cOldBonePtr->ofsAngles;
		}

		pf      = mds_angles;
		*(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = 0;
		SKL_LocalAngleVector(mds_angles[PITCH], mds_angles[YAW], mds_v2);   // new

		pf      = mds_angles;
		*(pf++) = SHORT2ANGLE(*(sh2++)); *(pf++) = SHORT2ANGLE(*(sh2++)); *(pf++) = 0;
		SKL_LocalAngleVector(mds_angles[PITCH], mds_angles[YAW], mds_vec);  // old

		if (mds_fullTorso)
		{
			SKL_SLerp_Normal(mds_vec, mds_v2, mds_torsoFrontlerp, mds_vec);
		}
		else
		{
			SKL_SLerp_Normal(mds_vec, mds_v2, mds_frontlerp, mds_vec);
		}

		// partial torso blending
		if (!mds_fullTorso && mds_isTorso)
		{
			sh  = (short *)mds_cTBonePtr->ofsAngles;
			sh2 = (short *)mds_cOldTBonePtr->ofsAngles;

			pf      = mds_angles;
			*(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = 0;
			SKL_LocalAngleVector(mds_angles[PITCH], mds_angles[YAW], mds_v2);

			pf      = mds_angles;
			*(pf++) = SHORT2ANGLE(*(sh2++)); *(pf++) = SHORT2ANGLE(*(sh2++)); *(pf++) = 0;
			SKL_LocalAngleVector(mds_angles[PITCH], mds_angles[YAW], mds_angles);  // reuse for old

			SKL_SLerp_Normal(mds_angles, mds_v2, mds_torsoFrontlerp, mds_v2);
			SKL_SLerp_Normal(mds_vec, mds_v2, mds_thisBoneInfo->torsoWeight, mds_vec);
		}

		SKL_LocalVectorMA(mds_parentBone->translation, mds_thisBoneInfo->parentDist, mds_vec, mds_bonePtr->translation);
	}
	else
	{
		// root bone: interpolate parent offset
		mds_bonePtr->translation[0] = mds_frontlerp * mds_frame->parentOffset[0] + mds_backlerp * mds_oldFrame->parentOffset[0];
		mds_bonePtr->translation[1] = mds_frontlerp * mds_frame->parentOffset[1] + mds_backlerp * mds_oldFrame->parentOffset[1];
		mds_bonePtr->translation[2] = mds_frontlerp * mds_frame->parentOffset[2] + mds_backlerp * mds_oldFrame->parentOffset[2];
	}

	if (boneNum == header->torsoParent)
	{
		VectorCopy(mds_bonePtr->translation, mds_torsoParentOffset);
	}

	mds_validBones[boneNum] = 1;
	mds_rawBones[boneNum]   = *mds_bonePtr;
	mds_newBones[boneNum]   = 1;
}

/**
 * Build the full bone list for the given entity and compute world-space bone
 * transforms.  Mirrors GL's R_CalcBones (tr_animation_mds.c).
 *
 * @param header    Pointer to the raw MDS data.
 * @param refent    Entity whose frame / oldframe / backlerp / torso* to use.
 * @param boneList  Bone indices to compute (in parent-first order).
 * @param numBones  Number of entries in boneList.
 */
static void MDS_CalcBones(mdsHeader_t *header, const refEntity_t *refent,
                          int *boneList, int numBones)
{
	int        i, *boneRefs;
	float      torsoWeight;
	vec4_t     m1[4], m2[4];
	vec3_t     t;

	// Invalidate cache if entity has changed
	if (memcmp(&mds_lastEnt, refent, sizeof(refEntity_t)) != 0)
	{
		memset(mds_validBones, 0, header->numBones);
		mds_lastEnt = *refent;
	}

	memset(mds_newBones, 0, header->numBones);

	if (refent->oldframe == refent->frame)
	{
		mds_backlerp  = 0;
		mds_frontlerp = 1;
	}
	else
	{
		mds_backlerp  = refent->backlerp;
		mds_frontlerp = 1.0f - refent->backlerp;
	}

	if (refent->oldTorsoFrame == refent->torsoFrame)
	{
		mds_torsoBacklerp  = 0;
		mds_torsoFrontlerp = 1;
	}
	else
	{
		mds_torsoBacklerp  = refent->torsoBacklerp;
		mds_torsoFrontlerp = 1.0f - refent->torsoBacklerp;
	}

	mds_frameSize = (int)(sizeof(mdsFrame_t) + (header->numBones - 1) * sizeof(mdsBoneFrameCompressed_t));

	mds_frame = (mdsFrame_t *)((byte *)header + header->ofsFrames + refent->frame * mds_frameSize);
	mds_torsoFrame = (mdsFrame_t *)((byte *)header + header->ofsFrames + refent->torsoFrame * mds_frameSize);
	mds_oldFrame = (mdsFrame_t *)((byte *)header + header->ofsFrames + refent->oldframe * mds_frameSize);
	mds_oldTorsoFrame = (mdsFrame_t *)((byte *)header + header->ofsFrames + refent->oldTorsoFrame * mds_frameSize);

	mds_cBoneList      = mds_frame->bones;
	mds_cBoneListTorso = mds_torsoFrame->bones;

	mds_boneInfo = (mdsBoneInfo_t *)((byte *)header + header->ofsBones);
	SKL_Matrix3Transpose(refent->torsoAxis, mds_torsoAxis);

	boneRefs = boneList;

	if (mds_backlerp == 0.f && mds_torsoBacklerp == 0.f)
	{
		for (i = 0; i < numBones; i++, boneRefs++)
		{
			if (mds_validBones[*boneRefs])
			{
				mds_bones[*boneRefs] = mds_rawBones[*boneRefs];
				continue;
			}
			// ensure parent is computed first
			if (mds_boneInfo[*boneRefs].parent >= 0 &&
			    !mds_validBones[mds_boneInfo[*boneRefs].parent] &&
			    !mds_newBones[mds_boneInfo[*boneRefs].parent])
			{
				MDS_CalcBone(header, mds_boneInfo[*boneRefs].parent);
			}
			MDS_CalcBone(header, *boneRefs);
		}
	}
	else
	{
		mds_cOldBoneList      = mds_oldFrame->bones;
		mds_cOldBoneListTorso = mds_oldTorsoFrame->bones;

		for (i = 0; i < numBones; i++, boneRefs++)
		{
			if (mds_validBones[*boneRefs])
			{
				mds_bones[*boneRefs] = mds_rawBones[*boneRefs];
				continue;
			}
			if (mds_boneInfo[*boneRefs].parent >= 0 &&
			    !mds_validBones[mds_boneInfo[*boneRefs].parent] &&
			    !mds_newBones[mds_boneInfo[*boneRefs].parent])
			{
				MDS_CalcBoneLerp(header, mds_boneInfo[*boneRefs].parent);
			}
			MDS_CalcBoneLerp(header, *boneRefs);
		}
	}

	// adjust for torso rotations
	torsoWeight = 0;
	boneRefs    = boneList;
	for (i = 0; i < numBones; i++, boneRefs++)
	{
		mds_thisBoneInfo = &mds_boneInfo[*boneRefs];
		mds_bonePtr      = &mds_bones[*boneRefs];

		if (mds_thisBoneInfo->torsoWeight > 0)
		{
			if (!mds_newBones[*boneRefs])
			{
				mds_bones[*boneRefs] = mds_oldBones[*boneRefs];
				continue;
			}

			if (!(mds_thisBoneInfo->flags & BONEFLAG_TAG))
			{
				VectorSubtract(mds_bonePtr->translation, mds_torsoParentOffset, t);
				SKL_Matrix4FromAxisPlusTranslation(mds_bonePtr->matrix, t, m1);

				if (torsoWeight != mds_thisBoneInfo->torsoWeight)
				{
					SKL_Matrix4FromScaledAxisPlusTranslation(mds_torsoAxis, mds_thisBoneInfo->torsoWeight, mds_torsoParentOffset, m2);
					torsoWeight = mds_thisBoneInfo->torsoWeight;
				}
				SKL_Matrix4MultiplyInto3x3AndTranslation(m2, m1, mds_bonePtr->matrix, mds_bonePtr->translation);
			}
			else
			{
				// tag bones require special handling
				SKL_LocalScaledMatrixTransformVector(mds_bonePtr->matrix[0], mds_thisBoneInfo->torsoWeight, mds_torsoAxis, mds_tmpAxis[0]);
				SKL_LocalScaledMatrixTransformVector(mds_bonePtr->matrix[1], mds_thisBoneInfo->torsoWeight, mds_torsoAxis, mds_tmpAxis[1]);
				SKL_LocalScaledMatrixTransformVector(mds_bonePtr->matrix[2], mds_thisBoneInfo->torsoWeight, mds_torsoAxis, mds_tmpAxis[2]);
				memcpy(mds_bonePtr->matrix, mds_tmpAxis, sizeof(mds_tmpAxis));

				VectorSubtract(mds_bonePtr->translation, mds_torsoParentOffset, t);
				SKL_LocalScaledMatrixTransformVector(t, mds_thisBoneInfo->torsoWeight, mds_torsoAxis, mds_bonePtr->translation);
				VectorAdd(mds_bonePtr->translation, mds_torsoParentOffset, mds_bonePtr->translation);
			}
		}
	}

	memcpy(mds_oldBones, mds_bones, sizeof(mds_bones[0]) * header->numBones);
}

/**
 * Recursively add a bone and all its ancestors to boneList.
 * Mirrors GL's R_RecursiveBoneListAdd (tr_animation_mds.c).
 */
static void MDS_RecursiveBoneListAdd(int bi, int *boneList, int *numBones, mdsBoneInfo_t *boneInfoList)
{
	if (boneInfoList[bi].parent >= 0)
	{
		MDS_RecursiveBoneListAdd(boneInfoList[bi].parent, boneList, numBones, boneInfoList);
	}
	boneList[(*numBones)++] = bi;
}

// ===========================================================================
// MDX / MDM Bone Calculator
// (Port of tr_animation_mdm.c R_CalcBone / R_CalcBoneLerp / R_CalcBones)
// ===========================================================================

static mdxBoneFrame_t           mdx_bones[MDX_MAX_BONES];
static mdxBoneFrame_t           mdx_rawBones[MDX_MAX_BONES];
static mdxBoneFrame_t           mdx_oldBones[MDX_MAX_BONES];
static char                     mdx_validBones[MDX_MAX_BONES];
static char                     mdx_newBones[MDX_MAX_BONES];
static refEntity_t              mdx_lastEnt;

static float                    mdx_frontlerp, mdx_backlerp;
static float                    mdx_torsoFrontlerp, mdx_torsoBacklerp;

static mdxBoneFrame_t           *mdx_bonePtr, *mdx_bone, *mdx_parentBone;
static mdxBoneFrameCompressed_t *mdx_cBonePtr, *mdx_cTBonePtr;
static mdxBoneFrameCompressed_t *mdx_cOldBonePtr, *mdx_cOldTBonePtr;
static mdxBoneFrameCompressed_t *mdx_cBoneList, *mdx_cOldBoneList;
static mdxBoneFrameCompressed_t *mdx_cBoneListTorso, *mdx_cOldBoneListTorso;
static mdxBoneInfo_t            *mdx_boneInfo, *mdx_thisBoneInfo;
static mdxFrame_t               *mdx_frame, *mdx_torsoFrame;
static mdxFrame_t               *mdx_oldFrame, *mdx_oldTorsoFrame;
static int                      mdx_frameSize;

static vec3_t   mdx_angles, mdx_tangles;
static vec3_t   mdx_torsoParentOffset, mdx_torsoAxis[3];
static vec3_t   mdx_vec, mdx_v2;
static float    mdx_diff;
static qboolean mdx_isTorso, mdx_fullTorso;
static vec4_t   mdx_m1[4], mdx_m2[4];
static vec3_t   mdx_t;

/** Compute a single MDX bone without interpolation. Mirrors MDM R_CalcBone. */
static void MDX_CalcBone(int torsoParent, int boneNum)
{
	short *sh;
	float *pf;
	int    j;

	mdx_thisBoneInfo = &mdx_boneInfo[boneNum];

	if (mdx_thisBoneInfo->torsoWeight != 0.f)
	{
		mdx_cTBonePtr = &mdx_cBoneListTorso[boneNum];
		mdx_isTorso   = qtrue;
		mdx_fullTorso = (mdx_thisBoneInfo->torsoWeight == 1.0f) ? qtrue : qfalse;
	}
	else
	{
		mdx_isTorso   = qfalse;
		mdx_fullTorso = qfalse;
	}
	mdx_cBonePtr = &mdx_cBoneList[boneNum];
	mdx_bonePtr  = &mdx_bones[boneNum];

	if (mdx_thisBoneInfo->parent >= 0)
	{
		mdx_parentBone = &mdx_bones[mdx_thisBoneInfo->parent];
	}
	else
	{
		mdx_parentBone = NULL;
	}

	// rotation
	if (mdx_fullTorso)
	{
		sh = (short *)mdx_cTBonePtr->angles;
		pf = mdx_angles;
		SKL_ANGLES_SHORT_TO_FLOAT(pf, sh);
	}
	else
	{
		sh = (short *)mdx_cBonePtr->angles;
		pf = mdx_angles;
		SKL_ANGLES_SHORT_TO_FLOAT(pf, sh);
		if (mdx_isTorso)
		{
			sh = (short *)mdx_cTBonePtr->angles;
			pf = mdx_tangles;
			SKL_ANGLES_SHORT_TO_FLOAT(pf, sh);
			for (j = 0; j < 3; j++)
			{
				mdx_diff = mdx_tangles[j] - mdx_angles[j];
				if (Q_fabs(mdx_diff) > 180)
				{
					mdx_diff = AngleNormalize180(mdx_diff);
				}
				mdx_angles[j] = mdx_angles[j] + mdx_thisBoneInfo->torsoWeight * mdx_diff;
			}
		}
	}
	AnglesToAxis(mdx_angles, mdx_bonePtr->matrix);

	// translation
	if (mdx_parentBone)
	{
		if (mdx_fullTorso)
		{
			sh          = (short *)mdx_cTBonePtr->ofsAngles; pf = mdx_angles;
			*(pf++)     = SHORT2ANGLE(*(sh++)); *(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = 0;
			SKL_LocalAngleVector(mdx_angles[PITCH], mdx_angles[YAW], mdx_vec);
			VectorMA(mdx_parentBone->translation, mdx_thisBoneInfo->parentDist, mdx_vec, mdx_bonePtr->translation);
		}
		else
		{
			sh          = (short *)mdx_cBonePtr->ofsAngles; pf = mdx_angles;
			*(pf++)     = SHORT2ANGLE(*(sh++)); *(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = 0;
			SKL_LocalAngleVector(mdx_angles[PITCH], mdx_angles[YAW], mdx_vec);

			if (mdx_isTorso)
			{
				sh          = (short *)mdx_cTBonePtr->ofsAngles; pf = mdx_tangles;
				*(pf++)     = SHORT2ANGLE(*(sh++)); *(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = 0;
				SKL_LocalAngleVector(mdx_tangles[PITCH], mdx_tangles[YAW], mdx_v2);
				SKL_SLerp_Normal(mdx_vec, mdx_v2, mdx_thisBoneInfo->torsoWeight, mdx_vec);
				VectorMA(mdx_parentBone->translation, mdx_thisBoneInfo->parentDist, mdx_vec, mdx_bonePtr->translation);
			}
			else
			{
				VectorMA(mdx_parentBone->translation, mdx_thisBoneInfo->parentDist, mdx_vec, mdx_bonePtr->translation);
			}
		}
	}
	else
	{
		VectorCopy(mdx_frame->parentOffset, mdx_bonePtr->translation);
	}

	if (boneNum == torsoParent)
	{
		VectorCopy(mdx_bonePtr->translation, mdx_torsoParentOffset);
	}

	mdx_validBones[boneNum] = 1;
	mdx_rawBones[boneNum]   = *mdx_bonePtr;
	mdx_newBones[boneNum]   = 1;
}

/** Compute a single MDX bone with interpolation. Mirrors MDM R_CalcBoneLerp. */
static void MDX_CalcBoneLerp(int torsoParent, int boneNum)
{
	short *sh, *sh2;
	float *pf;
	float  a1, a2;
	int    j;

	if (boneNum < 0 || boneNum >= MDX_MAX_BONES)
	{
		return;
	}

	mdx_thisBoneInfo = &mdx_boneInfo[boneNum];

	if (mdx_thisBoneInfo->parent >= 0)
	{
		mdx_parentBone = &mdx_bones[mdx_thisBoneInfo->parent];
	}
	else
	{
		mdx_parentBone = NULL;
	}

	if (mdx_thisBoneInfo->torsoWeight != 0.f)
	{
		mdx_cTBonePtr    = &mdx_cBoneListTorso[boneNum];
		mdx_cOldTBonePtr = &mdx_cOldBoneListTorso[boneNum];
		mdx_isTorso      = qtrue;
		mdx_fullTorso    = (mdx_thisBoneInfo->torsoWeight == 1.0f) ? qtrue : qfalse;
	}
	else
	{
		mdx_isTorso   = qfalse;
		mdx_fullTorso = qfalse;
	}
	mdx_cBonePtr    = &mdx_cBoneList[boneNum];
	mdx_cOldBonePtr = &mdx_cOldBoneList[boneNum];
	mdx_bonePtr     = &mdx_bones[boneNum];
	mdx_newBones[boneNum] = 1;

	if (mdx_fullTorso)
	{
		sh  = (short *)mdx_cTBonePtr->angles;
		sh2 = (short *)mdx_cOldTBonePtr->angles;
		pf  = mdx_angles;

		a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
		mdx_diff = AngleNormalize180(a1 - a2);
		*(pf++) = a1 - mdx_torsoBacklerp * mdx_diff;
		a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
		mdx_diff = AngleNormalize180(a1 - a2);
		*(pf++) = a1 - mdx_torsoBacklerp * mdx_diff;
		a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
		mdx_diff = AngleNormalize180(a1 - a2);
		*(pf++) = a1 - mdx_torsoBacklerp * mdx_diff;
	}
	else
	{
		sh  = (short *)mdx_cBonePtr->angles;
		sh2 = (short *)mdx_cOldBonePtr->angles;
		pf  = mdx_angles;

		a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
		mdx_diff = AngleNormalize180(a1 - a2);
		*(pf++) = a1 - mdx_backlerp * mdx_diff;
		a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
		mdx_diff = AngleNormalize180(a1 - a2);
		*(pf++) = a1 - mdx_backlerp * mdx_diff;
		a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
		mdx_diff = AngleNormalize180(a1 - a2);
		*(pf++) = a1 - mdx_backlerp * mdx_diff;

		if (mdx_isTorso)
		{
			sh  = (short *)mdx_cTBonePtr->angles;
			sh2 = (short *)mdx_cOldTBonePtr->angles;
			pf  = mdx_tangles;

			a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
			mdx_diff = AngleNormalize180(a1 - a2);
			*(pf++) = a1 - mdx_torsoBacklerp * mdx_diff;
			a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
			mdx_diff = AngleNormalize180(a1 - a2);
			*(pf++) = a1 - mdx_torsoBacklerp * mdx_diff;
			a1 = SHORT2ANGLE(*(sh++)); a2 = SHORT2ANGLE(*(sh2++));
			mdx_diff = AngleNormalize180(a1 - a2);
			*(pf++) = a1 - mdx_torsoBacklerp * mdx_diff;

			for (j = 0; j < 3; j++)
			{
				mdx_diff = mdx_tangles[j] - mdx_angles[j];
				if (Q_fabs(mdx_diff) > 180)
				{
					mdx_diff = AngleNormalize180(mdx_diff);
				}
				mdx_angles[j] = mdx_angles[j] + mdx_thisBoneInfo->torsoWeight * mdx_diff;
			}
		}
	}
	AnglesToAxis(mdx_angles, mdx_bonePtr->matrix);

	// translation
	if (mdx_parentBone)
	{
		if (mdx_fullTorso)
		{
			sh  = (short *)mdx_cTBonePtr->ofsAngles;
			sh2 = (short *)mdx_cOldTBonePtr->ofsAngles;
		}
		else
		{
			sh  = (short *)mdx_cBonePtr->ofsAngles;
			sh2 = (short *)mdx_cOldBonePtr->ofsAngles;
		}

		pf      = mdx_angles;
		*(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = 0;
		SKL_LocalAngleVector(mdx_angles[PITCH], mdx_angles[YAW], mdx_v2);   // new

		pf      = mdx_angles;
		*(pf++) = SHORT2ANGLE(*(sh2++)); *(pf++) = SHORT2ANGLE(*(sh2++)); *(pf++) = 0;
		SKL_LocalAngleVector(mdx_angles[PITCH], mdx_angles[YAW], mdx_vec);  // old

		if (mdx_fullTorso)
		{
			SKL_SLerp_Normal(mdx_vec, mdx_v2, mdx_torsoFrontlerp, mdx_vec);
		}
		else
		{
			SKL_SLerp_Normal(mdx_vec, mdx_v2, mdx_frontlerp, mdx_vec);
		}

		if (!mdx_fullTorso && mdx_isTorso)
		{
			sh  = (short *)mdx_cTBonePtr->ofsAngles;
			sh2 = (short *)mdx_cOldTBonePtr->ofsAngles;

			pf      = mdx_angles;
			*(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = SHORT2ANGLE(*(sh++)); *(pf++) = 0;
			SKL_LocalAngleVector(mdx_angles[PITCH], mdx_angles[YAW], mdx_v2);

			pf      = mdx_angles;
			*(pf++) = SHORT2ANGLE(*(sh2++)); *(pf++) = SHORT2ANGLE(*(sh2++)); *(pf++) = 0;
			SKL_LocalAngleVector(mdx_angles[PITCH], mdx_angles[YAW], mdx_angles);

			SKL_SLerp_Normal(mdx_angles, mdx_v2, mdx_torsoFrontlerp, mdx_v2);
			SKL_SLerp_Normal(mdx_vec, mdx_v2, mdx_thisBoneInfo->torsoWeight, mdx_vec);
		}

		VectorMA(mdx_parentBone->translation, mdx_thisBoneInfo->parentDist, mdx_vec, mdx_bonePtr->translation);
	}
	else
	{
		mdx_bonePtr->translation[0] = mdx_frontlerp * mdx_frame->parentOffset[0] + mdx_backlerp * mdx_oldFrame->parentOffset[0];
		mdx_bonePtr->translation[1] = mdx_frontlerp * mdx_frame->parentOffset[1] + mdx_backlerp * mdx_oldFrame->parentOffset[1];
		mdx_bonePtr->translation[2] = mdx_frontlerp * mdx_frame->parentOffset[2] + mdx_backlerp * mdx_oldFrame->parentOffset[2];
	}

	if (boneNum == torsoParent)
	{
		VectorCopy(mdx_bonePtr->translation, mdx_torsoParentOffset);
	}

	mdx_validBones[boneNum] = 1;
	mdx_rawBones[boneNum]   = *mdx_bonePtr;
	mdx_newBones[boneNum]   = 1;
}

/**
 * Compute MDX bones from four MDX headers.  Mirrors MDM R_CalcBones.
 *
 * @param torsoParent  Index of the torso-parent bone (from mdxFrameHeader).
 * @param refent       Entity (frame indices, backlerp, torsoAxis…).
 * @param mdxFrame         MDX data for refent->frame / frameModel.
 * @param mdxOldFrame      MDX data for refent->oldframe / oldframeModel.
 * @param mdxTorso         MDX data for refent->torsoFrame / torsoFrameModel.
 * @param mdxOldTorso      MDX data for refent->oldTorsoFrame / oldTorsoFrameModel.
 * @param boneList         Bone indices to compute.
 * @param numBones         Count of boneList.
 */
static void MDX_CalcBones(const refEntity_t *refent,
                          mdxHeader_t *mdxFrame, mdxHeader_t *mdxOldFrame,
                          mdxHeader_t *mdxTorso, mdxHeader_t *mdxOldTorso,
                          int *boneList, int numBones)
{
	int   i, *boneRefs;
	float torsoWeight;

	if (!mdxFrame || !mdxOldFrame || !mdxTorso || !mdxOldTorso)
	{
		return;
	}

	if (memcmp(&mdx_lastEnt, refent, sizeof(refEntity_t)) != 0)
	{
		memset(mdx_validBones, 0, mdxFrame->numBones);
		mdx_lastEnt = *refent;
	}

	memset(mdx_newBones, 0, mdxFrame->numBones);

	if (refent->oldframe == refent->frame && refent->oldframeModel == refent->frameModel)
	{
		mdx_backlerp  = 0;
		mdx_frontlerp = 1;
	}
	else
	{
		mdx_backlerp  = refent->backlerp;
		mdx_frontlerp = 1.0f - refent->backlerp;
	}

	if (refent->oldTorsoFrame == refent->torsoFrame && refent->oldTorsoFrameModel == refent->oldframeModel)
	{
		mdx_torsoBacklerp  = 0;
		mdx_torsoFrontlerp = 1;
	}
	else
	{
		mdx_torsoBacklerp  = refent->torsoBacklerp;
		mdx_torsoFrontlerp = 1.0f - refent->torsoBacklerp;
	}

	// MDX frame layout: mdxFrame_t then mdxBoneFrameCompressed_t[numBones], per frame
	mdx_frameSize = (int)sizeof(mdxBoneFrameCompressed_t) * mdxFrame->numBones;

	mdx_frame = (mdxFrame_t *)((byte *)mdxFrame + mdxFrame->ofsFrames +
	                           refent->frame * (int)sizeof(mdxBoneFrameCompressed_t) * mdxFrame->numBones +
	                           refent->frame * (int)sizeof(mdxFrame_t));
	mdx_torsoFrame = (mdxFrame_t *)((byte *)mdxTorso + mdxTorso->ofsFrames +
	                                refent->torsoFrame * (int)sizeof(mdxBoneFrameCompressed_t) * mdxTorso->numBones +
	                                refent->torsoFrame * (int)sizeof(mdxFrame_t));
	mdx_oldFrame = (mdxFrame_t *)((byte *)mdxOldFrame + mdxOldFrame->ofsFrames +
	                              refent->oldframe * (int)sizeof(mdxBoneFrameCompressed_t) * mdxOldFrame->numBones +
	                              refent->oldframe * (int)sizeof(mdxFrame_t));
	mdx_oldTorsoFrame = (mdxFrame_t *)((byte *)mdxOldTorso + mdxOldTorso->ofsFrames +
	                                   refent->oldTorsoFrame * (int)sizeof(mdxBoneFrameCompressed_t) * mdxOldTorso->numBones +
	                                   refent->oldTorsoFrame * (int)sizeof(mdxFrame_t));

	mdx_cBoneList = (mdxBoneFrameCompressed_t *)((byte *)mdxFrame + mdxFrame->ofsFrames +
	                                             (refent->frame + 1) * (int)sizeof(mdxFrame_t) +
	                                             refent->frame * mdx_frameSize);
	mdx_cBoneListTorso = (mdxBoneFrameCompressed_t *)((byte *)mdxTorso + mdxTorso->ofsFrames +
	                                                  (refent->torsoFrame + 1) * (int)sizeof(mdxFrame_t) +
	                                                  refent->torsoFrame * mdx_frameSize);

	mdx_boneInfo = (mdxBoneInfo_t *)((byte *)mdxFrame + mdxFrame->ofsBones);
	SKL_Matrix3Transpose(refent->torsoAxis, mdx_torsoAxis);

	boneRefs = boneList;
	const int torsoParent = mdxFrame->torsoParent;

	if (mdx_backlerp == 0.f && mdx_torsoBacklerp == 0.f)
	{
		for (i = 0; i < numBones; i++, boneRefs++)
		{
			if (mdx_validBones[*boneRefs])
			{
				mdx_bones[*boneRefs] = mdx_rawBones[*boneRefs];
				continue;
			}
			if (mdx_boneInfo[*boneRefs].parent >= 0 &&
			    !mdx_validBones[mdx_boneInfo[*boneRefs].parent] &&
			    !mdx_newBones[mdx_boneInfo[*boneRefs].parent])
			{
				MDX_CalcBone(torsoParent, mdx_boneInfo[*boneRefs].parent);
			}
			MDX_CalcBone(torsoParent, *boneRefs);
		}
	}
	else
	{
		mdx_cOldBoneList = (mdxBoneFrameCompressed_t *)((byte *)mdxOldFrame + mdxOldFrame->ofsFrames +
		                                                (refent->oldframe + 1) * (int)sizeof(mdxFrame_t) +
		                                                refent->oldframe * mdx_frameSize);
		mdx_cOldBoneListTorso = (mdxBoneFrameCompressed_t *)((byte *)mdxOldTorso + mdxOldTorso->ofsFrames +
		                                                     (refent->oldTorsoFrame + 1) * (int)sizeof(mdxFrame_t) +
		                                                     refent->oldTorsoFrame * mdx_frameSize);

		for (i = 0; i < numBones; i++, boneRefs++)
		{
			if (mdx_validBones[*boneRefs])
			{
				mdx_bones[*boneRefs] = mdx_rawBones[*boneRefs];
				continue;
			}
			if (mdx_boneInfo[*boneRefs].parent >= 0 &&
			    !mdx_validBones[mdx_boneInfo[*boneRefs].parent] &&
			    !mdx_newBones[mdx_boneInfo[*boneRefs].parent])
			{
				MDX_CalcBoneLerp(torsoParent, mdx_boneInfo[*boneRefs].parent);
			}
			MDX_CalcBoneLerp(torsoParent, *boneRefs);
		}
	}

	// adjust for torso rotations
	torsoWeight = 0;
	boneRefs    = boneList;
	for (i = 0; i < numBones; i++, boneRefs++)
	{
		mdx_thisBoneInfo = &mdx_boneInfo[*boneRefs];
		mdx_bonePtr      = &mdx_bones[*boneRefs];

		if (mdx_thisBoneInfo->torsoWeight > 0)
		{
			if (!mdx_newBones[*boneRefs])
			{
				mdx_bones[*boneRefs] = mdx_oldBones[*boneRefs];
				continue;
			}

			VectorSubtract(mdx_bonePtr->translation, mdx_torsoParentOffset, mdx_t);
			SKL_Matrix4FromAxisPlusTranslation(mdx_bonePtr->matrix, mdx_t, mdx_m1);

			if (torsoWeight != mdx_thisBoneInfo->torsoWeight)
			{
				SKL_Matrix4FromScaledAxisPlusTranslation(mdx_torsoAxis, mdx_thisBoneInfo->torsoWeight, mdx_torsoParentOffset, mdx_m2);
				torsoWeight = mdx_thisBoneInfo->torsoWeight;
			}
			SKL_Matrix4MultiplyInto3x3AndTranslation(mdx_m2, mdx_m1, mdx_bonePtr->matrix, mdx_bonePtr->translation);
		}
	}

	memcpy(mdx_oldBones, mdx_bones, sizeof(mdx_bones[0]) * mdxFrame->numBones);
}

/**
 * Recursively add an MDX bone and all its ancestors to boneList.
 * Mirrors GL's MDM equivalent.
 */
static void MDX_RecursiveBoneListAdd(int bi, int *boneList, int *numBones, mdxBoneInfo_t *boneInfoList)
{
	if (boneInfoList[bi].parent >= 0)
	{
		MDX_RecursiveBoneListAdd(boneInfoList[bi].parent, boneList, numBones, boneInfoList);
	}
	boneList[(*numBones)++] = bi;
}

// ===========================================================================
// Public API – Tag lookup
// ===========================================================================

/**
 * @brief DX12_GetBoneTagMDS
 *
 * Mirrors GL's R_GetBoneTag (tr_animation_mds.c).
 */
int DX12_GetBoneTagMDS(orientation_t *outTag, mdsHeader_t *mds,
                       const refEntity_t *refent, const char *tagName, int startTagIndex)
{
	int           i;
	mdsTag_t      *pTag;
	mdsBoneInfo_t *boneInfoList;
	int           boneList[MDS_MAX_BONES];
	int           numBones;

	if (!mds || !outTag || !refent || !tagName)
	{
		if (outTag)
		{
			AxisClear(outTag->axis);
			VectorClear(outTag->origin);
		}
		return -1;
	}

	if (startTagIndex >= mds->numTags)
	{
		AxisClear(outTag->axis);
		VectorClear(outTag->origin);
		return -1;
	}

	// find the named tag
	pTag = (mdsTag_t *)((byte *)mds + mds->ofsTags) + startTagIndex;

	for (i = startTagIndex; i < mds->numTags; i++, pTag++)
	{
		if (!strcmp(pTag->name, tagName))
		{
			break;
		}
	}

	if (i >= mds->numTags)
	{
		AxisClear(outTag->axis);
		VectorClear(outTag->origin);
		return -1;
	}

	// build the bone dependency list for this tag
	boneInfoList = (mdsBoneInfo_t *)((byte *)mds + mds->ofsBones);
	numBones     = 0;
	MDS_RecursiveBoneListAdd(pTag->boneIndex, boneList, &numBones, boneInfoList);

	// compute the bones
	MDS_CalcBones(mds, refent, boneList, numBones);

	// extract the orientation from the bone that represents this tag
	memcpy(outTag->axis, mds_bones[pTag->boneIndex].matrix, sizeof(outTag->axis));
	VectorCopy(mds_bones[pTag->boneIndex].translation, outTag->origin);

	return i;
}

/**
 * @brief DX12_GetBoneTagMDM
 *
 * Mirrors GL's R_MDM_GetBoneTag (tr_animation_mdm.c).
 */
int DX12_GetBoneTagMDM(orientation_t *outTag, mdmHeader_t *mdm,
                       mdxHeader_t *mdxFrameHdr, mdxHeader_t *mdxOldFrameHdr,
                       mdxHeader_t *mdxTorsoHdr, mdxHeader_t *mdxOldTorsoHdr,
                       const refEntity_t *refent, const char *tagName, int startTagIndex)
{
	int      i, j;
	mdmTag_t *pTag;
	int      *boneList;
	int      numBones;

	if (!mdm || !outTag || !refent || !tagName ||
	    !mdxFrameHdr || !mdxOldFrameHdr || !mdxTorsoHdr || !mdxOldTorsoHdr)
	{
		if (outTag)
		{
			AxisClear(outTag->axis);
			VectorClear(outTag->origin);
		}
		return -1;
	}

	if (startTagIndex >= mdm->numTags)
	{
		AxisClear(outTag->axis);
		VectorClear(outTag->origin);
		return -1;
	}

	// find the named tag (tags use linked-list offsets: pTag->ofsEnd)
	pTag = (mdmTag_t *)((byte *)mdm + mdm->ofsTags);
	for (i = 0; i < startTagIndex; i++)
	{
		pTag = (mdmTag_t *)((byte *)pTag + pTag->ofsEnd);
	}

	for (i = startTagIndex; i < mdm->numTags; i++)
	{
		if (!strcmp(pTag->name, tagName))
		{
			break;
		}
		pTag = (mdmTag_t *)((byte *)pTag + pTag->ofsEnd);
	}

	if (i >= mdm->numTags)
	{
		AxisClear(outTag->axis);
		VectorClear(outTag->origin);
		return -1;
	}

	// Use the tag's own bone reference list directly.
	// The exporter pre-computes this to include the full ancestor chain.
	// This matches GL's R_MDM_GetBoneTag which uses pTag->ofsBoneReferences directly.
	boneList = (int *)((byte *)pTag + pTag->ofsBoneReferences);
	numBones = pTag->numBoneReferences;

	// compute the bones
	MDX_CalcBones(refent, mdxFrameHdr, mdxOldFrameHdr, mdxTorsoHdr, mdxOldTorsoHdr,
	              boneList, numBones);

	// extract orientation from the tag's bone
	mdx_bone = &mdx_bones[pTag->boneIndex];
	VectorClear(outTag->origin);
	SKL_LocalAddScaledMatrixTransformVectorTranslate(pTag->offset, 1.f, mdx_bone->matrix, mdx_bone->translation, outTag->origin);
	for (j = 0; j < 3; j++)
	{
		SKL_LocalMatrixTransformVector(pTag->axis[j], mdx_bone->matrix, outTag->axis[j]);
	}

	return i;
}

// ===========================================================================
// Loaders
// ===========================================================================

/**
 * @brief DX12_LoadMDS
 */
qboolean DX12_LoadMDS(const char *name, void **outData, int *outSize)
{
	void         *buf    = NULL;
	int           len;
	mdsHeader_t  *header;

	if (!name || !outData || !outSize)
	{
		return qfalse;
	}

	len = dx12.ri.FS_ReadFile(name, &buf);
	if (len <= 0 || !buf)
	{
		return qfalse;
	}

	header = (mdsHeader_t *)buf;

	if (header->ident != MDS_IDENT)
	{
		dx12.ri.FS_FreeFile(buf);
		return qfalse;
	}
	if (header->version != MDS_VERSION)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_LoadMDS: %s has wrong version (%d should be %d)\n",
		               name, header->version, MDS_VERSION);
		dx12.ri.FS_FreeFile(buf);
		return qfalse;
	}
	if (header->numFrames < 1)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_LoadMDS: %s has no frames\n", name);
		dx12.ri.FS_FreeFile(buf);
		return qfalse;
	}

	// Heap-copy the raw data so we own it independently of the VFS cache
	*outData = dx12.ri.Z_Malloc((size_t)len);
	if (!*outData)
	{
		dx12.ri.FS_FreeFile(buf);
		return qfalse;
	}
	memcpy(*outData, buf, (size_t)len);
	*outSize = len;

	dx12.ri.FS_FreeFile(buf);
	return qtrue;
}

/**
 * @brief DX12_LoadMDX
 */
qboolean DX12_LoadMDX(const char *name, void **outData, int *outSize)
{
	void         *buf  = NULL;
	int           len;
	mdxHeader_t  *header;

	if (!name || !outData || !outSize)
	{
		return qfalse;
	}

	len = dx12.ri.FS_ReadFile(name, &buf);
	if (len <= 0 || !buf)
	{
		return qfalse;
	}

	header = (mdxHeader_t *)buf;

	if (header->ident != MDX_IDENT)
	{
		dx12.ri.FS_FreeFile(buf);
		return qfalse;
	}
	if (header->version != MDX_VERSION)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_LoadMDX: %s has wrong version (%d should be %d)\n",
		               name, header->version, MDX_VERSION);
		dx12.ri.FS_FreeFile(buf);
		return qfalse;
	}

	*outData = dx12.ri.Z_Malloc((size_t)len);
	if (!*outData)
	{
		dx12.ri.FS_FreeFile(buf);
		return qfalse;
	}
	memcpy(*outData, buf, (size_t)len);
	*outSize = len;

	dx12.ri.FS_FreeFile(buf);
	return qtrue;
}

/**
 * @brief DX12_LoadMDM
 */
qboolean DX12_LoadMDM(const char *name, void **outData, int *outSize)
{
	void         *buf  = NULL;
	int           len;
	mdmHeader_t  *header;

	if (!name || !outData || !outSize)
	{
		return qfalse;
	}

	len = dx12.ri.FS_ReadFile(name, &buf);
	if (len <= 0 || !buf)
	{
		return qfalse;
	}

	header = (mdmHeader_t *)buf;

	if (header->ident != MDM_IDENT)
	{
		dx12.ri.FS_FreeFile(buf);
		return qfalse;
	}
	if (header->version != MDM_VERSION)
	{
		dx12.ri.Printf(PRINT_WARNING, "DX12_LoadMDM: %s has wrong version (%d should be %d)\n",
		               name, header->version, MDM_VERSION);
		dx12.ri.FS_FreeFile(buf);
		return qfalse;
	}

	*outData = dx12.ri.Z_Malloc((size_t)len);
	if (!*outData)
	{
		dx12.ri.FS_FreeFile(buf);
		return qfalse;
	}
	memcpy(*outData, buf, (size_t)len);
	*outSize = len;

	dx12.ri.FS_FreeFile(buf);
	return qtrue;
}

/**
 * @brief DX12_FreeSkeletal
 */
void DX12_FreeSkeletal(void *data)
{
	if (data)
	{
		dx12.ri.Free(data);
	}
}

#endif // _WIN32
