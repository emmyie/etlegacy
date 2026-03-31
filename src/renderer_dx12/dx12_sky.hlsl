// dx12_sky.hlsl — Sky surface shader for the DX12 renderer.
// Extracted from dx12_scene.cpp for shader debugging support.

cbuffer SceneConstants : register(b0)
{
    float4x4 viewProj;
    float4x4 modelMatrix;
    float4   cameraPos;
    float4   fogColor;
    float    fogStart;
    float    fogEnd;
    float    fogEnabled;
    float    overBrightFactor;
    float4   entityAmbient;
    float4   entityDirected;
    float4   entityLightDir;
};

cbuffer PerSurfConstants : register(b1)
{
    float uvM00;
    float uvM01;
    float uvOffsetU;
    float uvM10;
    float uvM11;
    float uvOffsetV;
    float alphaTestThreshold;
    float isEntity;
    float stageColorR;
    float stageColorG;
    float stageColorB;
    float stageColorA;
};

Texture2D    gDiffuse  : register(t0);
SamplerState gSampler  : register(s0);

struct VSIn
{
    float3 pos   : POSITION;
    float2 st    : TEXCOORD0;
    float2 lm    : TEXCOORD1;
    float3 norm  : NORMAL;
    float4 col   : COLOR;
};

struct VSOut
{
    float4 pos    : SV_POSITION;
    float2 st     : TEXCOORD0;
};

VSOut VSMain(VSIn vin)
{
    VSOut vout;
    // Remove translation from viewProj so sky stays centred on the camera.
    // HLSL reads the C row-major matrix as column-major, so the translation
    // terms live in column 3 (rows 0-2) of the HLSL matrix.  Zero those three
    // elements so that the w=1 position component contributes no translation.
    float4x4 vpNoTrans = viewProj;
    vpNoTrans[0][3] = 0.0f;
    vpNoTrans[1][3] = 0.0f;
    vpNoTrans[2][3] = 0.0f;
    float4 clipPos = mul(vpNoTrans, float4(vin.pos, 1.0f));
    // Force depth to far plane (NDC depth = 1.0) by setting w = z.
    vout.pos = clipPos.xyww;
    vout.st  = float2(uvM00 * vin.st.x + uvM01 * vin.st.y + uvOffsetU,
                      uvM10 * vin.st.x + uvM11 * vin.st.y + uvOffsetV);
    return vout;
}

float4 PSMain(VSOut pin) : SV_TARGET
{
    float4 col = gDiffuse.Sample(gSampler, pin.st);
    if (alphaTestThreshold > 0.0f && col.a < alphaTestThreshold) { discard; }
    // Apply rgbGen / alphaGen stage color modulator.
    col.rgb = saturate(col.rgb * float3(stageColorR, stageColorG, stageColorB));
    col.a   = saturate(col.a   * stageColorA);
    return col;
}
