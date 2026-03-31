// dx12_world.hlsl — World/entity surface shader for the DX12 renderer.
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

cbuffer PerSurface : register(b1)
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
    float useLightmap;
    float useVertexColor;
};

Texture2D    g_diffuse  : register(t0);
Texture2D    g_lightmap : register(t1);
SamplerState g_sampler  : register(s0);

struct VSInput
{
    float3 pos    : POSITION;
    float2 uv     : TEXCOORD0;
    float2 lm     : TEXCOORD1;
    float3 normal : NORMAL;
    float4 color  : COLOR;
};

struct PSInput
{
    float4 pos      : SV_POSITION;
    float2 uv       : TEXCOORD0;
    float2 lm       : TEXCOORD1;
    float4 color    : COLOR;
    float3 worldPos : TEXCOORD2;
    float3 normal   : TEXCOORD3;
};

PSInput VSMain(VSInput input)
{
    PSInput o;
    float4 worldPos4 = mul(modelMatrix, float4(input.pos, 1.0));
    o.pos      = mul(viewProj, worldPos4);
    // Apply 2x3 affine UV transform: u' = M00*u + M01*v + offsetU
    o.uv       = float2(uvM00 * input.uv.x + uvM01 * input.uv.y + uvOffsetU,
                        uvM10 * input.uv.x + uvM11 * input.uv.y + uvOffsetV);
    o.lm       = input.lm;
    o.color    = input.color;
    o.worldPos = worldPos4.xyz;
    // Transform normal to world space (rotation part of modelMatrix only)
    o.normal   = normalize(mul((float3x3)modelMatrix, input.normal));
    return o;
}

float4 PSMain(PSInput input) : SV_TARGET
{
    float4 diffuse = g_diffuse.Sample(g_sampler, input.uv);

    // Alpha test: positive threshold = GE (keep if alpha >= threshold),
    // negative = LT (keep if alpha < |threshold|).
    if (alphaTestThreshold > 0.0)
    {
        clip(diffuse.a - alphaTestThreshold);
    }
    else if (alphaTestThreshold < 0.0)
    {
        clip(-alphaTestThreshold - diffuse.a);
    }

    float4 result;
    if (isEntity > 0.0)
    {
        // Entity shading: ambient + N.L * directed, then overbright.
        float  nDotL  = saturate(dot(input.normal, entityLightDir.xyz));
        float3 light  = entityAmbient.rgb + nDotL * entityDirected.rgb;
        result = diffuse * float4(saturate(light * overBrightFactor), 1.0);
    }
    else if (useLightmap > 0.0)
    {
        // Lightmapped world surface.
        float4 lightmap = g_lightmap.Sample(g_sampler, input.lm);
        result = diffuse * (lightmap * overBrightFactor);
    }
    else if (useVertexColor > 0.0)
    {
        // Vertex-lit surface (rgbGen vertex / exactVertex).
        result = diffuse * input.color;
    }
    else
    {
        // rgbGen identity / constant / wave: no per-vertex colour.
        result = diffuse;
    }
    result = float4(saturate(result.rgb), result.a);

    // Per-stage color modulator from rgbGen / alphaGen.
    float4 stageColor = float4(stageColorR, stageColorG, stageColorB, stageColorA);
    result = float4(saturate(result.rgb * stageColor.rgb),
                    saturate(result.a   * stageColor.a));

    // Linear depth fog
    if (fogEnabled > 0.0)
    {
        float viewDist  = length(input.worldPos - cameraPos.xyz);
        float fogFactor = saturate((fogEnd - viewDist) / max(fogEnd - fogStart, 1.0));
        result.rgb      = lerp(fogColor.rgb, result.rgb, fogFactor);
    }
    // Gamma correction: entityLightDir.w holds 1/gamma (set by CPU each frame).
    float invGamma = entityLightDir.w;
    if (invGamma > 0.01)
    {
        result.rgb = pow(max(result.rgb, float3(0.0001, 0.0001, 0.0001)), invGamma);
    }
    return result;
}
