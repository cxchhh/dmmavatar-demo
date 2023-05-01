export function getVertexShaderSource(J) {
    return `#version 300 es
    // an attribute is an input (in) to a vertex shader.
    // It will receive data from a buffer
    #define VINROW 40
    in vec4 a_position;
    in vec3 a_normal;
    in vec2 a_index;
    in vec2 a_uv;
    int idx;
    uniform sampler2D betasTex;
    uniform sampler2D shapedirsTex;
    uniform sampler2D posesTex;
    uniform sampler2D posedirsTex;
    uniform sampler2D transformTex;
    uniform sampler2D lbsweightTex;
    // A matrix to transform the positions by
    uniform mat4 u_matrix;
    uniform mat4 u_normal_matrix;
    uniform mat4 u_view_matrix;
    // a varying the color to the fragment shader
    //out vec4 v_color;
    out vec2 v_uv;
    out vec3 v_normal;
    out vec3 v_viewdir;
    out vec2 v_idx;
    float betas(int i){
        return texelFetch(betasTex,ivec2(i,0),0).x;
    }
    vec3 shapedirs(int i,int j){
        j=(i%VINROW)*50+j;
        i=i/VINROW;
        return texelFetch(shapedirsTex,ivec2(j,i),0).xyz;
    }
    float poses(int i){
        return texelFetch(posesTex,ivec2(i,0),0).x;
    }
    vec3 posedirs(int i,int j){
        j=(i%VINROW)*36+j;
        i=i/VINROW;
        return texelFetch(posedirsTex,ivec2(j,i),0).xyz;
    }
    float transform(int i,int j){
        return texelFetch(transformTex,ivec2(j,i),0).x;
    }
    float lbsweight(int i,int j){
        j=(i%VINROW)*${J}+j;
        i=i/VINROW;
        return texelFetch(lbsweightTex,ivec2(j,i),0).x;
    }
    float random (vec2 st) {
        return fract(sin(dot(st.xy,vec2(12.9898,78.233)))*43758.5453123);
    }
    vec4 shapeMatMul(){
        vec4 sum=vec4(0,0,0,0);
        float b=0.0;
        for(int i=0;i<50;i++){
            b=betas(i);
            sum.xyz+=shapedirs(idx,i)*b;
        }
        return sum;
    }
    vec4 poseMatMul(){
        vec4 sum=vec4(0,0,0,0);
        float p=0.0;
        for(int i=0;i<36;i++){
            p=poses(i);
            sum.xyz+=posedirs(idx,i)*p;
        }
        return sum;
    }
    mat4 lbsMatMul(){
        mat4 rot=mat4(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
        for(int i=0;i<16;i++){
            for(int j=0;j<${J};j++){
                rot[i>>2][i&3]+=lbsweight(idx,j)*transform(j,i);
            }
        }
        return transpose(rot);
    }
    // all shaders have a main function
    void main() { 
        idx=int(a_index.x);
        float fidx=float(idx);
        vec4 apos=a_position;
        apos+=shapeMatMul()+poseMatMul();
        apos=lbsMatMul()*apos;
        apos=vec4(apos.xyz*200.0f,1);
        gl_Position = u_matrix * apos;

        v_viewdir = normalize((u_view_matrix * apos).xyz);
        v_uv=a_uv;
        v_normal=normalize(mat3(u_normal_matrix)*a_normal);
        v_idx=vec2(fidx,0);
    }`;
}

export function getFragmentShaderSource() {
    return `#version 300 es
    precision highp float;

    // the varied color passed from the vertex shader
    //in vec4 v_color;
    in vec2 v_uv;
    in vec3 v_normal;
    in vec3 v_viewdir;
    in vec2 v_idx;
    // we need to declare an output for the fragment shader
    out vec4 outColor;
    void main() {
        outColor = vec4((v_normal+1.0)/2.0,1);
        outColor.w=0.8;
    }`;
}