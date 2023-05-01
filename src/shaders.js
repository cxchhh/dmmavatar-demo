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
    uniform mat4 u_matrix;
    uniform mat4 u_normal_matrix;
    uniform mat4 u_view_matrix;

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
    in vec2 v_uv;
    in vec3 v_normal;
    in vec3 v_viewdir;
    in vec2 v_idx;
    vec4 v_posf0,v_posf1;
    uniform sampler2D pos_featureTex;
    uniform sampler2D sfc0Tex;
    uniform sampler2D sfc1Tex;
    uniform sampler2D imgTex;
    float out1[16],coef[8];
    float sfc0(int i,int j){
        return texelFetch(sfc0Tex,ivec2(j,i),0).x;
    }
    float sfc1(int i,int j){
        return texelFetch(sfc1Tex,ivec2(j,i),0).x;
    }
    // we need to declare an output for the fragment shader
    out vec4 outColor;
    void main() {
        v_posf0=texture(pos_featureTex,vec2(v_uv.x/2.0,v_uv.y));
        v_posf1=texture(pos_featureTex,vec2(v_uv.x/2.0+0.5,v_uv.y));
        for(int i=0;i<16;i++){
            out1[i]=0.0;
            out1[i]+=dot(vec4(sfc0(i,0),sfc0(i,1),sfc0(i,2),sfc0(i,3)),v_posf0);
            out1[i]+=dot(vec4(sfc0(i,4),sfc0(i,5),sfc0(i,6),sfc0(i,7)),v_posf1);
            out1[i]+=dot(vec3(sfc0(i,8),sfc0(i,9),sfc0(i,10)),v_normal);
            out1[i]+=dot(vec3(sfc0(i,11),sfc0(i,12),sfc0(i,13)),v_viewdir);
            out1[i]=max(out1[i],0.0);
        }
        float esum=0.0;
        for(int i=0;i<8;i++){
            coef[i]=0.0;
            for(int j=0;j<16;j++){
                coef[i]+=sfc1(i,j)*out1[j];
            }
            coef[i]=exp(10.0*coef[i]);
            esum+=coef[i];
        }
        //outColor = vec4((v_normal+1.0)/2.0,0.8);
        vec4 sumColor=vec4(0,0,0,0),tmpColor;
        for(int i=0;i<8;i++){
            coef[i]/=esum;
            tmpColor=texture(imgTex,vec2(v_uv.x/4.0+0.25*float(i&3),v_uv.y/2.0+0.5*float(i>>2)));
            sumColor+=tmpColor*coef[i];
        }
        vec4 c0=vec4(0.12156862745098039, 0.4666666666666667, 0.7058823529411765,1);
        vec4 c1=vec4(1.0, 0.4980392156862745, 0.054901960784313725,1);
        vec4 c2=vec4(0.17254901960784313, 0.6274509803921569, 0.17254901960784313,1);
        vec4 c3=vec4(0.8392156862745098, 0.15294117647058825, 0.1568627450980392,1); 
        vec4 c4=vec4(0.5803921568627451, 0.403921568627451, 0.7411764705882353,1); 
        vec4 c5=vec4(0.5490196078431373, 0.33725490196078434, 0.29411764705882354,1);
        vec4 c6=vec4(0.8901960784313725, 0.4666666666666667, 0.7607843137254902,1);
        vec4 c7=vec4(0.4980392156862745, 0.4980392156862745, 0.4980392156862745,1);
        //outColor=c0*coef[0]+c1*coef[1]+c2*coef[2]+c3*coef[3]+c4*coef[4]+c5*coef[5]+c6*coef[6]+c7*coef[7];
        outColor=sumColor;
        //outColor.w=1.0;
    }`;
}