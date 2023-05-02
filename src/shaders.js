export function getVertexShaderSource(J) {
    return `#version 300 es
    // an attribute is an input (in) to a vertex shader.
    // It will receive data from a buffer
    #define VINROW 40
    in vec4 a_position;
    in vec3 a_normal;
    in vec2 a_index;
    in vec2 a_uv;
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
    out vec3 v_viewpos;
    int idx;

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
    vec3 shapeMatMul(){
        vec3 sum=vec3(0,0,0);
        float b=0.0;
        for(int i=0;i<50;i++){
            b=betas(i);
            sum+=shapedirs(idx,i)*b;
        }
        return sum;
    }
    vec3 poseMatMul(){
        vec3 sum=vec3(0,0,0);
        float p=0.0;
        for(int i=0;i<36;i++){
            p=poses(i);
            sum+=posedirs(idx,i)*p;
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
        idx = int(a_index.x);
        vec3 blendshape = shapeMatMul() + poseMatMul();
        mat4 transform = lbsMatMul();
        mat3 transform_normal = mat3(transpose(inverse(transform)));
        vec4 pos = transform * (a_position + vec4(blendshape, 0));

        v_viewpos = (u_view_matrix * pos).xyz;
        v_uv = a_uv;
        v_normal = normalize(mat3(u_normal_matrix) * transform_normal * a_normal);

        gl_Position = u_matrix * pos;
    }`;
}

export function getFragmentShaderSource() {
    return `#version 300 es
    precision highp float;
    in vec2 v_uv;
    in vec3 v_normal;
    in vec3 v_viewpos;
    uniform sampler2D posfeatTex;
    uniform sampler2D radfeatTex;
    uniform sampler2D sfcTex;
    uniform int depthOnly;
    out vec4 outColor;

    vec2 posfeat_s = 1.0 / vec2(2.0,1.0);
    vec2 radiance_s = 1.0 / vec2(4.0,2.0);

    float coef[8];
    float sfc0(int i,int j) {
        return texelFetch(sfcTex,ivec2(j,i),0).x;
    }
    float sfc1(int i,int j) {
        return texelFetch(sfcTex,ivec2(j,i+16),0).x;
    }
    vec4 posfeat(int i) {
        vec2 uv_offset = vec2(float(i),0.0);
        return texture(posfeatTex,(v_uv+uv_offset)*posfeat_s);
    }
    vec4 radianceBasis(int i) {
        vec2 uv_offset = vec2(float(i&3),float(i>>2));
        return texture(radfeatTex,(v_uv+uv_offset)*radiance_s);
    }
    void runSpatialMLP(vec4 posf0, vec4 posf1, vec3 normal, vec3 viewdir) {
        float out1[16];
        for(int i=0;i<16;i++){
            out1[i]=0.0;
            out1[i]+=dot(vec4(sfc0(i,0),sfc0(i,1),sfc0(i,2),sfc0(i,3)),posf0);
            out1[i]+=dot(vec4(sfc0(i,4),sfc0(i,5),sfc0(i,6),sfc0(i,7)),posf1);
            out1[i]+=dot(vec3(sfc0(i,8),sfc0(i,9),sfc0(i,10)),normal);
            out1[i]+=dot(vec3(sfc0(i,11),sfc0(i,12),sfc0(i,13)),viewdir);
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
        float inv_esum=1.0/esum;
        for(int i=0;i<8;i++){
            coef[i]*=inv_esum;
        }
    }
    vec4 coefColor() {
        vec4 c0=vec4(0.12156862745098039, 0.4666666666666667, 0.7058823529411765,1);
        vec4 c1=vec4(1.0, 0.4980392156862745, 0.054901960784313725,1);
        vec4 c2=vec4(0.17254901960784313, 0.6274509803921569, 0.17254901960784313,1);
        vec4 c3=vec4(0.8392156862745098, 0.15294117647058825, 0.1568627450980392,1); 
        vec4 c4=vec4(0.5803921568627451, 0.403921568627451, 0.7411764705882353,1); 
        vec4 c5=vec4(0.5490196078431373, 0.33725490196078434, 0.29411764705882354,1);
        vec4 c6=vec4(0.8901960784313725, 0.4666666666666667, 0.7607843137254902,1);
        vec4 c7=vec4(0.4980392156862745, 0.4980392156862745, 0.4980392156862745,1);
        return c0*coef[0]+c1*coef[1]+c2*coef[2]+c3*coef[3]+c4*coef[4]+c5*coef[5]+c6*coef[6]+c7*coef[7];
    }
    void main() {
        if (depthOnly != 0)
            return;

        vec3 v_viewdir=normalize(v_viewpos);
        vec4 v_posf0=posfeat(0);
        vec4 v_posf1=posfeat(1);
        runSpatialMLP(v_posf0, v_posf1, v_normal, v_viewdir);
        
        vec4 sumColor=vec4(0);
        for(int i=0;i<8;i++){
            sumColor+=coef[i]*radianceBasis(i);
        }

        //outColor=radianceBasis(0);
        //outColor=posfeat(0);
        //outColor=vec4((v_normal+1.0)/2.0,1.0);
        //outColor=vec4((v_viewdir+1.0)/2.0,1.0);
        //outColor=coefColor();
        outColor=sumColor;
        //outColor=vec4(vec3(outColor)*outColor.w+vec3(1.0)*(1.0-outColor.w),1.0);
    }`;
}
