"use strict";
import ndarray from "ndarray";
import cwise from "cwise";
import zeros from "zeros";
import { GPU } from "gpu.js";
import show from "ndarray-show";
import FLAME from "./flame.js";

class Renderer {
    constructor(vertices, faces, lbs_weights, posedirs, shapedirs) {
        this.vertices = vertices;
        this.faces = faces;
        this.lbs_weights = lbs_weights;
        this.posedirs = posedirs;
        this.shapedirs = shapedirs;
        this.V = vertices.shape[0];
        this.J = lbs_weights.shape[1];
        this.gpu = new GPU();
        this.canvas = document.querySelector("#canvas");
        this.gl = this.canvas.getContext("webgl2");
        this.index = new Int32Array(this.V);
        for (var i = 0; i < this.V; i++) {
            this.index[i] = i;
        }
        //console.log(this.index);
        this.vertexShaderSource = `#version 300 es
        // an attribute is an input (in) to a vertex shader.
        // It will receive data from a buffer
        in vec4 a_position;
        in vec2 a_index;
        int idx;
        uniform sampler2D betasTex;
        uniform sampler2D shapedirsTex;
        uniform sampler2D posesTex;
        uniform sampler2D posedirsTex;
        uniform sampler2D transformTex;
        uniform sampler2D lbsweightTex;
        // A matrix to transform the positions by
        uniform mat4 u_matrix;
        // a varying the color to the fragment shader
        out vec4 v_color;
        float betas(int i){
            return texelFetch(betasTex,ivec2(i,0),0).x;
        }
        float shapedirs(int i,int j,int k){
            j=(i%11)*50+j;
            i=i/11;
            if(k==0) return texelFetch(shapedirsTex,ivec2(j,i),0).x;
            if(k==1) return texelFetch(shapedirsTex,ivec2(j,i),0).y;
            if(k==2) return texelFetch(shapedirsTex,ivec2(j,i),0).z;
        }
        float poses(int i){
            return texelFetch(posesTex,ivec2(i,0),0).x;
        }
        float posedirs(int i,int j,int k){
            j=(i%11)*36+j;
            i=i/11;
            if(k==0) return texelFetch(posedirsTex,ivec2(j,i),0).x;
            if(k==1) return texelFetch(posedirsTex,ivec2(j,i),0).y;
            if(k==2) return texelFetch(posedirsTex,ivec2(j,i),0).z;
        }
        float transform(int i,int j){
            return texelFetch(transformTex,ivec2(j,i),0).x;
        }
        float lbsweight(int i,int j){
            j=(i%11)*5+j;
            i=i/11;
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
                sum.x+=shapedirs(idx,i,0)*b;
                sum.y+=shapedirs(idx,i,1)*b;
                sum.z+=shapedirs(idx,i,2)*b;
            }
            return sum;
        }
        vec4 poseMatMul(){
            vec4 sum=vec4(0,0,0,0);
            float p=0.0;
            for(int i=0;i<36;i++){
                p=poses(i);
                sum.x+=posedirs(idx,i,0)*p;
                sum.y+=posedirs(idx,i,1)*p;
                sum.z+=posedirs(idx,i,2)*p;
            }
            return sum;
        }
        mat4 lbsMatMul(){
            mat4 rot=mat4(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
            for(int i=0;i<16;i++){
                for(int j=0;j<5;j++){
                    rot[i/4][i%4]+=lbsweight(idx,j)*transform(j,i);
                }
            }
            return rot;
        }
        // all shaders have a main function
        void main() { 
            idx=int(a_index.x);
            float fidx=float(idx);
            vec4 apos=a_position;
            apos+=shapeMatMul()+poseMatMul();
            apos=lbsMatMul()*apos;
            apos=vec4(apos.xyz*200.0f,1);
            gl_Position = u_matrix * (apos);
            // Pass the color to the fragment shader.
            v_color = vec4(idx,0,lbsweight(1,1),1);
        }`;
        this.fragmentShaderSource = `#version 300 es
            precision highp float;

            // the varied color passed from the vertex shader
            in vec4 v_color;

            // we need to declare an output for the fragment shader
            out vec4 outColor;

            void main() {
                outColor = v_color;
            }`;
    }

    init(betas, transform, pose_feature) {
        //var st=Date.now();
        const shapeMatMul = this.gpu.createKernel(function (a, b, v) {
            let sum = 0;
            for (let i = 0; i < 50; i++) {
                sum += a[this.thread.x * 50 + i] * b[i];
            }
            return sum + v[this.thread.x];
        }).setOutput([this.V * 3]);
        this.shapemm = shapeMatMul(this.shapedirs.data, betas.data, this.vertices.data);
        //alert((Date.now()-st)+'ms');
        const poseMatMul = this.gpu.createKernel(function (a, b, v) {
            let sum = 0;
            let mo = this.thread.x % 3;
            for (let i = 0; i < 36; i++) {
                sum += a[i] * b[(((this.thread.x - mo) / 3) * 36 + i) * 3 + mo];
            }
            return sum + v[this.thread.x];
        }).setOutput([this.V * 3]);
        this.vertsmm = poseMatMul(pose_feature.data, this.posedirs.data, this.shapemm);

        const lbswMatMul = this.gpu.createKernel(function (a, b, J) {
            let sum = 0;
            let mo = this.thread.x % 16;
            for (let i = 0; i < J; i++) {
                sum += a[i] * b[(((this.thread.x - mo) / 16) * J + i) * 16 + mo];
            }
            return sum;
        }).setOutput([this.V * 16]);
        this.lbswmm = lbswMatMul(this.lbs_weights.data, transform.data, this.J);
        //var st=Date.now();
        const homoMatMul = this.gpu.createKernel(function (a, b) {
            let sum = 0;
            let mo = this.thread.x % 4;
            for (let i = 0; i < 3; i++) {
                sum += a[this.thread.x * 4 + i] * b[((this.thread.x - mo) / 4) * 3 + i];
            }
            sum += a[this.thread.x * 4 + 3];
            return sum;
        }).setOutput([this.V * 4]);
        this.v_homo = homoMatMul(this.lbswmm, this.vertsmm);
        //alert((Date.now()-st)+'ms');
    }

    async render(flame, betas, pose_params) {
        var poses, transform;
        async function forward(betas, pose_params) {
            var retVal = await flame.lbs(betas, pose_params);
            poses = retVal.ret1;
            transform = retVal.ret2;
        }
        await forward(betas, pose_params);
        this.init(betas, transform, poses);

        //console.log(this.v_homo);
        // Get A WebGL context
        /** @type {HTMLCanvasElement} */
        var gl = this.gl;
        if (!gl) {
            return;
        }
        // First let's make some variables
        // to hold the translation,
        var translation = [0, 0, -360];
        var rotation = [0, 0, 0];
        var scale = [1, 1, 1];
        var fieldOfViewRadians = degToRad(45);
        var F = this.faces.shape[0];
        var F_num = F*3;
        var J=transform.shape[0];
        webglLessonsUI.setupSlider("#fieldOfView", { value: radToDeg(fieldOfViewRadians), slide: updateFieldOfView, min: 1, max: 179 });
        webglLessonsUI.setupSlider("#x", { value: translation[0], slide: updatePosition(0), min: -200, max: 200 });
        webglLessonsUI.setupSlider("#y", { value: translation[1], slide: updatePosition(1), min: -200, max: 200 });
        webglLessonsUI.setupSlider("#z", { value: translation[2], slide: updatePosition(2), min: -1000, max: 0 });
        webglLessonsUI.setupSlider("#angleX", { value: radToDeg(rotation[0]), slide: updateRotation(0), max: 360 });
        webglLessonsUI.setupSlider("#angleY", { value: radToDeg(rotation[1]), slide: updateRotation(1), max: 360 });
        webglLessonsUI.setupSlider("#angleZ", { value: radToDeg(rotation[2]), slide: updateRotation(2), max: 360 });
        webglLessonsUI.setupSlider("#F_num", { value: F_num, slide: updateF(), max: F_num });
        webglLessonsUI.setupSlider("#exp1", {
            value: betas.data[0], slide: async function (e, ui) { 
                betas.set(0, ui.value); 
                await forward(betas, pose_params); 
                setBetas(); setPoses(); setTransform();
                drawScene(); }, step: 0.001, min: -2, max: 2, precision: 3
        });
        webglLessonsUI.setupSlider("#pose1", {
            value: pose_params.data[6], slide: async function (e, ui) { 
                pose_params.set(6, ui.value);
                await forward(betas, pose_params); 
                setBetas(); setPoses(); setTransform();
                drawScene(); }, step: 0.001, min: -0.5, max: 0.5, precision: 3
        });
        // Use our boilerplate utils to compile the shaders and link into a program
        var program = webglUtils.createProgramFromSources(gl,
            [this.vertexShaderSource, this.fragmentShaderSource]);

        // look up where the vertex data needs to go.
        var positionAttributeLocation = gl.getAttribLocation(program, "a_position");
        var indexAttributeLocation = gl.getAttribLocation(program, "a_index");
        var betasLocation = gl.getUniformLocation(program, "betasTex");
        var shapedirsLocation = gl.getUniformLocation(program, "shapedirsTex");
        var posesLocation = gl.getUniformLocation(program, "posesTex");
        var posedirsLocation = gl.getUniformLocation(program, "posedirsTex");
        var transformLocation=gl.getUniformLocation(program,"transformTex");
        var lbsweightLocation=gl.getUniformLocation(program,"lbsweightTex");
        // look up uniform locations
        var matrixLocation = gl.getUniformLocation(program, "u_matrix");

        // Create a buffer
        var positionBuffer = gl.createBuffer();

        // Create a vertex array object (attribute state)
        var vao = gl.createVertexArray();

        // and make it the one we're currently working with
        gl.bindVertexArray(vao);

        // Turn on the attribute
        gl.enableVertexAttribArray(positionAttributeLocation);

        // Bind it to ARRAY_BUFFER (think of it as ARRAY_BUFFER = positionBuffer)
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        // Set Geometry.
        //console.log(this.vertices.data);
        gl.bufferData(
            gl.ARRAY_BUFFER,
            this.vertices.data,
            gl.STATIC_DRAW);

        // Tell the attribute how to get data out of positionBuffer (ARRAY_BUFFER)
        var size = 3;          // 3 components per iteration
        var type = gl.FLOAT;   // the data is 32bit floats
        var normalize = false; // don't normalize the data
        var stride = 0;        // 0 = move forward size * sizeof(type) each iteration to get the next position
        var offset = 0;        // start at the beginning of the buffer
        gl.vertexAttribPointer(
            positionAttributeLocation, size, type, normalize, stride, offset);

        var vindexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vindexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.index, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(indexAttributeLocation);
        var size = 1;
        var type = gl.INT;
        var normalize = false;
        var stride = 0;
        var offset = 0;
        gl.vertexAttribPointer(indexAttributeLocation, size, type, normalize, stride, offset);

        var indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        gl.bufferData(
            gl.ELEMENT_ARRAY_BUFFER,
            this.faces.data,
            gl.STATIC_DRAW
        )

        function setBetas() {
            var betasTexture = gl.createTexture();
            gl.activeTexture(gl.TEXTURE0 + 0);
            gl.bindTexture(gl.TEXTURE_2D, betasTexture);
            var level = 0;
            var internalFormat = gl.R32F;
            var width = 50;
            var height = 1;
            var border = 0;
            var format = gl.RED;
            var type = gl.FLOAT;
            //console.log(betas);
            //gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
            gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, width, height, border, format, type, betas.data);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        }
        setBetas();

        var shapedirsTexture = gl.createTexture();
        gl.activeTexture(gl.TEXTURE0 + 1);
        gl.bindTexture(gl.TEXTURE_2D, shapedirsTexture);
        var level = 0;
        var internalFormat = gl.RGB32F;
        var height = 7283;// 80113/11
        var width = 50 * 11;
        var border = 0;
        var format = gl.RGB;
        var type = gl.FLOAT;
        gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, width, height, border, format, type, this.shapedirs.data);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        function setPoses() {
            var posesTexture = gl.createTexture();
            gl.activeTexture(gl.TEXTURE0 + 2);
            gl.bindTexture(gl.TEXTURE_2D, posesTexture);
            var level = 0;
            var internalFormat = gl.R32F;
            var width = 36;
            var height = 1;
            var border = 0;
            var format = gl.RED;
            var type = gl.FLOAT;
            //poses.set(35,0.5);
            gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, width, height, border, format, type, poses.data);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        }
        setPoses();

        var posedirsTexture = gl.createTexture();
        gl.activeTexture(gl.TEXTURE0 + 3);
        gl.bindTexture(gl.TEXTURE_2D, posedirsTexture);
        var level = 0;
        var internalFormat = gl.RGB32F;
        var height = 7283;// 80113/11
        var width = 36 * 11;
        var border = 0;
        var format = gl.RGB;
        var type = gl.FLOAT;
        gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, width, height, border, format, type, this.posedirs.data);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        function setTransform(){
            var transformTexture = gl.createTexture();
            gl.activeTexture(gl.TEXTURE0 + 4);
            gl.bindTexture(gl.TEXTURE_2D, transformTexture);
            var level = 0;
            var internalFormat = gl.R32F;
            var width = 16;
            var height = 5;
            var border = 0;
            var format = gl.RED;
            var type = gl.FLOAT;
            //transform.set(4,2,3,0.5)
            gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, width, height, border, format, type, transform.data);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        }
        setTransform();

        var lbsweightTexture = gl.createTexture();
        gl.activeTexture(gl.TEXTURE0 + 5);
        gl.bindTexture(gl.TEXTURE_2D, lbsweightTexture);
        var level = 0;
        var internalFormat = gl.R32F;
        var height = 7283;// 80113/11
        var width = J * 11;
        var border = 0;
        var format = gl.RED;
        var type = gl.FLOAT;
        //console.log(this.lbs_weights,J);
        //this.lbs_weights.set(1,1,1);
        gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, width, height, border, format, type, this.lbs_weights.data);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        function radToDeg(r) {
            return r * 180 / Math.PI;
        }

        function degToRad(d) {
            return d * Math.PI / 180;
        }



        // console.log(show(this.vertices.pick(3,null)));
        // console.log(show(this.vertices.pick(22066,null)));
        // console.log(show(this.vertices.pick(22065,null)));
        // //7, 20090, 20089
        // console.log(show(this.vertices.pick(7,null)));
        // console.log(show(this.vertices.pick(22090,null)));
        // console.log(show(this.vertices.pick(22089,null)));
        console.log("start draw");
        //var st=console.time("at");
        //for(var i=0;i<10000;i++){
        drawScene();
        //}
        //console.log(console.timeEnd("at"));
        // Setup a ui.


        function updateFieldOfView(event, ui) {
            fieldOfViewRadians = degToRad(ui.value);
            drawScene();
        }

        function updatePosition(index) {
            return function (event, ui) {
                translation[index] = ui.value;
                drawScene();
            };
        }

        function updateRotation(index) {
            return function (event, ui) {
                var angleInDegrees = ui.value;
                var angleInRadians = degToRad(angleInDegrees);
                rotation[index] = angleInRadians;
                drawScene();
            };
        }

        function updateScale(index) {
            return function (event, ui) {
                scale[index] = ui.value;
                drawScene();
            };
        }
        function updateF() {
            return function (event, ui) {
                F_num = ui.value;
                drawScene();
            };
        }
        // Draw the scene.
        function drawScene() {

            webglUtils.resizeCanvasToDisplaySize(gl.canvas);

            // Tell WebGL how to convert from clip space to pixels
            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

            // Clear the canvas
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

            // turn on depth testing
            gl.enable(gl.DEPTH_TEST);

            // tell webgl to cull faces
            gl.enable(gl.CULL_FACE);

            // Tell it to use our program (pair of shaders)
            gl.useProgram(program);

            // Bind the attribute/buffer set we want.
            gl.bindVertexArray(vao);

            // Compute the matrix
            var aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
            var zNear = 1;
            var zFar = 2000;
            //var matrix=m4.scaling(1,1,1);
            var matrix = m4.perspective(fieldOfViewRadians, aspect, zNear, zFar);
            // //console.log(matrix);
            // //var matrix = m4.projection(gl.canvas.clientWidth, gl.canvas.clientHeight, 400);
            matrix = m4.translate(matrix, translation[0], translation[1], translation[2]);
            matrix = m4.xRotate(matrix, rotation[0]);
            matrix = m4.yRotate(matrix, rotation[1]);
            matrix = m4.zRotate(matrix, rotation[2]);
            // matrix = m4.scale(matrix, scale[0], scale[1], scale[2]);

            // Set the matrix.
            gl.uniformMatrix4fv(matrixLocation, false, matrix);
            gl.uniform1i(betasLocation, 0);
            gl.uniform1i(shapedirsLocation, 1);
            gl.uniform1i(posesLocation, 2);
            gl.uniform1i(posedirsLocation, 3);
            gl.uniform1i(transformLocation, 4);
            gl.uniform1i(lbsweightLocation, 5);
            // Draw the geometry.
            var primitiveType = gl.TRIANGLES;
            var offset = 0;
            var count = F_num;
            gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_INT, offset);
            //gl.drawArrays(gl.TRIANGLES,offset,count);

        }
    }

    // Fill the current ARRAY_BUFFER buffer
    // with the values that define a letter 'F'.
    // setGeometry(gl) {

    // }


}


var m4 = {

    perspective: function (fieldOfViewInRadians, aspect, near, far) {
        var f = Math.tan(Math.PI * 0.5 - 0.5 * fieldOfViewInRadians);
        var rangeInv = 1.0 / (near - far);

        return [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (near + far) * rangeInv, -1,
            0, 0, near * far * rangeInv * 2, 0,
        ];
    },

    projection: function (width, height, depth) {
        // Note: This matrix flips the Y axis so 0 is at the top.
        return [
            2 / width, 0, 0, 0,
            0, -2 / height, 0, 0,
            0, 0, 2 / depth, 0,
            -1, 1, 0, 1,
        ];
    },

    multiply: function (a, b) {
        var a00 = a[0 * 4 + 0];
        var a01 = a[0 * 4 + 1];
        var a02 = a[0 * 4 + 2];
        var a03 = a[0 * 4 + 3];
        var a10 = a[1 * 4 + 0];
        var a11 = a[1 * 4 + 1];
        var a12 = a[1 * 4 + 2];
        var a13 = a[1 * 4 + 3];
        var a20 = a[2 * 4 + 0];
        var a21 = a[2 * 4 + 1];
        var a22 = a[2 * 4 + 2];
        var a23 = a[2 * 4 + 3];
        var a30 = a[3 * 4 + 0];
        var a31 = a[3 * 4 + 1];
        var a32 = a[3 * 4 + 2];
        var a33 = a[3 * 4 + 3];
        var b00 = b[0 * 4 + 0];
        var b01 = b[0 * 4 + 1];
        var b02 = b[0 * 4 + 2];
        var b03 = b[0 * 4 + 3];
        var b10 = b[1 * 4 + 0];
        var b11 = b[1 * 4 + 1];
        var b12 = b[1 * 4 + 2];
        var b13 = b[1 * 4 + 3];
        var b20 = b[2 * 4 + 0];
        var b21 = b[2 * 4 + 1];
        var b22 = b[2 * 4 + 2];
        var b23 = b[2 * 4 + 3];
        var b30 = b[3 * 4 + 0];
        var b31 = b[3 * 4 + 1];
        var b32 = b[3 * 4 + 2];
        var b33 = b[3 * 4 + 3];
        return [
            b00 * a00 + b01 * a10 + b02 * a20 + b03 * a30,
            b00 * a01 + b01 * a11 + b02 * a21 + b03 * a31,
            b00 * a02 + b01 * a12 + b02 * a22 + b03 * a32,
            b00 * a03 + b01 * a13 + b02 * a23 + b03 * a33,
            b10 * a00 + b11 * a10 + b12 * a20 + b13 * a30,
            b10 * a01 + b11 * a11 + b12 * a21 + b13 * a31,
            b10 * a02 + b11 * a12 + b12 * a22 + b13 * a32,
            b10 * a03 + b11 * a13 + b12 * a23 + b13 * a33,
            b20 * a00 + b21 * a10 + b22 * a20 + b23 * a30,
            b20 * a01 + b21 * a11 + b22 * a21 + b23 * a31,
            b20 * a02 + b21 * a12 + b22 * a22 + b23 * a32,
            b20 * a03 + b21 * a13 + b22 * a23 + b23 * a33,
            b30 * a00 + b31 * a10 + b32 * a20 + b33 * a30,
            b30 * a01 + b31 * a11 + b32 * a21 + b33 * a31,
            b30 * a02 + b31 * a12 + b32 * a22 + b33 * a32,
            b30 * a03 + b31 * a13 + b32 * a23 + b33 * a33,
        ];
    },

    translation: function (tx, ty, tz) {
        return [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            tx, ty, tz, 1,
        ];
    },

    xRotation: function (angleInRadians) {
        var c = Math.cos(angleInRadians);
        var s = Math.sin(angleInRadians);

        return [
            1, 0, 0, 0,
            0, c, s, 0,
            0, -s, c, 0,
            0, 0, 0, 1,
        ];
    },

    yRotation: function (angleInRadians) {
        var c = Math.cos(angleInRadians);
        var s = Math.sin(angleInRadians);

        return [
            c, 0, -s, 0,
            0, 1, 0, 0,
            s, 0, c, 0,
            0, 0, 0, 1,
        ];
    },

    zRotation: function (angleInRadians) {
        var c = Math.cos(angleInRadians);
        var s = Math.sin(angleInRadians);

        return [
            c, s, 0, 0,
            -s, c, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ];
    },

    scaling: function (sx, sy, sz) {
        return [
            sx, 0, 0, 0,
            0, sy, 0, 0,
            0, 0, sz, 0,
            0, 0, 0, 1,
        ];
    },

    translate: function (m, tx, ty, tz) {
        return m4.multiply(m, m4.translation(tx, ty, tz));
    },

    xRotate: function (m, angleInRadians) {
        return m4.multiply(m, m4.xRotation(angleInRadians));
    },

    yRotate: function (m, angleInRadians) {
        return m4.multiply(m, m4.yRotation(angleInRadians));
    },

    zRotate: function (m, angleInRadians) {
        return m4.multiply(m, m4.zRotation(angleInRadians));
    },

    scale: function (m, sx, sy, sz) {
        return m4.multiply(m, m4.scaling(sx, sy, sz));
    },

};

//render();
export default Renderer;