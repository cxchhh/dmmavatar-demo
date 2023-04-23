"use strict";
import ndarray from "ndarray";
import cwise from "cwise";
import zeros from "zeros";
import { GPU } from "gpu.js";
import show from "ndarray-show";

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
        this.vertexShaderSource = `#version 300 es
        // an attribute is an input (in) to a vertex shader.
        // It will receive data from a buffer
        in vec4 a_position;
        uniform vec4 a_color;
        // A matrix to transform the positions by
        uniform mat4 u_matrix;
        // a varying the color to the fragment shader
        out vec4 v_color;
        float random (vec2 st) {
            return fract(sin(dot(st.xy,vec2(12.9898,78.233)))*43758.5453123);
        }
        // all shaders have a main function
        void main() {
            
            // Multiply the position by the matrix.
            vec4 apos=vec4(a_position.xyz*200.0f,1);
            gl_Position = u_matrix * (apos);
        
            // Pass the color to the fragment shader.
            v_color = vec4(random(a_position.xy),random(a_position.yz),random(a_position.zx),1);
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

    render(betas, transform, pose_feature) {

        this.init(betas, transform, pose_feature);

        //console.log(this.v_homo);
        // Get A WebGL context
        /** @type {HTMLCanvasElement} */
        var gl = this.gl;
        if (!gl) {
            return;
        }

        // Use our boilerplate utils to compile the shaders and link into a program
        var program = webglUtils.createProgramFromSources(gl,
            [this.vertexShaderSource, this.fragmentShaderSource]);

        // look up where the vertex data needs to go.
        var positionAttributeLocation = gl.getAttribLocation(program, "a_position");
        //var colorAttributeLocation = gl.getAttribLocation(program, "a_color");
        var colorLocation=gl.getUniformLocation(program,"a_color");
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
            this.v_homo,
            gl.STATIC_DRAW);

        // Tell the attribute how to get data out of positionBuffer (ARRAY_BUFFER)
        var size = 4;          // 3 components per iteration
        var type = gl.FLOAT;   // the data is 32bit floats
        var normalize = false; // don't normalize the data
        var stride = 0;        // 0 = move forward size * sizeof(type) each iteration to get the next position
        var offset = 0;        // start at the beginning of the buffer
        gl.vertexAttribPointer(
            positionAttributeLocation, size, type, normalize, stride, offset);

        // // create the color buffer, make it the current ARRAY_BUFFER
        // // and copy in the color values
        // var colorBuffer = gl.createBuffer();
        // gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        // this.setColors(gl);

        // // Turn on the attribute
        // gl.enableVertexAttribArray(colorAttributeLocation);

        // // Tell the attribute how to get data out of colorBuffer (ARRAY_BUFFER)
        // var size = 3;          // 3 components per iteration
        // var type = gl.UNSIGNED_BYTE;   // the data is 8bit unsigned bytes
        // var normalize = true;  // convert from 0-255 to 0.0-1.0
        // var stride = 0;        // 0 = move forward size * sizeof(type) each iteration to get the next color
        // var offset = 0;        // start at the beginning of the buffer
        // gl.vertexAttribPointer(
        //     colorAttributeLocation, size, type, normalize, stride, offset);
        
        var indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indexBuffer);
        //console.log(this.faces.data);

        gl.bufferData(
            gl.ELEMENT_ARRAY_BUFFER,
            this.faces.data,
            gl.STATIC_DRAW
        )

        function radToDeg(r) {
            return r * 180 / Math.PI;
        }

        function degToRad(d) {
            return d * Math.PI / 180;
        }

        // First let's make some variables
        // to hold the translation,
        var translation = [0, 0, -360];
        var rotation = [0, 0, 0];
        var scale = [1, 1, 1];
        var fieldOfViewRadians = degToRad(60);
        var F=this.faces.shape[0]*3;
        var F_num=F;

        // console.log(show(this.vertices.pick(3,null)));
        // console.log(show(this.vertices.pick(22066,null)));
        // console.log(show(this.vertices.pick(22065,null)));
        // //7, 20090, 20089
        // console.log(show(this.vertices.pick(7,null)));
        // console.log(show(this.vertices.pick(22090,null)));
        // console.log(show(this.vertices.pick(22089,null)));
        drawScene();

        // Setup a ui.
        webglLessonsUI.setupSlider("#fieldOfView", { value: radToDeg(fieldOfViewRadians), slide: updateFieldOfView, min: 1, max: 179 });
        webglLessonsUI.setupSlider("#x", { value: translation[0], slide: updatePosition(0), min: -200, max: 200 });
        webglLessonsUI.setupSlider("#y", { value: translation[1], slide: updatePosition(1), min: -200, max: 200 });
        webglLessonsUI.setupSlider("#z", { value: translation[2], slide: updatePosition(2), min: -1000, max: 0 });
        webglLessonsUI.setupSlider("#angleX", { value: radToDeg(rotation[0]), slide: updateRotation(0), max: 360 });
        webglLessonsUI.setupSlider("#angleY", { value: radToDeg(rotation[1]), slide: updateRotation(1), max: 360 });
        webglLessonsUI.setupSlider("#angleZ", { value: radToDeg(rotation[2]), slide: updateRotation(2), max: 360 });
        webglLessonsUI.setupSlider("#F_num", { value: F_num, slide: updateF(), max: F });

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
        function updateF(){
            return function (event, ui) {
                F_num=ui.value;
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
            gl.uniform4f(colorLocation, Math.random(), Math.random(), Math.random(), 1);
            // Draw the geometry.
            var primitiveType = gl.TRIANGLES;
            var offset = 0;
            var count = F_num;
            gl.drawElements(gl.TRIANGLES, count,gl.UNSIGNED_INT,offset);
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