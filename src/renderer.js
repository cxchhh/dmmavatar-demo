"use strict";
import ndarray from "ndarray";
import cwise from "cwise";
import zeros from "zeros";
import show from "ndarray-show";
import { getFragmentShaderSource, getVertexShaderSource } from "./shaders.js";

class Renderer {
    constructor(idx, vertices, faces, lbs_weights, posedirs, shapedirs, uvs, normals) {
        this.idx = idx;
        this.vertices = vertices;
        this.faces = faces;
        this.lbs_weights = lbs_weights;
        this.posedirs = posedirs;
        this.shapedirs = shapedirs;
        this.uvs = uvs;
        this.normals = normals;
        this.V = vertices.shape[0];
        this.F = faces.shape[0];
        this.J = lbs_weights.shape[1];
        this.T = 6;
        this.F_num = 1.0;
        this.canvas = document.querySelector("#canvas");
        this.gl = this.canvas.getContext("webgl2");
        this.index = new Int32Array(this.V);
        for (var i = 0; i < this.V; i++) {
            this.index[i] = i;
        }
        this.vertexShaderSource = getVertexShaderSource(this.J);
        this.fragmentShaderSource = getFragmentShaderSource();
    }
    render(betas,poses,transform) {
        this.setDynamics(betas,poses,transform);
        var gl = this.gl;
        if (this.idx == 0) {
            webglUtils.resizeCanvasToDisplaySize(gl.canvas);
            // Tell WebGL how to convert from clip space to pixels
            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            // Clear the canvas
            gl.clearColor(0, 0, 0, 1);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            // turn on depth testing
            gl.enable(gl.DEPTH_TEST);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
            // tell webgl to cull faces
            gl.enable(gl.CULL_FACE);
            gl.depthMask(false);
        }
        // Tell it to use our program (pair of shaders)
        gl.useProgram(this.program);
        // Bind the attribute/buffer set we want.
        gl.bindVertexArray(this.vao);
        // Set the matrix.
        gl.uniformMatrix4fv(this.matrixLocation, false, this.matrix);
        gl.uniformMatrix4fv(this.normalMatrixLocation, false, this.normal_matrix);
        gl.uniformMatrix4fv(this.viewMatrixLocation, false, this.view_matrix);
        gl.uniform1i(this.betasLocation, 0);
        gl.uniform1i(this.shapedirsLocation, 1);
        gl.uniform1i(this.posesLocation, 2);
        gl.uniform1i(this.posedirsLocation, 3);
        gl.uniform1i(this.transformLocation, 4);
        gl.uniform1i(this.lbsweightLocation, 5);
        // Draw the geometry.
        var primitiveType = gl.TRIANGLES;
        var offset = 0;
        var count = Math.floor(this.F * 3*this.F_num);
        gl.drawElements(primitiveType, count, gl.UNSIGNED_INT, offset);
        //gl.drawArrays(gl.TRIANGLES,offset,count);
    }

    setStatics() {
        var gl = this.gl;

        // Use our boilerplate utils to compile the shaders and link into a program
        this.program = webglUtils.createProgramFromSources(gl, [
            this.vertexShaderSource,
            this.fragmentShaderSource,
        ]);
        var program = this.program;
        // look up where the vertex data needs to go.
        var positionAttributeLocation = gl.getAttribLocation(program, "a_position");
        var indexAttributeLocation = gl.getAttribLocation(program, "a_index");
        this.betasLocation = gl.getUniformLocation(program, "betasTex");
        this.shapedirsLocation = gl.getUniformLocation(program, "shapedirsTex");
        this.posesLocation = gl.getUniformLocation(program, "posesTex");
        this.posedirsLocation = gl.getUniformLocation(program, "posedirsTex");
        this.transformLocation = gl.getUniformLocation(program, "transformTex");
        this.lbsweightLocation = gl.getUniformLocation(program, "lbsweightTex");
        var uvsAttributeLocation = gl.getAttribLocation(program, "a_uv");
        var normalsAttributeLocation = gl.getAttribLocation(program, "a_normal");
        // look up uniform locations
        this.matrixLocation = gl.getUniformLocation(program, "u_matrix");
        this.normalMatrixLocation = gl.getUniformLocation(program, "u_normal_matrix");
        this.viewMatrixLocation = gl.getUniformLocation(program, "u_view_matrix");

        // Create a buffer
        var positionBuffer = gl.createBuffer();

        // Create a vertex array object (attribute state)
        this.vao = gl.createVertexArray();

        // and make it the one we're currently working with
        gl.bindVertexArray(this.vao);

        // Turn on the attribute
        gl.enableVertexAttribArray(positionAttributeLocation);

        // Bind it to ARRAY_BUFFER (think of it as ARRAY_BUFFER = positionBuffer)
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        // Set Geometry.
        //console.log(this.vertices.data);
        gl.bufferData(gl.ARRAY_BUFFER, this.vertices.data, gl.STATIC_DRAW);

        // Tell the attribute how to get data out of positionBuffer (ARRAY_BUFFER)
        var size = 3; // 3 components per iteration
        var type = gl.FLOAT; // the data is 32bit floats
        var normalize = false; // don't normalize the data
        var stride = 0; // 0 = move forward size * sizeof(type) each iteration to get the next position
        var offset = 0; // start at the beginning of the buffer
        gl.vertexAttribPointer(positionAttributeLocation, size, type, normalize, stride, offset);

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

        var uvsBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, uvsBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.uvs.data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(uvsAttributeLocation);
        var size = 2;
        var type = gl.FLOAT;
        var normalize = false;
        var stride = 0;
        var offset = 0;
        //console.log(this.uvs);
        gl.vertexAttribPointer(uvsAttributeLocation, size, type, normalize, stride, offset);

        var normalsBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, normalsBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.normals.data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(normalsAttributeLocation);
        var size = 3;
        var type = gl.FLOAT;
        var normalize = false;
        var stride = 0;
        var offset = 0;
        gl.vertexAttribPointer(normalsAttributeLocation, size, type, normalize, stride, offset);

        var indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.faces.data, gl.STATIC_DRAW);

        var height = Math.ceil(this.V / 40);
        var width = 50 * 40;
        this.betasTexture = gl.createTexture();
        this.shapedirsTexture = gl.createTexture();
        this.shapedata = new Float32Array(height * width * 3);
        this.shapedata.set(this.shapedirs.data, 0);

        var height = Math.ceil(this.V / 40);
        var width = 36 * 40;
        this.posesTexture = gl.createTexture();
        this.posedirsTexture = gl.createTexture();
        this.posedata = new Float32Array(height * width * 3);
        this.posedata.set(this.posedirs.data, 0);

        var height = Math.ceil(this.V / 40);
        var width = this.J * 40;
        this.transformTexture = gl.createTexture();
        this.lbsweightTexture = gl.createTexture();
        this.lbswdata = new Float32Array(height * width);
        this.lbswdata.set(this.lbs_weights.data, 0);
    }
    setDynamics(betas,poses,transform){
        var gl = this.gl;
        gl.activeTexture(gl.TEXTURE0 + 0);
        gl.bindTexture(gl.TEXTURE_2D, this.betasTexture);
        var level = 0;
        var internalFormat = gl.R32F;
        var width = 50;
        var height = 1;
        var border = 0;
        var format = gl.RED;
        var type = gl.FLOAT;
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            width,
            height,
            border,
            format,
            type,
            betas.data
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.activeTexture(gl.TEXTURE0 + 1);
        gl.bindTexture(gl.TEXTURE_2D, this.shapedirsTexture);
        var level = 0;
        var internalFormat = gl.RGB32F;
        var height = Math.ceil(this.V / 40);
        var width = 50 * 40;
        var border = 0;
        var format = gl.RGB;
        var type = gl.FLOAT;
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            width,
            height,
            border,
            format,
            type,
            this.shapedata
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.activeTexture(gl.TEXTURE0 + 2);
        gl.bindTexture(gl.TEXTURE_2D, this.posesTexture);
        var level = 0;
        var internalFormat = gl.R32F;
        var width = 36;
        var height = 1;
        var border = 0;
        var format = gl.RED;
        var type = gl.FLOAT;
        //poses.set(35,0.5);
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            width,
            height,
            border,
            format,
            type,
            poses.data
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.activeTexture(gl.TEXTURE0 + 3);
        gl.bindTexture(gl.TEXTURE_2D, this.posedirsTexture);
        var level = 0;
        var internalFormat = gl.RGB32F;
        var height = Math.ceil(this.V / 40);
        var width = 36 * 40;
        var border = 0;
        var format = gl.RGB;
        var type = gl.FLOAT;
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            width,
            height,
            border,
            format,
            type,
            this.posedata
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.activeTexture(gl.TEXTURE0 + 4);
        gl.bindTexture(gl.TEXTURE_2D, this.transformTexture);
        var level = 0;
        var internalFormat = gl.R32F;
        var width = 16;
        var height = this.J;
        var border = 0;
        var format = gl.RED;
        var type = gl.FLOAT;
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            width,
            height,
            border,
            format,
            type,
            transform.data
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.activeTexture(gl.TEXTURE0 + 5);
        gl.bindTexture(gl.TEXTURE_2D, this.lbsweightTexture);
        var level = 0;
        var internalFormat = gl.R32F;
        var height = Math.ceil(this.V / 40);
        var width = this.J * 40;
        var border = 0;
        var format = gl.RED;
        var type = gl.FLOAT;
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            width,
            height,
            border,
            format,
            type,
            this.lbswdata
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
}
export default Renderer;
