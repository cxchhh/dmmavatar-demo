"use strict";
import ndarray from "ndarray";
import cwise from "cwise";
import zeros from "zeros";
import { GPU } from "gpu.js";
import show from "ndarray-show";
import FLAME from "./flame.js";

class Renderer {
  constructor(
    flame,
    vertices,
    faces,
    lbs_weights,
    posedirs,
    shapedirs,
    betas,
    pose_params,
    uvs,
    normals
  ) {
    this.flame = flame;
    this.vertices = vertices;
    this.faces = faces;
    this.lbs_weights = lbs_weights;
    this.posedirs = posedirs;
    this.shapedirs = shapedirs;
    this.betas = betas;
    this.pose_params = pose_params;
    this.uvs = uvs;
    this.normals = normals;
    this.V = vertices.shape[0];
    this.J = lbs_weights.shape[1];
    this.gpu = new GPU();
    this.canvas = document.querySelector("#canvas");
    this.gl = this.canvas.getContext("webgl2");
    // Get A WebGL context
    /** @type {HTMLCanvasElement} */
    var gl = this.gl;
    if (!gl) {
      alert("WebGL isn't available on your broswer!");
      return;
    }
    this.index = new Int32Array(this.V);
    for (var i = 0; i < this.V; i++) {
      this.index[i] = i;
    }
    //console.log(this.index);
    this.vertexShaderSource = `#version 300 es
        // an attribute is an input (in) to a vertex shader.
        // It will receive data from a buffer
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
            j=(i%11)*${this.J}+j;
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
                for(int j=0;j<${this.J};j++){
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
            // Pass the color to the fragment shader.
            //v_color = vec4(fidx/${this.V}.0,0,0,1);
            v_uv=a_uv;
            v_normal=normalize(mat3(u_normal_matrix)*a_normal);
        }`;
    this.fragmentShaderSource = `#version 300 es
            precision highp float;

            // the varied color passed from the vertex shader
            //in vec4 v_color;
            in vec2 v_uv;
            in vec3 v_normal;
            in vec3 v_viewdir;
            // we need to declare an output for the fragment shader
            out vec4 outColor;
            void main() {
                outColor = vec4((v_normal+1.0)/2.0,1);
        }`;
    this.uiInit();
  }

  uiInit() {
    // First let's make some variables
    // to hold the translation,
    this.translation = [0, 0, -360];
    this.rotation = [0, 0, 0];
    this.fieldOfViewRadians = this.degToRad(45);
    var F = this.faces.shape[0];
    this.F_num = F * 3;
    var th = this;
    webglLessonsUI.setupSlider("#fieldOfView", {
      value: this.radToDeg(this.fieldOfViewRadians),
      slide: updateFieldOfView,
      min: 1,
      max: 179,
    });
    //webglLessonsUI.setupSlider("#x", { value: this.translation[0], slide: updatePosition(0), min: -200, max: 200 });
    //webglLessonsUI.setupSlider("#y", { value: this.translation[1], slide: updatePosition(1), min: -200, max: 200 });
    //webglLessonsUI.setupSlider("#z", { value: this.translation[2], slide: updatePosition(2), min: -1000, max: 0 });
    webglLessonsUI.setupSlider("#angleX", {
      value: this.radToDeg(this.rotation[0]),
      slide: updateRotation(0),
      max: 360,
    });
    webglLessonsUI.setupSlider("#angleY", {
      value: this.radToDeg(this.rotation[1]),
      slide: updateRotation(1),
      max: 360,
    });
    webglLessonsUI.setupSlider("#angleZ", {
      value: this.radToDeg(this.rotation[2]),
      slide: updateRotation(2),
      max: 360,
    });
    webglLessonsUI.setupSlider("#F_num", {
      value: this.F_num,
      slide: updateF(),
      max: this.F_num,
    });
    for (var i = 0; i < 50; i++)
      webglLessonsUI.setupSlider("#exp" + (i + 1), {
        value: this.betas.data[i],
        slide: updateBetas(i),
        step: 0.001,
        min: -2,
        max: 2,
        precision: 3,
      });

    for (var i = 0; i < 15; i++)
      webglLessonsUI.setupSlider("#pose" + (i + 1), {
        value: this.pose_params.data[i],
        slide: updatePoses(i),
        step: 0.001,
        min: -0.5,
        max: 0.5,
        precision: 3,
      });
    function updateBetas(i) {
      return async function (e, ui) {
        th.betas.set(i, ui.value);
        await th.forward(th.betas, th.pose_params);
        th.setBetas();
        th.setPoses();
        th.setTransform();
        ////th.drawScene();
      };
    }
    function updatePoses(i) {
      return async function (e, ui) {
        th.pose_params.set(i, ui.value);
        await th.forward(th.betas, th.pose_params);
        th.setBetas();
        th.setPoses();
        th.setTransform();
        //th.drawScene();
      };
    }
    function updateFieldOfView(event, ui) {
      th.fieldOfViewRadians = th.degToRad(ui.value);
      ////th.drawScene();
    }
    function updatePosition(index) {
      return function (event, ui) {
        th.translation[index] = ui.value;
        //th.drawScene();
      };
    }
    function updateRotation(index) {
      return function (event, ui) {
        var angleInDegrees = ui.value;
        var angleInRadians = th.degToRad(angleInDegrees);
        th.rotation[index] = angleInRadians;
        //th.drawScene();
      };
    }
    function updateF() {
      return function (event, ui) {
        th.F_num = ui.value;
        //th.drawScene();
      };
    }
    this.canvas.addEventListener("contextmenu", function (e) {
      e.preventDefault();
    });
    var lastx = 0,
      lasty = 0,
      incanvas = true,
      down = false,
      basex,
      basey,
      btn = 0;
    this.canvas.onmouseenter = function (e) {
      incanvas = true;
      unScroll();
    };
    this.canvas.onmouseleave = function (e) {
      incanvas = false;
      down = false;
      removeUnScroll();
    };
    this.canvas.onmousedown = function (e) {
      if (!incanvas) return;
      down = true;
      if (e.button == 0) {
        btn = 0;
        lastx = e.clientX;
        lasty = e.clientY;
        basex = th.translation[0];
        basey = th.translation[1];
      } else if (e.button == 2) {
        btn = 2;
      }
      //console.log(e.button);
    };
    this.canvas.onmousemove = function (e) {
      if (!down) return;
      if (btn == 0) {
        th.translation[0] = basex + e.clientX - lastx;
        th.translation[1] = basey - (e.clientY - lasty);
      }

      ////th.drawScene();
    };
    this.canvas.onmouseup = function (e) {
      if (!incanvas) return;
      down = false;
    };
    this.canvas.onwheel = function (e) {
      if (!incanvas) return;
      //th.fieldOfViewRadians += e.wheelDelta / 100;
      //if (th.fieldOfViewRadians < th.degToRad(1)) th.fieldOfViewRadians = th.degToRad(1);
      //if (th.fieldOfViewRadians > th.degToRad(179)) th.fieldOfViewRadians = th.degToRad(179);
      //console.log(th.fieldOfViewRadians)
      th.translation[2] += e.wheelDelta;
      if (th.translation[2] > 1) th.translation[2] = 1;
      if (th.translation[2] < -1000) th.translation[2] = -1000;
      //th.drawScene();
    };
    //禁用滚动条
    function unScroll() {
      var top = $(document).scrollTop();
      $(document).on("scroll.unable", function (e) {
        $(document).scrollTop(top);
      });
    }
    //停止禁用滚动条
    function removeUnScroll() {
      $(document).unbind("scroll.unable");
    }

    var framwCount = 0;
    var lastTime = performance.now();
    var timeTest = document.getElementById("timeTest");
    function drawLoop() {
      th.drawScene();
      framwCount++;
      if (framwCount % 10 == 0) {
        var now = performance.now();
        var time = now - lastTime;
        lastTime = now;
        timeTest.innerHTML = "FPS:" + (1000 / (time / 10)).toFixed(2);
      }
      requestAnimationFrame(drawLoop);
    }
    requestAnimationFrame(drawLoop);
  }
  radToDeg(r) {
    return (r * 180) / Math.PI;
  }

  degToRad(d) {
    return (d * Math.PI) / 180;
  }
  async forward(betas, pose_params) {
    var retVal = await this.flame.lbs(betas, pose_params);
    this.poses = retVal.ret1;
    this.transform = retVal.ret2;
  }
  async render() {
    var betas = this.betas;
    var pose_params = this.pose_params;

    await this.forward(betas, pose_params);
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
    this.normalMatrixLocation = gl.getUniformLocation(
      program,
      "u_normal_matrix"
    );
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
    gl.vertexAttribPointer(
      positionAttributeLocation,
      size,
      type,
      normalize,
      stride,
      offset
    );

    var vindexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vindexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.index, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(indexAttributeLocation);
    var size = 1;
    var type = gl.INT;
    var normalize = false;
    var stride = 0;
    var offset = 0;
    gl.vertexAttribPointer(
      indexAttributeLocation,
      size,
      type,
      normalize,
      stride,
      offset
    );

    var uvsBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, uvsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.uvs.data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(uvsAttributeLocation);
    var size = 2;
    var type = gl.FLOAT;
    var normalize = false;
    var stride = 0;
    var offset = 0;
    console.log(this.uvs);
    gl.vertexAttribPointer(
      uvsAttributeLocation,
      size,
      type,
      normalize,
      stride,
      offset
    );

    var normalsBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normalsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.normals.data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(normalsAttributeLocation);
    var size = 3;
    var type = gl.FLOAT;
    var normalize = false;
    var stride = 0;
    var offset = 0;
    gl.vertexAttribPointer(
      normalsAttributeLocation,
      size,
      type,
      normalize,
      stride,
      offset
    );

    var indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.faces.data, gl.STATIC_DRAW);

    this.setBetas();

    var shapedirsTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + 1);
    gl.bindTexture(gl.TEXTURE_2D, shapedirsTexture);
    var level = 0;
    var internalFormat = gl.RGB32F;
    var height = 7283; // 80113/11
    var width = 50 * 11;
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
      this.shapedirs.data
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.setPoses();

    var posedirsTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + 3);
    gl.bindTexture(gl.TEXTURE_2D, posedirsTexture);
    var level = 0;
    var internalFormat = gl.RGB32F;
    var height = 7283; // 80113/11
    var width = 36 * 11;
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
      this.posedirs.data
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.setTransform();

    var lbsweightTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + 5);
    gl.bindTexture(gl.TEXTURE_2D, lbsweightTexture);
    var level = 0;
    var internalFormat = gl.R32F;
    var height = 7283; // 80113/11
    var width = this.J * 11;
    var border = 0;
    var format = gl.RED;
    var type = gl.FLOAT;
    //console.log(this.lbs_weights,J);
    //this.lbs_weights.set(1,1,1);
    gl.texImage2D(
      gl.TEXTURE_2D,
      level,
      internalFormat,
      width,
      height,
      border,
      format,
      type,
      this.lbs_weights.data
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    console.log("start rendering");

    this.drawScene(); // Draw the scene.
  }
  drawScene() {
    var gl = this.gl;
    webglUtils.resizeCanvasToDisplaySize(gl.canvas);
    // Tell WebGL how to convert from clip space to pixels
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    // Clear the canvas
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // turn on depth testing
    gl.enable(gl.DEPTH_TEST);

    // tell webgl to cull faces
    //gl.enable(gl.CULL_FACE);

    // Tell it to use our program (pair of shaders)
    gl.useProgram(this.program);

    // Bind the attribute/buffer set we want.
    gl.bindVertexArray(this.vao);

    // Compute the matrix
    var aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    var zNear = 1;
    var zFar = 2000;
    //var matrix=m4.scaling(1,1,1);
    var projection_matrix = m4.perspective(
      this.fieldOfViewRadians,
      aspect,
      zNear,
      zFar
    );
    // //console.log(matrix);
    // //var matrix = m4.projection(gl.canvas.clientWidth, gl.canvas.clientHeight, 400);

    var view_matrix = m4.scaling(1, 1, 1);
    view_matrix = m4.translate(
      view_matrix,
      this.translation[0],
      this.translation[1],
      this.translation[2]
    );
    view_matrix = m4.xRotate(view_matrix, this.rotation[0]);
    view_matrix = m4.yRotate(view_matrix, this.rotation[1]);
    view_matrix = m4.zRotate(view_matrix, this.rotation[2]);

    this.matrix = m4.multiply(projection_matrix, view_matrix);
    this.normal_matrix = m4.transpose(m4.inverse(view_matrix));
    // Set the matrix.
    gl.uniformMatrix4fv(this.matrixLocation, false, this.matrix);
    gl.uniformMatrix4fv(this.normalMatrixLocation, false, this.normal_matrix);
    gl.uniformMatrix4fv(this.viewMatrixLocation, false, view_matrix);
    gl.uniform1i(this.betasLocation, 0);
    gl.uniform1i(this.shapedirsLocation, 1);
    gl.uniform1i(this.posesLocation, 2);
    gl.uniform1i(this.posedirsLocation, 3);
    gl.uniform1i(this.transformLocation, 4);
    gl.uniform1i(this.lbsweightLocation, 5);
    // Draw the geometry.
    var primitiveType = gl.TRIANGLES;
    var offset = 0;
    var count = this.F_num;
    gl.drawElements(primitiveType, count, gl.UNSIGNED_INT, offset);
    //gl.drawArrays(gl.TRIANGLES,offset,count);
  }

  setBetas() {
    var gl = this.gl;
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
    gl.texImage2D(
      gl.TEXTURE_2D,
      level,
      internalFormat,
      width,
      height,
      border,
      format,
      type,
      this.betas.data
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }
  setPoses() {
    var gl = this.gl;
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
    gl.texImage2D(
      gl.TEXTURE_2D,
      level,
      internalFormat,
      width,
      height,
      border,
      format,
      type,
      this.poses.data
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }
  setTransform() {
    var gl = this.gl;
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
    gl.texImage2D(
      gl.TEXTURE_2D,
      level,
      internalFormat,
      width,
      height,
      border,
      format,
      type,
      this.transform.data
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }
}

var m4 = {
  perspective: function (fieldOfViewInRadians, aspect, near, far) {
    var f = Math.tan(Math.PI * 0.5 - 0.5 * fieldOfViewInRadians);
    var rangeInv = 1.0 / (near - far);

    return new Float32Array([
      f / aspect,
      0,
      0,
      0,
      0,
      f,
      0,
      0,
      0,
      0,
      (near + far) * rangeInv,
      -1,
      0,
      0,
      near * far * rangeInv * 2,
      0,
    ]);
  },

  projection: function (width, height, depth) {
    // Note: This matrix flips the Y axis so 0 is at the top.
    return new Float32Array([
      2 / width,
      0,
      0,
      0,
      0,
      -2 / height,
      0,
      0,
      0,
      0,
      2 / depth,
      0,
      -1,
      1,
      0,
      1,
    ]);
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
    return new Float32Array([
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
    ]);
  },

  translation: function (tx, ty, tz) {
    return new Float32Array([
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      tx,
      ty,
      tz,
      1,
    ]);
  },

  xRotation: function (angleInRadians) {
    var c = Math.cos(angleInRadians);
    var s = Math.sin(angleInRadians);

    return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
  },

  yRotation: function (angleInRadians) {
    var c = Math.cos(angleInRadians);
    var s = Math.sin(angleInRadians);

    return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
  },

  zRotation: function (angleInRadians) {
    var c = Math.cos(angleInRadians);
    var s = Math.sin(angleInRadians);

    return new Float32Array([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  },

  scaling: function (sx, sy, sz) {
    return new Float32Array([
      sx,
      0,
      0,
      0,
      0,
      sy,
      0,
      0,
      0,
      0,
      sz,
      0,
      0,
      0,
      0,
      1,
    ]);
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

  inverse: function (m) {
    var r = new Float32Array(16);

    r[0] =
      m[5] * m[10] * m[15] -
      m[5] * m[14] * m[11] -
      m[6] * m[9] * m[15] +
      m[6] * m[13] * m[11] +
      m[7] * m[9] * m[14] -
      m[7] * m[13] * m[10];
    r[1] =
      -m[1] * m[10] * m[15] +
      m[1] * m[14] * m[11] +
      m[2] * m[9] * m[15] -
      m[2] * m[13] * m[11] -
      m[3] * m[9] * m[14] +
      m[3] * m[13] * m[10];
    r[2] =
      m[1] * m[6] * m[15] -
      m[1] * m[14] * m[7] -
      m[2] * m[5] * m[15] +
      m[2] * m[13] * m[7] +
      m[3] * m[5] * m[14] -
      m[3] * m[13] * m[6];
    r[3] =
      -m[1] * m[6] * m[11] +
      m[1] * m[10] * m[7] +
      m[2] * m[5] * m[11] -
      m[2] * m[9] * m[7] -
      m[3] * m[5] * m[10] +
      m[3] * m[9] * m[6];

    r[4] =
      -m[4] * m[10] * m[15] +
      m[4] * m[14] * m[11] +
      m[6] * m[8] * m[15] -
      m[6] * m[12] * m[11] -
      m[7] * m[8] * m[14] +
      m[7] * m[12] * m[10];
    r[5] =
      m[0] * m[10] * m[15] -
      m[0] * m[14] * m[11] -
      m[2] * m[8] * m[15] +
      m[2] * m[12] * m[11] +
      m[3] * m[8] * m[14] -
      m[3] * m[12] * m[10];
    r[6] =
      -m[0] * m[6] * m[15] +
      m[0] * m[14] * m[7] +
      m[2] * m[4] * m[15] -
      m[2] * m[12] * m[7] -
      m[3] * m[4] * m[14] +
      m[3] * m[12] * m[6];
    r[7] =
      m[0] * m[6] * m[11] -
      m[0] * m[10] * m[7] -
      m[2] * m[4] * m[11] +
      m[2] * m[8] * m[7] +
      m[3] * m[4] * m[10] -
      m[3] * m[8] * m[6];

    r[8] =
      m[4] * m[9] * m[15] -
      m[4] * m[13] * m[11] -
      m[5] * m[8] * m[15] +
      m[5] * m[12] * m[11] +
      m[7] * m[8] * m[13] -
      m[7] * m[12] * m[9];
    r[9] =
      -m[0] * m[9] * m[15] +
      m[0] * m[13] * m[11] +
      m[1] * m[8] * m[15] -
      m[1] * m[12] * m[11] -
      m[3] * m[8] * m[13] +
      m[3] * m[12] * m[9];
    r[10] =
      m[0] * m[5] * m[15] -
      m[0] * m[13] * m[7] -
      m[1] * m[4] * m[15] +
      m[1] * m[12] * m[7] +
      m[3] * m[4] * m[13] -
      m[3] * m[12] * m[5];
    r[11] =
      -m[0] * m[5] * m[11] +
      m[0] * m[9] * m[7] +
      m[1] * m[4] * m[11] -
      m[1] * m[8] * m[7] -
      m[3] * m[4] * m[9] +
      m[3] * m[8] * m[5];

    r[12] =
      -m[4] * m[9] * m[14] +
      m[4] * m[13] * m[10] +
      m[5] * m[8] * m[14] -
      m[5] * m[12] * m[10] -
      m[6] * m[8] * m[13] +
      m[6] * m[12] * m[9];
    r[13] =
      m[0] * m[9] * m[14] -
      m[0] * m[13] * m[10] -
      m[1] * m[8] * m[14] +
      m[1] * m[12] * m[10] +
      m[2] * m[8] * m[13] -
      m[2] * m[12] * m[9];
    r[14] =
      -m[0] * m[5] * m[14] +
      m[0] * m[13] * m[6] +
      m[1] * m[4] * m[14] -
      m[1] * m[12] * m[6] -
      m[2] * m[4] * m[13] +
      m[2] * m[12] * m[5];
    r[15] =
      m[0] * m[5] * m[10] -
      m[0] * m[9] * m[6] -
      m[1] * m[4] * m[10] +
      m[1] * m[8] * m[6] +
      m[2] * m[4] * m[9] -
      m[2] * m[8] * m[5];

    var det = m[0] * r[0] + m[1] * r[4] + m[2] * r[8] + m[3] * r[12];
    for (var i = 0; i < 16; i++) r[i] /= det;
    return r;
  },

  transpose: function (m) {
    var r = new Float32Array(16);
    r[0] = m[0];
    r[1] = m[4];
    r[2] = m[8];
    r[3] = m[12];
    r[4] = m[1];
    r[5] = m[5];
    r[6] = m[9];
    r[7] = m[13];
    r[8] = m[2];
    r[9] = m[6];
    r[10] = m[10];
    r[11] = m[14];
    r[12] = m[3];
    r[13] = m[7];
    r[14] = m[11];
    r[15] = m[15];
    return r;
  },
};

export default Renderer;
