import npyjs from "npyjs";
import ndarray from "ndarray";
import FLAME from "./flame.js";
import show from "ndarray-show";
import zeros from "zeros";
import Renderer from "./renderer.js";
import { addeq, subseq, addseq, mul } from "ndarray-ops";
import MLP from "./mlp.js";
import m4 from "./m4.js";

let n = new npyjs();

async function loadnpy(url) {
    var npy = await n.load(url);
    return ndarray(new Float32Array(npy.data), npy.shape);
}

async function loadnpyu(url) {
    var npy = await n.load(url);
    return ndarray(new Uint32Array(npy.data), npy.shape);
}

function radToDeg(r) {
    return (r * 180) / Math.PI;
}

function degToRad(d) {
    return (d * Math.PI) / 180;
}

window.onload = async function () {
    this.canvas = document.querySelector("#canvas");
    this.gl = this.canvas.getContext("webgl2");
    if (!gl) {
        alert("WebGL isn't available on your broswer!");
        return;
    }
    var pp = document.getElementById("timeTest");

    var J_regressor_c = await loadnpy("mesh_data/common/J_regressor.npy");
    var parents_c = await loadnpy("mesh_data/common/parents.npy");
    var v_template = await loadnpy("mesh_data/common/v_template.npy");
    var shapedirs_c = await loadnpy("mesh_data/common/shapedirs.npy");
    var posedirs_c = await loadnpy("mesh_data/common/posedirs.npy");
    this.flame = new FLAME(v_template, shapedirs_c, posedirs_c, J_regressor_c, parents_c);
    pp.innerHTML = "common loaded\n";

    var fc0_weight = await loadnpy("mesh_data/global_mlp/fc0_weight.npy");
    var fc0_bias = await loadnpy("mesh_data/global_mlp/fc0_bias.npy");
    var fc1_weight = await loadnpy("mesh_data/global_mlp/fc1_weight.npy");
    var fc1_bias = await loadnpy("mesh_data/global_mlp/fc1_bias.npy");
    var fc2_weight = await loadnpy("mesh_data/global_mlp/fc2_weight.npy");
    var fc2_bias = await loadnpy("mesh_data/global_mlp/fc2_bias.npy");
    var act0 = await loadnpy("mesh_data/global_mlp/act0_weight.npy");
    var act1 = await loadnpy("mesh_data/global_mlp/act1_weight.npy");
    this.global_mlp = new MLP(
        fc0_weight,
        fc0_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        act0,
        act1
    );
    this.global_input = new Float32Array(53);
    pp.innerHTML += "MLP loaded\n";

    this.betas = zeros([50], "float32");
    this.pose_params = zeros([15], "float32");
    //this.pose_params.set(6,0.2);
    this.M = 8;
    this.vertices = [];
    this.faces = [];
    this.lbs_weights = [];
    this.normals = [];
    this.posedirs = [];
    this.shapedirs = [];
    this.uvs = [];
    this.renderers = [];
    for (let i = 0; i < this.M; i++) {
        this.vertices.push(await loadnpy(`mesh_data/meshes_${i}/vertices.npy`));
        this.faces.push(await loadnpyu(`mesh_data/meshes_${i}/faces.npy`));
        this.lbs_weights.push(await loadnpy(`mesh_data/meshes_${i}/lbs_weights.npy`));
        this.normals.push(await loadnpy(`mesh_data/meshes_${i}/normals.npy`));
        this.posedirs.push(await loadnpy(`mesh_data/meshes_${i}/posedirs.npy`));
        this.shapedirs.push(await loadnpy(`mesh_data/meshes_${i}/shapedirs.npy`));
        this.uvs.push(await loadnpy(`mesh_data/meshes_${i}/uvs.npy`));
        this.renderers.push(
            new Renderer(
                i,
                this.vertices[i],
                this.faces[i],
                this.lbs_weights[i],
                this.posedirs[i],
                this.shapedirs[i],
                this.uvs[i],
                this.normals[i]
            )
        );
    }
    pp.innerHTML += "meshes loaded\n";

    await forward(this);
    //console.log(show(this.transform));
    for (let i = 0; i < this.M; i++) {
        await this.renderers[i].setStatics();
    }
    uiInit(this);
};
async function forward(th) {
    var retVal = await th.flame.lbs(th.betas, th.pose_params);
    th.poses = retVal.ret1;
    th.transform = retVal.ret2;
    th.global_input.set(th.betas.data, 0);
    th.global_input[50] = th.pose_params.data[6];
    th.global_input[51] = th.pose_params.data[7];
    th.global_input[52] = th.pose_params.data[8];
    th.global_output = await th.global_mlp.forward(th.global_input);
    //console.log(this.global_output);
}

function uiInit(th) {
    // First let's make some variables
    // to hold the translation,
    var translation = [0, 0, -360];
    var rotation = [0, 0, 0];
    var fieldOfViewRadians = degToRad(45);
    webglLessonsUI.setupSlider("#fieldOfView", {
        value: radToDeg(fieldOfViewRadians),
        slide: updateFieldOfView,
        min: 1,
        max: 179,
    });
    webglLessonsUI.setupSlider("#F_num", {
        value: 1,
        slide: updateF(),
        min: 0,
        max: 1,
        step: 0.0001,
        precision: 4,
    });
    for (var i = 0; i < 50; i++)
        webglLessonsUI.setupSlider("#exp" + (i + 1), {
            value: th.betas.data[i],
            slide: updateBetas(i),
            step: 0.001,
            min: -2,
            max: 2,
            precision: 3,
        });

    for (var i = 0; i < 15; i++)
        webglLessonsUI.setupSlider("#pose" + (i + 1), {
            value: th.pose_params.data[i],
            slide: updatePoses(i),
            step: 0.001,
            min: -0.5,
            max: 0.5,
            precision: 3,
        });
    function updateBetas(i) {
        return async function (e, ui) {
            th.betas.set(i, ui.value);
            await forward(th);
        };
    }
    function updatePoses(i) {
        return async function (e, ui) {
            th.pose_params.set(i, ui.value);
            await forward(th);
        };
    }
    function updateFieldOfView(event, ui) {
        fieldOfViewRadians = degToRad(ui.value);
    }
    function updateF() {
        return function (event, ui) {
            for (let i = 0; i < th.M; i++) th.renderers[i].F_num = ui.value;
        };
    }
    th.canvas.addEventListener("contextmenu", function (e) {
        e.preventDefault();
    });
    var lastx = 0,
        lasty = 0,
        incanvas = true,
        down = false,
        basex,
        basey,
        btn = 0;
    var gaze_t, gaze_f, rotx, roty, lastrotx, lastroty;
    th.canvas.onmouseenter = function (e) {
        incanvas = true;
        disableWindowScroll();
    };
    th.canvas.onmouseleave = function (e) {
        incanvas = false;
        down = false;
        enableWindowScroll();
    };
    th.canvas.onmousedown = function (e) {
        if (!incanvas) return;
        down = true;
        if (e.button == 2) {
            btn = 2;
            basex = translation[0];
            basey = translation[1];
            lastx = e.clientX;
            lasty = e.clientY;
        } else if (e.button == 0) {
            btn = 0;
            rotx = rotation[0];
            roty = rotation[1];
            lastrotx = e.clientY;
            lastroty = e.clientX;
        }
    };
    th.canvas.onmousemove = function (e) {
        if (!down) return;
        if (btn == 2) {
            translation[0] = basex + e.clientX - lastx;
            translation[1] = basey - (e.clientY - lasty);
        }
        if (btn == 0) {
            rotation[0] = rotx + (e.clientY - lastrotx) / 250;
            rotation[1] = roty + (e.clientX - lastroty) / 250;
        }
    };
    th.canvas.onmouseup = function (e) {
        if (!incanvas) return;
        down = false;
    };
    th.canvas.onwheel = function (e) {
        if (!incanvas) return;
        translation[2] += e.wheelDelta / 2;
        if (translation[2] > 1) translation[2] = 1;
        if (translation[2] < -1000) translation[2] = -1000;
    };
    var winX = null;
    var winY = null;
    window.addEventListener("scroll", function () {
        if (winX !== null && winY !== null) {
            window.scrollTo(winX, winY);
        }
    });
    function disableWindowScroll() {
        winX = window.scrollX;
        winY = window.scrollY;
    }
    function enableWindowScroll() {
        winX = null;
        winY = null;
    }

    var framwCount = 0;
    var lastTime = performance.now();
    var timeTest = document.getElementById("timeTest");
    var gl = th.gl;

    async function drawLoop() {
        // Compute the matrix
        var aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
        var zNear = 1;
        var zFar = 2000;
        //var matrix=m4.scaling(1,1,1);
        var projection_matrix = m4.perspective(fieldOfViewRadians, aspect, zNear, zFar);
        var view_matrix = m4.scaling(1, 1, 1);
        view_matrix = m4.translate(view_matrix, translation[0], translation[1], translation[2]);
        view_matrix = m4.xRotate(view_matrix, rotation[0]);
        view_matrix = m4.yRotate(view_matrix, rotation[1]);
        view_matrix = m4.zRotate(view_matrix, rotation[2]);

        for (let i = 0; i < th.M; i++) {
            th.renderers[i].matrix = m4.multiply(projection_matrix, view_matrix);
            th.renderers[i].normal_matrix = m4.transpose(m4.inverse(view_matrix));
            th.renderers[i].view_matrix = view_matrix;
            th.renderers[i].render(th.betas, th.poses, th.transform);
        }

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
