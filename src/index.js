import npyjs from "npyjs";
import ndarray from "ndarray";
import FLAME from "./flame.js";
import show from "ndarray-show";
import zeros from "zeros";
import Renderer from "./renderer.js";
import { mul } from "ndarray-ops";

let n = new npyjs();
async function loadnpy(url) {
    var npy = await n.load(url);
    return ndarray(npy.data, npy.shape);
}
window.onload = async function () {
    pp = document.getElementById("timeTest");
    var J_regressor_c = (await loadnpy("mesh_data/common/J_regressor.npy"));
    var parents_c = (await loadnpy("mesh_data/common/parents.npy"));
    var v_template = (await loadnpy("mesh_data/common/v_template.npy"));
    var shapedirs_c = (await loadnpy("mesh_data/common/shapedirs.npy"));
    var posedirs_c = (await loadnpy("mesh_data/common/posedirs.npy"));
    var lbs_weights_c = (await loadnpy("mesh_data/common/lbs_weights.npy"));
    pp.innerHTML = 'common loaded\n';

    var vertices = (await loadnpy("mesh_data/meshes_0/vertices.npy"));
    var faces = (await loadnpy("mesh_data/meshes_0/faces.npy"));
    var faces_uv = (await loadnpy("mesh_data/meshes_0/faces_uv.npy"));
    pp.innerHTML += 'verts & faces loaded\n';
    var lbs_weights = (await loadnpy("mesh_data/meshes_0/lbs_weights.npy"));
    var normals = (await loadnpy("mesh_data/meshes_0/normals.npy"));
    pp.innerHTML += 'lbs_w & normals loaded\n';
    var posedirs = (await loadnpy("mesh_data/meshes_0/posedirs.npy"));
    pp.innerHTML += 'posedirs loaded\n';
    var shapedirs = (await loadnpy("mesh_data/meshes_0/shapedirs.npy"));
    pp.innerHTML += 'shapedirs loaded\n';
    var uvs = (await loadnpy("mesh_data/meshes_0/uvs.npy"));

    var radian = 3.14159265 / 180;
    var shape_params = zeros([100], 'float32');
    var expression_params = zeros([50], 'float32');
    var pose_params = zeros([15], 'float32');
    var pp = document.getElementById("timeTest");
    var V=vertices.shape[0];

    let flame = new FLAME(v_template, shapedirs_c, posedirs_c, J_regressor_c, parents_c, lbs_weights_c);
    let renderer = new Renderer(vertices, faces, lbs_weights, posedirs, shapedirs);
    var pose_feature, transform, retVal;

    //pp.innerHTML=uvs.shape;
    
    var st=Date.now();
    //expression_params.set(0, 1);
    //pose_params.set(0,1);
    retVal = await flame.lbs(expression_params, pose_params);
    pose_feature = retVal.ret1;
    transform = retVal.ret2;
    
    await renderer.render(expression_params,transform,pose_feature);
    alert((Date.now()-st)+'ms');
}
