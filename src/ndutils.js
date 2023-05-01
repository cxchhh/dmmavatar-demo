import ndarray from "ndarray";
export function copy(arr) {
    return ndarray(arr.data.slice(), arr.shape);
}

export function getsz(shape) {
    var sz = 1;
    for (var i = 0; i < shape.length; ++i) {
        sz *= shape[i];
    }
    return sz;
}

export function reshape(ndarr, new_shape) {
    var sz = getsz(ndarr.shape);
    var sd = 1;
    for (var i = new_shape.length - 1; i >= 0; --i) {
        if (i > 0) sd *= new_shape[i];
    }
    if (new_shape[0] < 0) new_shape[0] = sz / sd;
    var new_arr = ndarray(ndarr.data.slice(), new_shape);
    return new_arr;
}
export function reshape_(ndarr, new_shape) {
    var sz = getsz(ndarr.shape);
    var sd = 1;
    for (var i = new_shape.length - 1; i >= 0; --i) {
        if (i > 0) sd *= new_shape[i];
    }
    if (new_shape[0] < 0) new_shape[0] = sz / sd;
    ndarr = ndarray(ndarr.data.slice(), new_shape);
    return ndarr;
}