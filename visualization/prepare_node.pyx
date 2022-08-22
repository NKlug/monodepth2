from panda3d.core import *

def prepare_scatter_node(float alpha,
                        float [:, :, :] colors,
                        double[:, :, :] coords_3d, float [:, :] scale,
                        float [:, :] relative_depths,
                        float max_depth, loader, str model_path,
                        use_relative_depths, float base_point_scale,
                        *args, **kwargs):
    cdef Py_ssize_t w = coords_3d.shape[0]
    cdef Py_ssize_t h = coords_3d.shape[1]
    frame_node = NodePath('frame node')

    cdef double [:, :, :] coords_3d_view = coords_3d
    cdef float [:, :] scale_view = scale
    cdef float [:, :, :] colors_view = colors
    cdef float [:, :] relative_depths_view = relative_depths

    cdef unsigned int i, j
    for i in range(w):
        for j in range(h):
            if max_depth is not None and relative_depths_view[i, j] > max_depth:
                continue
            sphere = loader.loadModel(model_path)
            texture = loader.loadTexture('../assets/mono_color.rgb')

            sphere.reparentTo(frame_node)

            if not use_relative_depths:
                sphere.setScale(base_point_scale)
            else:
                sphere.setScale(scale_view[i, j])
                # sphere.setScale(np.maximum(0.01, np.random.normal(0.03, 0.01)))

            sphere.setTexture(texture, 1)
            sphere.setPos(*coords_3d_view[i, j])
            sphere.setTransparency(True)
            sphere.setColor(*colors_view[i, j], alpha)

    return frame_node
