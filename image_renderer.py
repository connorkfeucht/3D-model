import h5py
import numpy as np
import pyvista as pv

# Open and locate the mesh groups 
with h5py.File("Box.hdf5", "r") as f:
    mesh_root = f["parts"]["part_001"]["mesh"] # contains subgroups 000, 001, ... which each contain one small mesh

    # Load each sub-mesh into a PolyData and collect them
    polys = []
    for mesh_id in sorted(mesh_root.keys(), key=int): # get names like 000, 001, ... and sort them numerically
        grp = mesh_root[mesh_id]
        pts = grp["points"][...]       # shape (4,3), absolute coords of each vertex
        tris = grp["triangle"][...]    # shape (2,3), how those vertices connect to form triangles

        # VTK wants faces in "flat" format: [3, i0, i1, i2,  3, j0, j1, j2, …]
        faces = np.hstack([
            np.concatenate([[3], tri.astype(np.int64)])
            for tri in tris
        ]) # loop over each triangle, prepend the count 3, cast to int64, then horizontally stack them all into one flat array

        poly = pv.PolyData(pts, faces) # builds PolyData objects, which are the small meshes in parts 000, 001, ...
        polys.append(poly)

# Merge all sub-meshes into one (optional)
mesh = polys[0]
for poly in polys[1:]:
    mesh = mesh.merge(poly)

# Render & save screenshot
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, show_edges=True)
plotter.set_background("white")
plotter.camera_position = "iso"
plotter.show(screenshot="Box.png")
