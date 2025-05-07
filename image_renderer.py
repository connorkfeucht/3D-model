import h5py
import numpy as np
import pyvista as pv

# --- 1) Open and locate the mesh groups ---
with h5py.File("Box.hdf5", "r") as f:
    mesh_root = f["parts"]["part_001"]["mesh"]

    # --- 2) Load each sub-mesh into a PolyData and collect them ---
    polys = []
    for mesh_id in sorted(mesh_root.keys(), key=int):
        grp = mesh_root[mesh_id]
        pts = grp["points"][...]       # shape (4,3)
        tris = grp["triangle"][...]    # shape (2,3)

        # VTK wants faces in "flat" format: [3, i0, i1, i2,  3, j0, j1, j2, …]
        faces = np.hstack([
            np.concatenate([[3], tri.astype(np.int64)])
            for tri in tris
        ])

        poly = pv.PolyData(pts, faces)
        polys.append(poly)

# --- 3) Merge all sub-meshes into one (optional) ---
mesh = polys[0]
for poly in polys[1:]:
    mesh = mesh.merge(poly)

# --- 4) Render & save screenshot ---
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, show_edges=True)
plotter.set_background("white")
plotter.camera_position = "iso"
plotter.show(screenshot="Box.png")

print("Wrote Box.png")
