#!/usr/bin/env python3
import sys
import numpy as np
import glob
import os
from scipy.spatial import cKDTree
# -------------------------------------------------
# Usage
# -------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python pivotal_shift_from_gro.py LIPID:HEAD:PIVOTDIST ...")
    print("Example: python pivotal_shift_from_gro.py POPE:PO4:13 POPG:PO4:13")
    sys.exit(1)

lipid_args = sys.argv[1:]

# -------------------------------------------------
# Parse lipid definitions
# -------------------------------------------------
lipid_head = {}
lipid_pivotdist = {}

for l in lipid_args:
    parts = l.split(":")
    if len(parts) != 3:
        print(f"Error parsing '{l}'. Expected format LIPID:HEAD:PIVOTDIST")
        sys.exit(1)

    name, head, pivot = parts

    lipid_head[name] = head
    lipid_pivotdist[name] = float(pivot) / 10.0   # Å → nm

print("\nLoaded lipid definitions:")
for k in lipid_head:
    print(f"  {k}: head={lipid_head[k]}, pivot={lipid_pivotdist[k]:.3f} nm")

# -------------------------------------------------
# Find GRO files
# -------------------------------------------------
gro_files = sorted(glob.glob("*.gro"))

if not gro_files:
    print("No .gro files found in current directory.")
    sys.exit(1)

# -------------------------------------------------
# Process each GRO
# -------------------------------------------------
for input_gro in gro_files:

    # Skip already shifted files
    if input_gro.startswith("pivotal_"):
        continue

    output_gro = "pivotal_" + input_gro

    print(f"\nProcessing: {input_gro}")

    with open(input_gro, "r") as f:
        lines = f.readlines()

    title = lines[0]
    natoms = int(lines[1].strip())
    box_line = lines[-1]
    atom_lines = lines[2:-1]

    atoms = []

    # -------------------------
    # Read atoms
    # -------------------------
    for idx, line in enumerate(atom_lines):   # <--- add enumerate to get line index
        resnum = int(line[0:5])
        resname = line[5:10].strip()
        atomname = line[10:15].strip()
        atomnum = int(line[15:20])
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])

        atoms.append({
            "resnum": resnum,
            "resname": resname,
            "atomname": atomname,
            "atomnum": atomnum,
            "pos": np.array([x, y, z]),
            "line_idx": idx  # <-- NEW
        })

    # -------------------------
    # Group residues
    # -------------------------
    residues = {}
    residue_line_idx = {}

    current_residue_id = 0
    last_resnum = None
    last_resname = None

    for atom in atoms:
        # Detect new residue by change in resnum OR resname
        if atom["resnum"] != last_resnum or atom["resname"] != last_resname:
            current_residue_id += 1
            residue_line_idx[current_residue_id] = atom["line_idx"]

        residues.setdefault(current_residue_id, []).append(atom)

        last_resnum = atom["resnum"]
        last_resname = atom["resname"]

    # -------------------------
    # Build KD-tree for fast neighbor lookup
    # -------------------------
    all_head_positions = []
    all_resnums = []

    for rnum, atom_list in residues.items():
        resname = atom_list[0]["resname"]
        if resname in lipid_head:
            head_pos_tmp = np.mean([
                a["pos"] for a in atom_list if a["atomname"] == lipid_head[resname]
            ], axis=0)
            all_head_positions.append(head_pos_tmp)
            all_resnums.append(rnum)

    all_head_positions = np.array(all_head_positions)
    tree = cKDTree(all_head_positions)  # ✅ build once

    # -------------------------
    # Compute and apply shifts (robust)
    # -------------------------
    shifted_lipids = 0

    for resnum, atom_list in residues.items():
        resname = atom_list[0]["resname"]
        if resname not in lipid_head:
            continue  # skip non-target lipids

        # ----- 1. Compute head position -----
        head_atoms = [a["pos"] for a in atom_list if a["atomname"] == lipid_head[resname]]
        head_pos = np.mean(head_atoms, axis=0) if head_atoms else atom_list[0]["pos"]

        # ----- 2. Compute neighbor-aware tail direction -----
        tail_positions = np.array([a["pos"] for a in atom_list if a["atomname"].startswith("C")])

        if len(tail_positions) < 3:
            tail_vec = np.array([0.0, 0.0, 1.0])
        else:
            # Local tail vector
            local_tail_vec = np.mean(tail_positions, axis=0) - head_pos
            norm = np.linalg.norm(local_tail_vec)
            local_tail_vec = local_tail_vec / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])

            # Neighbor smoothing with line window
            cutoff = 1.5  # nm
            alpha = 0.6
            window = 50  # ±50 lines ~ 100-line window
            neighbors_idx = tree.query_ball_point(head_pos, cutoff)

            neighbor_vecs = []
            for idx in neighbors_idx:
                neighbor_resnum = all_resnums[idx]
                if neighbor_resnum == resnum:
                    continue  # skip self

                # Skip residues outside the local line window
                if abs(residue_line_idx[neighbor_resnum] - residue_line_idx[resnum]) > window:
                    continue

                neighbor_atoms = residues[neighbor_resnum]
                neighbor_tail = np.array([a["pos"] for a in neighbor_atoms if a["atomname"].startswith("C")])
                if len(neighbor_tail) < 1:
                    continue
                vec_j = np.mean(neighbor_tail, axis=0) - all_head_positions[idx]
                norm_j = np.linalg.norm(vec_j)
                if norm_j < 1e-8:
                    continue
                vec_j /= norm_j
                neighbor_vecs.append(vec_j)

            if neighbor_vecs:
                neighbor_avg = np.mean(neighbor_vecs, axis=0)
                neighbor_avg /= np.linalg.norm(neighbor_avg)
                tail_vec = alpha * local_tail_vec + (1 - alpha) * neighbor_avg
                tail_vec /= np.linalg.norm(tail_vec)
            else:
                tail_vec = local_tail_vec

        # ----- 3. Compute shift vector -----
        shift_vector = tail_vec * lipid_pivotdist[resname]

        # ----- 4. Apply shift to all atoms in residue -----
        for atom in atom_list:
            atom["pos"] += shift_vector

        shifted_lipids += 1

    print(f"  Shifted {shifted_lipids} lipid residues")

   

    # -------------------------
    # Write new GRO
    # -------------------------
    with open(output_gro, "w") as f:
        f.write(title)
        f.write(f"{natoms:5d}\n")

        for atom in atoms:
            x, y, z = atom["pos"]
            f.write(
                f"{atom['resnum']:5d}"
                f"{atom['resname']:<5}"
                f"{atom['atomname']:<5}"
                f"{atom['atomnum']:5d}"
                f"{x:8.3f}{y:8.3f}{z:8.3f}\n"
            )

        f.write(box_line)

    print(f"  → Wrote {output_gro}")

print("\nDone.")
