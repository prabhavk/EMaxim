import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm, colors

# --- helpers ---------------------------------------------------------------

def _load_grouped(tsv_path: str, root_col=0, init_col=4, final_col=5):
    init_by_root: Dict[str, List[float]] = defaultdict(list)
    final_by_root: Dict[str, List[float]] = defaultdict(list)
    with open(tsv_path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            row = s.split()  # whitespace-separated
            try:
                root = row[root_col]
                initv = float(row[init_col])
                finalv = float(row[final_col])
            except (IndexError, ValueError):
                continue
            init_by_root[root].append(initv)
            final_by_root[root].append(finalv)
    return init_by_root, final_by_root

def _order_roots_anticlockwise_with_h21_top(distances: Dict[str, float], top="h_21") -> List[str]:
    ordered = sorted(distances.keys(), key=lambda k: (distances[k], k))
    i = ordered.index(top)
    return ordered[i:] + ordered[:i]

def _generate_arcs_anticlockwise(n: int, theta_deg: float, gap_total_deg: float, start_at_deg: float = 90.0):
    gap = gap_total_deg / n
    span = theta_deg + gap
    arcs = []
    for k in range(n):
        start_deg = start_at_deg + k * span
        end_deg   = start_deg + theta_deg
        arcs.append({"start_deg": start_deg, "end_deg": end_deg})
    return arcs

def _label_from_distance(root: str, distances: Dict[str, float]) -> str:
    if root not in distances:
        return root
    try:
        node_num = root.split("_", 1)[1]
    except IndexError:
        node_num = root
    return f"{node_num}({distances[root]})"

# --- main plotter ----------------------------------------------------------

def plot_flower_initial_final_dots_spiral(
    tsv_path: str,
    *,
    init_col: int = 4,
    final_col: int = 5,
    root_col: int = 0,
    # geometry
    theta: float = 20.0,
    gap_total: float = 20.0,
    r_inner: float = 0.70,
    r_outer: float = 1.00,
    center: Tuple[float, float] = (0.0, 0.0),
    # dots & spiral
    dot_radius: float = 0.008,      # in axis data units
    delta_deg: float = 0.6,         # angle offset per concentric ring (spiral feel)
    margin_deg: float = 1.0,        # keep dots away from arc edges
    # color
    cmap_name: str = "viridis",
    vmin: Optional[float] = None,   # if None, computed from all values (init+final)
    vmax: Optional[float] = None,
    # titles/labels
    title_left: str = "Initial log-likelihood",
    title_right: str = "Final log-likelihood",
    top_root: str = "h_21",
):
    """
    Read TSV and draw two flower plots (initial & final) using *dots* on each arc.
    Consecutive rings are angle-offset by `delta_deg` to look like a spiral.

    Shared color scale across both panels; colorbar shown on the right panel only.
    """
    # Hard-coded “distance from h_21” ordering (same as before)
    distances = {
        "h_21": 5,  "h_22": 12, "h_23": 15, "h_24": 21, "h_25": 26,
        "h_26": 22, "h_27": 17, "h_28": 15, "h_29": 18, "h_30": 10,
        "h_31": 18, "h_32": 12, "h_33": 18, "h_34": 18, "h_35": 20,
        "h_36": 22, "h_37": 8,
    }

    init_by_root, final_by_root = _load_grouped(
        tsv_path, root_col=root_col, init_col=init_col, final_col=final_col
    )

    order = _order_roots_anticlockwise_with_h21_top(distances, top=top_root)
    order = [r for r in order if r in init_by_root or r in final_by_root]
    if not order:
        raise ValueError("No matching roots from TSV in the distance list.")

    arcs = _generate_arcs_anticlockwise(
        n=len(order), theta_deg=theta, gap_total_deg=gap_total, start_at_deg=90.0
    )

    # Color scale shared across init + final
    all_vals = [v for r in order for v in init_by_root.get(r, [])] + \
               [v for r in order for v in final_by_root.get(r, [])]
    if not all_vals:
        raise ValueError("No values parsed from TSV.")
    if vmin is None: vmin = min(all_vals)
    if vmax is None: vmax = max(all_vals)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    cx, cy = center
    band = r_outer - r_inner
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    def _plot_panel(ax, by_root: Dict[str, List[float]], panel_title: str):
        for root, arc in zip(order, arcs):
            vals = by_root.get(root, [])
            if not vals:
                continue
            m = len(vals)
            for idx, val in enumerate(vals):
                # radial position (mid of slice)
                rin  = r_inner + (band * idx) / m
                rout = r_inner + (band * (idx + 1)) / m
                rmid = 0.5 * (rin + rout)

                # base angle is middle of the arc
                base = 0.5 * (arc["start_deg"] + arc["end_deg"])
                # spiral offset
                ang_deg = base + (idx - (m - 1) / 2.0) * delta_deg
                # clamp to stay inside arc margins
                lo = arc["start_deg"] + margin_deg
                hi = arc["end_deg"]   - margin_deg
                ang_deg = max(lo, min(hi, ang_deg))
                ang = math.radians(ang_deg)

                x = cx + rmid * math.cos(ang)
                y = cy + rmid * math.sin(ang)

                ax.add_patch(Circle((x, y),
                                    radius=dot_radius,
                                    facecolor=cmap(norm(val)),
                                    edgecolor='none'))
        # labels
        pad = 0.08 * r_outer
        for root, arc in zip(order, arcs):
            mid = 0.5 * (arc["start_deg"] + arc["end_deg"])
            ang = math.radians(mid)
            label_r = r_outer + 0.06
            ax.text(
                cx + label_r * math.cos(ang),
                cy + label_r * math.sin(ang),
                _label_from_distance(root, distances),
                ha="center", va="center",
                rotation=mid - 90, rotation_mode="anchor", fontsize=8,
            )
        ax.set_xlim(cx - r_outer - pad, cx + r_outer + pad)
        ax.set_ylim(cy - r_outer - pad, cy + r_outer + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(panel_title)

    # Left: initial; Right: final
    _plot_panel(axL, init_by_root, title_left)
    _plot_panel(axR, final_by_root, title_right)

    # single colorbar on the right
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axR, fraction=0.046, pad=0.04)
    cbar.set_label("Log-likelihood (shared scale)")

    return fig, (axL, axR)
