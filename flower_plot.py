#!/usr/bin/env python3
import math
from typing import Union, Dict, List, Tuple
import ete3 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib import cm, colors
import argparse
from pathlib import Path


# ---------- helpers ----------


def _read_table(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read TSV or whitespace-separated table. Tries '\t' first, then whitespace.
    """
    path = str(path)
    try:
        df = pd.read_csv(path, sep="\t", engine="python", comment="#")
        if df.shape[1] == 1:
            raise ValueError("looks whitespace-separated, retrying")
        return df
    except Exception:
        return pd.read_csv(path, delim_whitespace=True, engine="python", comment="#")


def _pick_col(df: pd.DataFrame, sel: Union[int, str]) -> pd.Series:
    """
    Pick a column either by name or 0-based index.
    """
    if isinstance(sel, int):
        return df.iloc[:, sel]
    return df[sel]


def _load_grouped(
    tsv_path: Union[str, Path],
    root_col: Union[int, str] = "root",
    init_col: Union[int, str] = "edc-ll first",
    final_col: Union[int, str] = "edc-ll final",
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Returns (init_by_root, final_by_root): dict root -> list of values.
    """
    df = _read_table(tsv_path)

    roots = _pick_col(df, root_col).astype(str)
    init_vals = pd.to_numeric(_pick_col(df, init_col), errors="coerce")
    final_vals = pd.to_numeric(_pick_col(df, final_col), errors="coerce")

    ok = roots.notna() & init_vals.notna() & final_vals.notna()
    df = pd.DataFrame(
        {"root": roots[ok], "init": init_vals[ok], "final": final_vals[ok]}
    )

    init_by_root: Dict[str, List[float]] = {}
    final_by_root: Dict[str, List[float]] = {}

    for r, sub in df.groupby("root", sort=False):
        init_by_root[r] = sub["init"].tolist()
        final_by_root[r] = sub["final"].tolist()

    return init_by_root, final_by_root


def _label_from_distance(root: str, distances: Dict[str, int]) -> str:
    """Show ONLY the distance number as label."""
    return str(distances.get(root, ""))

def _label_inc_expname_and_distance(root: str, distances: Dict[str, int]) -> str:
    """Show exp name and the distance number as label."""
    return str(distances.get(root, ""))
    return str(distances.get(root, ""))

def _order_roots_clockwise_by_distance(
    distances: Dict[str, int], top: str = "h_21"
) -> List[str]:
    """
    Order roots by ascending distance, rotate so `top` is first.
    """
    roots = [k for k, _ in sorted(distances.items(), key=lambda kv: kv[1])]
    if top in roots:
        i = roots.index(top)
        roots = roots[i:] + roots[:i]
    return roots


def _generate_arcs_clockwise(
    n: int, theta_deg: float, gap_total_deg: float, start_at_deg: float = 90.0
):
    """
    Create n arcs laid out CLOCKWISE with given petal width `theta_deg` and total gap sum.
    Returns a list of dicts with start_deg, end_deg, mid_deg.
    """
    if n <= 0:
        return []
    gap_per = gap_total_deg / n
    step = theta_deg + gap_per  # center-to-center step
    arcs = []
    for i in range(n):
        mid = (start_at_deg - i * step) % 360.0  # clockwise placement
        start = mid - theta_deg / 2.0
        end = mid + theta_deg / 2.0
        arcs.append({"start_deg": start, "end_deg": end, "mid_deg": mid})
    return arcs


# ---------- main plotting function ----------


def plot_flower_initial_and_final_two_gradients(
    tsv_path: Union[str, Path],
    init_col: Union[int, str] = "edc-ll first",
    final_col: Union[int, str] = "edc-ll final",
    theta: float = 20.0,
    gap_total: float = 20.0,
    r_inner: float = 0.70,
    r_outer: float = 1.00,
    center: Tuple[float, float] = (0.0, 0.0),
    cmap_init: str = "Blues",  # defaulted as requested
    cmap_final: str = "Reds",  # defaulted as requested
    root_col: Union[int, str] = "root",
    title_left: str = "Initial log-likelihood",
    title_right: str = "Final log-likelihood",
    show_colorbars: bool = True,
    top_root: str = "h_21",
    vmin_init: float = None,
    vmax_init: float = None,
    vmin_final: float = None,
    vmax_final: float = None,
):
    """
    Read table and draw two flower plots (initial & final) with separate color scales.
    Petals are placed CLOCKWISE (h_21 at 12 o'clock), ordered by ascending hard-coded distance.
    Labels show ONLY the distance from root.

    init_col / final_col can be column names (recommended) or 0-based indices.
    """
    if not (0.0 <= r_inner < r_outer):
        raise ValueError("Require 0 <= r_inner < r_outer.")

    # hard-coded distances
    distances = {
        "h_21": 5,
        "h_22": 12,
        "h_23": 15,
        "h_24": 21,
        "h_25": 26,
        "h_26": 22,
        "h_27": 17,
        "h_28": 15,
        "h_29": 18,
        "h_30": 10,
        "h_31": 18,
        "h_32": 12,
        "h_33": 18,
        "h_34": 18,
        "h_35": 20,
        "h_36": 22,
        "h_37": 8,
    }

    distances_with_exp_name = {
        "h_21": "21|5",
        "h_22": "22|12",
        "h_23": "23|15",
        "h_24": "24|21",
        "h_25": "25|26",
        "h_26": "26|22",
        "h_27": "27|17",
        "h_28": "28|15",
        "h_29": "29|18",
        "h_30": "30|10",
        "h_31": "31|18",
        "h_32": "32|12",
        "h_33": "33|18",
        "h_34": "34|18",
        "h_35": "35|20",
        "h_36": "36|22",
        "h_37": "37|8",
    }

    init_by_root, final_by_root = _load_grouped(
        tsv_path, root_col=root_col, init_col=init_col, final_col=final_col
    )

    # CLOCKWISE order & arcs
    order = _order_roots_clockwise_by_distance(distances, top=top_root)
    order = [r for r in order if r in init_by_root or r in final_by_root]
    if not order:
        raise ValueError("No matching roots from TSV in the hard-coded distance list.")

    arcs = _generate_arcs_clockwise(
        n=len(order), theta_deg=theta, gap_total_deg=gap_total, start_at_deg=90.0
    )

    cx, cy = center
    band = r_outer - r_inner

    all_init = [v for r in order for v in init_by_root.get(r, [])]
    all_final = [v for r in order for v in final_by_root.get(r, [])]
    if not all_init or not all_final:
        raise ValueError("Parsed data has no initial or final values.")

    # Use provided range or fallback to data min/max
    if vmin_init is None:
        vmin_init = float(np.min(all_init))
    if vmax_init is None:
        vmax_init = float(np.max(all_init))
    if vmin_final is None:
        vmin_final = float(np.min(all_final))
    if vmax_final is None:
        vmax_final = float(np.max(all_final))

    norm_init = colors.Normalize(vmin=vmin_init, vmax=vmax_init)
    norm_final = colors.Normalize(vmin=vmin_final, vmax=vmax_final)
    cm_init = cm.get_cmap(cmap_init)
    cm_final = cm.get_cmap(cmap_final)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    
    # angle for rotating exp labels

    def angle_for_labels(mid):        
        if 5 < mid and mid < 175:            
            return (270 + mid)
        elif 190 < mid and mid < 350:
            return (90 + mid)
        else:
            return (mid)
    
    
    # ----- Left: initial -----
    for root, arc in zip(order, arcs):
        vals = init_by_root.get(root, [])
        if not vals:
            continue
        m = len(vals)
        for idx, val in enumerate(vals):
            rin = r_inner + (band * idx) / m
            rout = r_inner + (band * (idx + 1)) / m
            axL.add_patch(
                Wedge(
                    (cx, cy),
                    rout,
                    theta1=arc["start_deg"],
                    theta2=arc["end_deg"],
                    width=(rout - rin),
                    facecolor=cm_init(norm_init(val)),
                    edgecolor="none",
                )
            )
    # Labels (left)
    pad = 0.12 * r_outer
    for root, arc in zip(order, arcs):
        mid = arc["mid_deg"]
        ang = math.radians(mid)
        label_r = r_outer + 0.06
        # print(mid, root)
        axL.text(
            cx + label_r * math.cos(ang),
            cy + label_r * math.sin(ang),
            _label_from_distance(root, distances_with_exp_name),
            ha="center",
            va="center",            
            rotation=(angle_for_labels(mid)),  # CW tangential
            # rotation=(90.0 + mid),
            rotation_mode="anchor",
            fontsize=8,
        )
    axL.set_xlim(cx - r_outer - pad, cx + r_outer + pad)
    axL.set_ylim(cy - r_outer - pad, cy + r_outer + pad)
    axL.set_aspect("equal", adjustable="box")
    axL.axis("off")
    axL.set_title(title_left)

    # ----- Right: final -----
    for root, arc in zip(order, arcs):
        vals = final_by_root.get(root, [])
        if not vals:
            continue
        m = len(vals)
        for idx, val in enumerate(vals):
            rin = r_inner + (band * idx) / m
            rout = r_inner + (band * (idx + 1)) / m
            axR.add_patch(
                Wedge(
                    (cx, cy),
                    rout,
                    theta1=arc["start_deg"],
                    theta2=arc["end_deg"],
                    width=(rout - rin),
                    facecolor=cm_final(norm_final(val)),
                    edgecolor="none",
                )
            )
    # Labels (right)
    for root, arc in zip(order, arcs):
        mid = arc["mid_deg"]
        ang = math.radians(mid)
        label_r = r_outer + 0.06
        axR.text(
            cx + label_r * math.cos(ang),
            cy + label_r * math.sin(ang),
            _label_from_distance(root, distances_with_exp_name),
            ha="center",
            va="center",
            rotation=(angle_for_labels(mid)),  # CW tangential
            rotation_mode="anchor",
            fontsize=8,
        )
    axR.set_xlim(cx - r_outer - pad, cx + r_outer + pad)
    axR.set_ylim(cy - r_outer - pad, cy + r_outer + pad)
    axR.set_aspect("equal", adjustable="box")
    axR.axis("off")
    axR.set_title(title_right)

    # Colorbars
    if show_colorbars:
        smL = cm.ScalarMappable(norm=norm_init, cmap=cm_init)
        smL.set_array([])
        cbarL = fig.colorbar(smL, ax=axL, fraction=0.046, pad=0.04)
        cbarL.set_label("Initial log-likelihood")

        smR = cm.ScalarMappable(norm=norm_final, cmap=cm_final)
        smR.set_array([])
        cbarR = fig.colorbar(smR, ax=axR, fraction=0.046, pad=0.04)
        cbarR.set_label("Final log-likelihood")

    return fig, (axL, axR)


# ---------- CLI ----------


def main():
    p = argparse.ArgumentParser(
        description="Flower plot (clockwise) for initial/final log-likelihoods."
    )
    p.add_argument("tsv", help="Input TSV or whitespace-separated table.")

    # columns: name or 0-based index
    p.add_argument(
        "--init-col",
        default="edc-ll first",
        help="Column for initial values (name or 0-based index).",
    )
    p.add_argument(
        "--final-col",
        default="edc-ll final",
        help="Column for final values (name or 0-based index).",
    )
    p.add_argument(
        "--root-col", default="root", help='Column for root labels (e.g., "h_21").'
    )

    # geometry
    p.add_argument(
        "--theta",
        type=float,
        default=20.0,
        help="Angular width of each petal (degrees).",
    )
    p.add_argument(
        "--gap-total",
        type=float,
        default=20.0,
        help="Total angular gap shared across all petals (degrees).",
    )
    p.add_argument(
        "--r-inner", type=float, default=0.70, help="Inner radius of the band."
    )
    p.add_argument(
        "--r-outer", type=float, default=1.00, help="Outer radius of the band."
    )
    p.add_argument("--top-root", default="h_21", help="Root to place at 12 o’clock.")

    # colormaps
    p.add_argument(
        "--cmap-init",
        default="Blues",
        help='Matplotlib colormap for initial values (e.g., "Blues", "Blues_r").',
    )
    p.add_argument(
        "--cmap-final",
        default="Reds",
        help='Matplotlib colormap for final values (e.g., "Reds", "Reds_r").',
    )

    # optional fixed color ranges
    p.add_argument(
        "--vmin-init",
        type=float,
        default=None,
        help="Fix lower bound for initial colormap.",
    )
    p.add_argument(
        "--vmax-init",
        type=float,
        default=None,
        help="Fix upper bound for initial colormap.",
    )
    p.add_argument(
        "--vmin-final",
        type=float,
        default=None,
        help="Fix lower bound for final colormap.",
    )
    p.add_argument(
        "--vmax-final",
        type=float,
        default=None,
        help="Fix upper bound for final colormap.",
    )

    # output
    p.add_argument(
        "--out",
        default="",
        help="Save image to this path (e.g., plot.png). If empty, show window.",
    )
    args = p.parse_args()

    # coerce possible numeric indexes
    init_col = int(args.init_col) if str(args.init_col).isdigit() else args.init_col
    final_col = int(args.final_col) if str(args.final_col).isdigit() else args.final_col
    root_col = int(args.root_col) if str(args.root_col).isdigit() else args.root_col

    fig, _ = plot_flower_initial_and_final_two_gradients(
        args.tsv,
        init_col=init_col,
        final_col=final_col,
        root_col=root_col,
        theta=args.theta,
        gap_total=args.gap_total,
        r_inner=args.r_inner,
        r_outer=args.r_outer,
        top_root=args.top_root,
        cmap_init=args.cmap_init,
        cmap_final=args.cmap_final,
        vmin_init=args.vmin_init,
        vmax_init=args.vmax_init,
        vmin_final=args.vmin_final,
        vmax_final=args.vmax_final,
    )

    if args.out:
        fig.savefig(args.out, dpi=800, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
