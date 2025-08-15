from emtr import mstB

from flower_plot import plot_flower_initial_and_final_two_gradients

seq_file_name = "data/Randall_leaf.phyx"
seq_file_format = "phylip"
topology_file_name = "data/leaf_topology.csv"
init_criterion = "dirichlet"
root_search = 0

num_rep = 50
max_iter = 1000
conv_thresh = 0.00005


complete_prefix_for_output_files = (
    "results/Aug_14_4_"
    + "rep_"
    + str(num_rep)
    + "_max_iter_"
    + str(max_iter)
    + "_conv_thresh_"
    + str(conv_thresh)
)
m = mstB(
    sequence_file=seq_file_name,
    seq_file_format=seq_file_format,
    topology_file=topology_file_name,
    prefix_for_output=complete_prefix_for_output_files,    
    num_repetitions=num_rep,
    max_iter=max_iter,
    conv_threshold=conv_thresh
)

# EM at each internal node with initial parameters set with Dirichlet
m.EMnoise()

# # # EM at each internal node with initial parameters set with parsimony,
# # # and saves parameters yielding maximum log likelihood score
m.EMpars()

m.SetprobFileforSSH()

# EM at each internal node with reparameterized parameters yielding maximum log likelihood score
# computed in previous step. Make sure to start this routine after the parsimony one
m.EMssh()



# results_files = {}

# results_files["dirichlet"] = "results/Aug_13_6_rep_100_max_iter_1000_conv_thresh_0.0005.dirichlet_rooting_initial_final_rep_loglik"
# results_files["parsimony"] = "results/Aug_13_6_rep_100_max_iter_1000_conv_thresh_0.0005.pars_rooting_initial_final_rep_loglik"
# results_files["ssh"]       = "results/Aug_13_6_rep_100_max_iter_1000_conv_thresh_0.0005.SSH_rooting_initial_final_rep_loglik"

# fig, _ = plot_flower_initial_and_final_two_gradients(tsv = "results/Aug_14_1_rep_100_max_iter_1000_conv_thresh_0.0005.dirichlet_rooting_initial_final_rep_loglik", out = "results/Aug_13_6_rep_100_dirichlet_flower.png", init_col="ll dirichlet", final_col="ll final", root_col="root", r_inner=0.1, r_outer=0.9, cmap_init="Blues", cmap_final="Reds")
# fig, _ = plot_flower_initial_and_final_two_gradients(tsv = "results/Aug_14_1_rep_100_max_iter_1000_conv_thresh_0.0005.pars_rooting_initial_final_rep_loglik", out = "results/Aug_13_6_rep_100_pars_flower.png", init_col="ll pars", final_col="ll final", root_col="root", r_inner=0.1, r_outer=0.9, cmap_init="Blues", cmap_final="Reds")
# fig, _ = plot_flower_initial_and_final_two_gradients(tsv = "results/Aug_14_1_rep_100_max_iter_1000_conv_thresh_0.0005.SSH_rooting_initial_final_rep_loglik", out = "results/Aug_13_6_rep_100_SSH_flower.png", init_col="ll SSH", final_col="ll final", root_col="root", r_inner=0.1, r_outer=0.9, cmap_init="Blues", cmap_final="Reds")


# from ring_plot import plot_flower_initial_final_dots_spiral


# plot_flower_initial_final_dots_spiral(tsv_path = "results/Aug_13_6_rep_100_max_iter_1000_conv_thresh_0.0005.dirichlet_rooting_initial_final_rep_loglik", init_col="ll dirichlet", final_col="ll final", root_col="root", r_inner=0.1, r_outer=0.9, cmap_init="Blues", cmap_final="Reds")

# python3 flower_plot.py results/Aug_14_4_rep_100_max_iter_1000_conv_thresh_5e-05.dirichlet_rooting_initial_final_rep_loglik --init-col "ll dirichlet" --final-col "ll final" --root-col "root" --out Aug_14_4_rep_100_dirichlet_flower.png --r-inner 0.1 --r-outer 0.8 --cmap-init "Blues" --cmap-final "Reds" --vmax-init -4380 --vmax-final -4360

# python3 flower_plot.py results/Aug_14_4_rep_100_max_iter_1000_conv_thresh_5e-05.dirichlet_rooting_initial_final_rep_loglik --init-col "ll dirichlet" --final-col "ll final" --root-col "root" --out Aug_14_4_rep_100_dirichlet_flower.png --r-inner 0.1 --r-outer 0.8 --cmap-init "Blues" --cmap-final "Reds" --vmin-final -4370 --vmax-final -4363

# python3 flower_plot.py results/Aug_14_4_rep_100_max_iter_1000_conv_thresh_5e-05.pars_rooting_initial_final_rep_loglik --init-col "ll pars" --final-col "ll final" --root-col "root" --out Aug_14_4_rep_100_parsimony_flower.png --r-inner 0.1 --r-outer 0.8 --cmap-init "Blues" --cmap-final "Reds" --vmin-final -4370 --vmax-final -4363 

# python3 flower_plot.py results/Aug_14_4_rep_100_max_iter_1000_conv_thresh_5e-05.SSH_rooting_initial_final_rep_loglik --init-col "ll SSH" --final-col "ll final" --root-col "root" --out Aug_14_4_rep_100_SSH_flower.png --r-inner 0.1 --r-outer 0.8 --cmap-init "Blues" --cmap-final "Reds"  --vmin-init -4370 --vmax-init -4363  --vmin-final -4370 --vmax-final -4363 