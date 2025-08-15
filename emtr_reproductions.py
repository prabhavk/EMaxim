import sys

from emtr import EMTR

from flower_plot import plot_flower_initial_and_final_two_gradients

seq_file_name = "data/Randall_leaf.phyx"
seq_file_format = "phylip"
topology_file_name = "data/leaf_topology.csv"
init_criterion = "dirichlet"
root_search = 0

num_rep = 50
max_iter = 1000
conv_thresh = 0.0005
reproduction = sys.argv[1]

complete_prefix_for_output_files = (
    "results/Aug_15_" + str(reproduction)
    + "_rep_"
    + str(num_rep)
    + "_max_iter_"
    + str(max_iter)
    + "_conv_thresh_"
    + str(conv_thresh)
)
m = EMTR(
    sequence_file=seq_file_name,
    seq_file_format=seq_file_format,
    topology_file=topology_file_name,
    prefix_for_output=complete_prefix_for_output_files,    
    num_repetitions=num_rep,
    max_iter=max_iter,
    conv_threshold=conv_thresh
)

# EM at each internal node with initial parameters set with Dirichlet
m.EMdirichlet()
# EM at each internal node with initial parameters set with parsimony,
m.EMparsimony()

