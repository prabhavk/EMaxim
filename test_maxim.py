from emaxim import mstB

seq_file_name = "data/Randall_leaf.phyx"
seq_file_format = "phylip"
topology_file_name = "data/leaf_topology.csv"
num_rep = 10
max_iter = 1000
conv_thresh = 0.0005
prefix_for_output_files = (
    "results/Aug_13_"
    + "rep_"
    + str(num_rep)
    + "_max_iter_"
    + str(max_iter)
    + "_conv_thresh_"
    + str(conv_thresh)
    + "_reproduction_3"
)
prob_file_name = (
    "results/probability_file_"
    + "rep_"
    + str(num_rep)
    + "_max_iter_"
    + str(max_iter)
    + "_conv_thresh_"
    + str(conv_thresh)
    + "_reproduction_3"
)

m = mstB(
    sequence_file=seq_file_name,
    seq_file_format=seq_file_format,
    topology_file=topology_file_name,
    num_repetitions=num_rep,
    max_iter=max_iter,
    conv_threshold=conv_thresh,
    probability_file=prob_file_name,
    prefix_for_output=prefix_for_output_files,
)

# EM at each internal node with initial parameters set with parsimony,
# and saves parameters yielding maximum log likelihood score
m.EMpars()

# EM at each internal node with reparameterized parameters yielding maximum log likelihood score
# computed in previous step. Make sure to start this routine after the parsimony one
m.EMssh()
