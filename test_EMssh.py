from maxim import mstB

seq_file_name = "data/Randall_leaf.phyx"
seq_file_format = "phylip"
topology_file_name = "data/leaf_topology.csv"

num_rep = 100
max_iter = 1000
conv_thresh = 0.0005

complete_prefix_for_output_files = (
    "results/Aug_14_1_"
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
    num_repetitions=num_rep,
    max_iter=max_iter,
    conv_threshold=conv_thresh,     
    prefix_for_output=complete_prefix_for_output_files,
)

m.SetprobFileforSSH()

m.EMssh()