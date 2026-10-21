from linear_mountaincar.evaluation.configuration import Configuration

output_dir = "agents"
results_dir = "results"

configuration: Configuration = Configuration(
    learning_rate_list=[0.005, 0.01, 0.02, 0.05, 0.1],
    number_of_centers_list=[9, 16, 25, 36, 49],
    seeds_list=[42, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
)