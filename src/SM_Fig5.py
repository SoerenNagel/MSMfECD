import plot_libarry as plt_lib
import pickle

if __name__ == "__main__":
    file = f"data/data_SM_Fig5ab"
    with open(file, "rb") as f:
        process = pickle.load(f)
    plt_lib.plot_saving_rate_color_plot(process)
    plt_lib.plot_production_and_mean_saving_rate(process)

    file = (
        f"data/data_SM_Fig5cd"
    )
    with open(file, "rb") as f:
        process = pickle.load(f)
    plt_lib.plot_saving_rate_color_plot(process)
    plt_lib.plot_production_and_mean_saving_rate(process)
