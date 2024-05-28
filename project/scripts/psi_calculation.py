import numpy as np
## TODO add equal freq binning


def quantile_binning(baseline:np.ndarray, production:np.ndarray, bin_num:int=10, eps:float=1e-5):
    baseline = np.sort(baseline)[::-1]
    production = np.sort(production)[::-1]

    min_val = min(min(baseline), min(production))
    max_val = max(max(baseline), max(production))

    return np.concatenate(([min_val - eps], np.quantile(baseline, np.arange(1/bin_num, 1, 1/bin_num)), [max_val + eps]))


def equal_width_binning(baseline:np.ndarray, production:np.ndarray, bin_num:int=10, eps:float=1e-5):
    baseline = np.sort(baseline)[::-1]
    production = np.sort(production)[::-1]

    min_val = min(min(baseline), min(production))
    max_val = max(max(baseline), max(production))
    diff = (max_val - min_val)/bin_num

    return [min_val - eps] + [min_val + diff*(i+1) for i in range(bin_num-1)] + [min_val + eps]


def median_centered_binning(baseline:np.ndarray, production:np.ndarray, bin_num:int=10, eps:float=1e-5):
    baseline = np.sort(baseline)[::-1]
    production = np.sort(production)[::-1]

    min_val = min(min(baseline), min(production))
    max_val = max(max(baseline), max(production))
    median = np.median(baseline)
    std = np.std(baseline)

    if bin_num % 2 == 1:
        bins = [min_val - eps] + [median - std/6 - i*std/3 for i in range(bin_num//2 - 1, 0, -1)] + [median - std/6 + i*std/3 for i in range(bin_num//2 + 1)] + [max_val + eps]
    else:
        bins = [min_val - eps] + [median - i*std/3 for i in range(bin_num//2 - 1, 0, -1)] + [median + i*std/3 for i in range(bin_num//2)] + [max_val + eps]

    return bins


def psi(baseline:np.ndarray, production:np.ndarray, bins:list, eps:float=1e-5) -> dict:
    bin_num = len(bins) - 1
    data = {"Boundary conditions":[], "Yb":[], "Y":[], "PSI":[]}

    for i in range(bin_num):
        data["Boundary conditions"].append(f"{bins[i]:.1f}-{bins[i+1]:.1f}")
        ybi = len([x for x in baseline if x >= bins[i] and x < bins[i+1]])/len(baseline)
        data["Yb"].append(ybi if ybi != 0 else eps)
        yi = len([x for x in production if x >= bins[i] and x < bins[i+1]])/len(production)
        data["Y"].append(yi if yi != 0 else eps)
        data["PSI"].append((data["Y"][i] - data["Yb"][i]) * (np.log(data["Y"][i]) - np.log(data["Yb"][i])))

    return data
