import numpy as np
## TODO add equal freq binning

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

def kl_divergence(baseline:np.ndarray, production:np.ndarray, bins:list, eps:float=1e-5) -> dict:
    bin_num = len(bins) - 1
    data = {"Boundary conditions":[], "Yb":[], "Y":[], "KL-divergence":[]}

    for i in range(bin_num):
        data["Boundary conditions"].append(f"{bins[i]:.1f}-{bins[i+1]:.1f}")
        ybi = len([x for x in baseline if x >= bins[i] and x < bins[i+1]])/len(baseline)
        data["Yb"].append(ybi if ybi != 0 else eps)
        yi = len([x for x in production if x >= bins[i] and x < bins[i+1]])/len(production)
        data["Y"].append(yi if yi != 0 else eps)
        data["KL-divergence"].append(data["Y"][i] * (np.log(data["Y"][i]) - np.log(data["Yb"][i])))

    return data
