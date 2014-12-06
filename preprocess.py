from sklearn.decomposition import PCA

def pca_whiten(data, residual):
    pca_model = PCA(n_components=data.shape[1])
    pca_model.fit(data)
    energy_dist = pca_model.explained_variance_ratio_
    total_energy = sum(energy_dist)
    target_energy = (1 - residual) * total_energy
    
    i = 0
    sum_energy = 0
    while sum_energy < target_energy:
        sum_energy += energy_dist[i]
        i += 1

    pca_model = PCA(n_components=i, whiten=True)
    pca_model.fit(data)
    return pca_model.transform(data), pca_model
