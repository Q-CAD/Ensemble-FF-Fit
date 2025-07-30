from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import sys

def sfpd(structure1, structure2):
    ssf = SiteStatsFingerprint(
            CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0),
            stats=('mean', 'std_dev', 'minimum', 'maximum'))

    s1_feature, s2_feature = ssf.featurize(structure1), ssf.feature(structure2)
    sfpd = np.linalg.norm(np.subtract(s1_feature, s2_feature))
    
    return sfpd

def dissimilarity_matrix(structures_list, method='sfpd'):
    dissimilarity_matrix = np.zeros((len(structures_list), len(structures_list)))
    for i1, s1 in enumerate(structures_list):
        for i2, s2 in enumerate(structures_list):
            if i2 <= i1:
                pass
            else:
                if method == 'sfpd':
                    s_sfpd = sfpd(s1, s2)
                    dissimilarity_matrix[i1, i2] = s_sfpd
                    dissimilarity_matrix[i2, i1] = s_sfpd
                else:
                    print('Only SFPD currently supported; exiting')
                    sys.exit(1)      

    return dissimilarity_matrix  

def cluster_objects(dissimilarity_matrix, dissimilarity_threshold=0.5):
    # Perform hierarchical clustering based on the dissimilarity
    Z = linkage(pdist(dissimilarity_matrix), method='single')
    # Group the objects into clusters using the similarity threshold
    clusters = fcluster(Z, t=dissimilarity_threshold, criterion='distance')

    return clusters

def choose_cluster_reprensentatives(structures_list, clusters, dissimilarity_matrix):
    '''Return a representative structure closest to the cluster center, as well
       as the structures it represents '''
    
    representative = []
    represented = []

    for cluster_num in list(np.unique(clusters)):
        use_structure_inds = [i for i, cluster in enumerate(clusters) if cluster == cluster_num]
        avg_dissimilarities = []
        for i in use_structure_inds:
            avg_dissimilarity = np.mean([dissimilarity_matrix[i][j] for j in use_structure_inds if i != j])
            avg_dissimilarities.append(avg_dissimilarity)
        representative_structure = structures_list[use_structures_inds[np.argmin(avg_dissimilarities)]]
        representative.append(representative_structure)
        represented_structures = [structures_list[use_structure_inds[i]] for i in range(len(use_structure_inds)) if i != np.argmin(avg_dissimilarities)] 
        represented.append(represented_structure)

    return representative, represented

