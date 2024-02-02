import numpy as np
from concurrent.futures import ThreadPoolExecutor
#fonction pour déterminer la taille de sous-ensemble 
def num_random_samples_function(taille_dataset):
    
    if taille_dataset < 1000:
        return int(0.2 * taille_dataset)
    
    else:
        return int(0.05 * taille_dataset)
#fonction pour déterminer le nombre de sous-ensemble    
def num_samples_function(taille_dataset,taille_sample):
 
    return int(taille_dataset/taille_sample)
#fonction pour affecter au petit dataset 10 itérations et 5 au petit dataset (itérations pour l'update du medoids)
def max_iterations_function(taille_dataset):
    
    if taille_dataset < 1000:
        return 10  
    else:
        return 5
#fonction pour calculer la distance de Manhattan entre les point et les medoids dans un dataset, un nouveau axe est ajouté pour stocker la distance entre chaque point et un medoid
def calculate_manhattan_cost(data, medoids_indices):
    return np.sum(np.abs(data.values[:, np.newaxis, :] - data.iloc[medoids_indices].values), axis=2)
#fonction pour calculer le coût entre les point et le medoid à l'intérieur d'un cluster en utilisant la distance de Manhattan 
def calculate_cost_cluster(sample_data,cluster_data,medoid_indice):
    return np.sum(np.abs(cluster_data.values[:, np.newaxis, :] - sample_data.iloc[medoid_indice].values), axis=2)
#fonction pour calculer le coût entre les point et leurs medoids dans le  dataset complet, après l'application de clustering avec chaque configuration de sous-ensemble
def calculate_cost_num_samples(data , medoids_indices):
    distance_o_m = calculate_manhattan_cost(data,medoids_indices)
    # Sélectionner l'indice du médoid le plus proche pour chaque point à l'aide de la fonction argmin qui retourne les medoides qui ont la plus petite distance avec chaque point 
    closest_medoid_indices = np.argmin(distance_o_m , axis=1)
    cost_num_samples = np.sum(distance_o_m[np.arange(len(data)), closest_medoid_indices])
    return cost_num_samples

#fonction pour qu'un medoide passe d'un point à un autre en mesurant le coût 
def k_medoids(sample_data, medoid_indice, candidate_index,cluster_points):
    
    # Calculer le coût avant de changer le medoide
    current_cost = calculate_cost_cluster(sample_data,sample_data.iloc[cluster_points], medoid_indice).sum()
   
    #changer le medoide par un point de cluster 
    medoid_indice = candidate_index[1]
    updated_cost = calculate_cost_cluster(sample_data,sample_data.iloc[cluster_points], medoid_indice).sum()

    #si le coût augmente, retourner au premier medoide
    if updated_cost > current_cost:
        medoid_indice = candidate_index[0]
    
    return medoid_indice

#fonction pour checher àchaque itération le meilleure medoide dans chaque cluster, chaque thread va exécuter cette fonction avec le cluster qui lui est affecté
def k_medoids_k(cluster_id, associated_points, sample_data, medoid_indice):
    
    for o in associated_points:
        updated_medoid_indice = k_medoids(sample_data, medoid_indice, (cluster_id, o),associated_points)

    return updated_medoid_indice

#fonction pour appliquer PAM sur un sous-ensemble, chaque thread va exécuter cette fonction avec le sous-ensemble qui lui est affecté
def clara_num_samples(data, k, num_random_samples, max_iterations, random_indices):
    #recupèrer les données correspond aux indices choisis
    sample_data = data.iloc[random_indices]
    #tirer du sous-ensemble k indices (k medoids initiaux )
    medoids_indices = np.random.choice(num_random_samples, size=k, replace=False)

    for _ in range(max_iterations):
        #affecter chaque point au medoide le plus proche 
        labels = np.argmin(calculate_manhattan_cost(sample_data, medoids_indices), axis=1)     
        #lancer un nombre de threads égale au nombre de cluster
        with ThreadPoolExecutor(max_workers=k) as executor:

            futures = []

            for cluster_id in range(k):
                #récupérer les indices des points appartiennent au même cluster
                associated_points = np.where(labels == cluster_id)[0]
                future = executor.submit(k_medoids_k, cluster_id, associated_points, sample_data, medoids_indices[cluster_id])
                futures.append(future)# ils seront ajoutés dans l'ordre où les tâches ont été soumises

            # Récupération des résultats dans l'ordre d'origine et affectation à medoids_indices
            for cluster_id, future in enumerate(futures):
                result = future.result()
                medoids_indices[cluster_id] = result
    
    #calculer le coût de clustering sur le dataset complet, en utilisant les medoides trouvés avec le sous-ensemble
    cost_num_samples = calculate_cost_num_samples(data,random_indices[medoids_indices])

    return random_indices, medoids_indices, cost_num_samples 

def clara(data, k, num_samples, num_random_samples, max_iterations):
    
    results = []
    #chaque thread va traiter un sous-ensemble
    #lancer un nombre de threads égale au nombre de sous-ensembles
    with ThreadPoolExecutor(max_workers=num_samples) as executor:
        futures = []
        for _ in range(num_samples):
            #random choice retourne les indices du dataset d'une manière aléatoire, le nombre d'indice à choisir est la taille de sous-ensemble,un indice ne peut pas être choisi plus qu'une seule fois
            random_indices = np.random.choice(len(data), size=num_random_samples, replace=False)
            future = executor.submit(clara_num_samples, data, k, num_random_samples, max_iterations, random_indices)
            futures.append(future)

        for future in futures:
            results.append(future.result())
    
    #initialiser min_cost et best_medoids par les valeurs de la première ligne pour 
    indices, medoids, min_cost = results[0]
    best_medoids = indices[medoids]
    #choisir la configuration qui donne le plus petit coût dans le dataset complet 
    for indices, medoids, cost in results[1:]:
        if cost < min_cost:
            min_cost = cost
            best_medoids = indices[medoids]#L'objectif ici est de récupérer les indices de medoids dans le sous ensemble pour les récuperer ensuite dans le dataset entier

    #affecter chaque point au medoide finale le plus proche dans le dataset complet
    labels = np.argmin(calculate_manhattan_cost(data, best_medoids), axis=1)

    return labels, data.iloc[best_medoids]