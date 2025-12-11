import numpy as np

class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def generate_synthetic_data(num_users=100, num_items=500, rating_density=0.05, social_density=0.03):
        """
        Generates synthetic data for testing the SoRec model.
        
        Returns:
            R (np.array): User-Item rating matrix (num_users x num_items). 0 means missing.
            C (np.array): User-User social matrix (num_users x num_users). 0 means no link.
            train_data (list): List of (user_id, item_id, rating) tuples.
            test_data (list): List of (user_id, item_id, rating) tuples.
        """
        print(f"Generating synthetic data: {num_users} users, {num_items} items...")
        
        # 1. Generate Ratings Matrix R
        R = np.zeros((num_users, num_items))
        all_ratings = []
        
        for u in range(num_users):
            for i in range(num_items):
                if np.random.random() < rating_density:
                    # Rating between 1 and 5
                    r = np.random.randint(1, 6)
                    R[u, i] = r
                    all_ratings.append((u, i, r))
        
        # Split into train/test
        np.random.shuffle(all_ratings)
        split_idx = int(len(all_ratings) * 0.8)
        train_data = all_ratings[:split_idx]
        test_data = all_ratings[split_idx:]
        
        # Reconstruct R_train for the matrix factorization (test entries should be 0)
        R_train = np.zeros((num_users, num_items))
        for u, i, r in train_data:
            R_train[u, i] = r
            
        # 2. Generate Social Matrix C
        # C_ik represents the trust/relationship from user i to user k
        # The paper uses normalized out-degree: C_ik = C_ik / sum(C_i*)
        C = np.zeros((num_users, num_users))
        for u1 in range(num_users):
            degree = 0
            neighbors = []
            for u2 in range(num_users):
                if u1 != u2 and np.random.random() < social_density:
                    neighbors.append(u2)
            
            if len(neighbors) > 0:
                val = 1.0 / len(neighbors)
                for u2 in neighbors:
                    C[u1, u2] = val
        
        print(f"Data generated. Train samples: {len(train_data)}, Test samples: {len(test_data)}")
        print(f"Social links: {np.count_nonzero(C)}")
        
        return R_train, C, train_data, test_data
