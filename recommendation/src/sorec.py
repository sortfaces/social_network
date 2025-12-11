import numpy as np

class SoRec:
    def __init__(self, num_users, num_items, latent_dim=10, 
                 learning_rate=0.01, lambda_c=10.0, lambda_reg=0.1, 
                 max_iter=100):
        """
        SoRec: Social Recommendation Using Probabilistic Matrix Factorization
        
        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            latent_dim (int): Dimension of latent features.
            learning_rate (float): Learning rate for SGD.
            lambda_c (float): Weight for the social network regularization term.
            lambda_reg (float): Regularization parameter.
            max_iter (int): Maximum number of iterations.
        """
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.lambda_c = lambda_c
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        
        # Initialize latent matrices with small random values
        # U: User latent features (shared)
        self.U = np.random.normal(0, 0.1, (num_users, latent_dim))
        # V: Item latent features
        self.V = np.random.normal(0, 0.1, (num_items, latent_dim))
        # Z: Social latent features
        self.Z = np.random.normal(0, 0.1, (num_users, latent_dim))
        
    def sigmoid(self, x):
        """Logistic function g(x) = 1 / (1 + exp(-x))"""
        # Clip x to avoid overflow
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid: g'(x) = g(x) * (1 - g(x))"""
        s = self.sigmoid(x)
        return s * (1 - s)

    def fit(self, R, C, train_data, test_data=None):
        """
        Train the model using Stochastic Gradient Descent.
        
        Args:
            R (np.array): User-Item rating matrix.
            C (np.array): User-User social matrix.
            train_data (list): List of (u, i, r) tuples.
            test_data (list): List of (u, i, r) tuples.
        """
        # Normalize ratings to [0, 1] for training
        # Assuming ratings are 1-5. 
        # Formula: (r - 1) / (5 - 1) = (r - 1) / 4
        # We will normalize on the fly during SGD updates.
        
        # Extract social links for SGD
        social_links = []
        rows, cols = C.nonzero()
        for u, k in zip(rows, cols):
            # C_ik is usually 1 (trust) or normalized value.
            # The paper treats C_ik as a value in [0, 1].
            social_links.append((u, k, C[u, k]))
            
        print("Training started (with Logistic Function)...")
        for epoch in range(self.max_iter):
            # Shuffle data
            np.random.shuffle(train_data)
            np.random.shuffle(social_links)
            
            loss = 0
            
            # --- Step 1: Update based on Ratings ---
            for u, i, r in train_data:
                # Normalize rating r to [0, 1]
                r_norm = (r - 1.0) / 4.0
                
                # Prediction (with Sigmoid)
                dot_val = np.dot(self.U[u], self.V[i])
                pred_r = self.sigmoid(dot_val)
                
                err_r = r_norm - pred_r
                
                # Derivative of sigmoid
                # g'(x) = pred_r * (1 - pred_r)
                g_prime = pred_r * (1 - pred_r)
                
                # Gradients
                # dJ/dU = -(r - g) * g' * V + reg * U
                common_term = -err_r * g_prime
                
                grad_U = common_term * self.V[i] + self.lambda_reg * self.U[u]
                grad_V = common_term * self.U[u] + self.lambda_reg * self.V[i]
                
                # Update
                self.U[u] -= self.lr * grad_U
                self.V[i] -= self.lr * grad_V
                
                loss += 0.5 * (err_r ** 2)

            # --- Step 2: Update based on Social Links ---
            for u, k, c in social_links:
                # Prediction for social link (with Sigmoid)
                dot_val_social = np.dot(self.U[u], self.Z[k])
                pred_c = self.sigmoid(dot_val_social)
                
                err_c = c - pred_c
                
                # Derivative
                g_prime_social = pred_c * (1 - pred_c)
                
                # Gradients
                common_term_social = self.lambda_c * (-err_c * g_prime_social)
                
                grad_U_social = common_term_social * self.Z[k] + self.lambda_reg * self.U[u]
                grad_Z = common_term_social * self.U[u] + self.lambda_reg * self.Z[k]
                
                # Update
                self.U[u] -= self.lr * grad_U_social
                self.Z[k] -= self.lr * grad_Z
                
                loss += 0.5 * self.lambda_c * (err_c ** 2)
                
            # Add regularization to loss
            loss += 0.5 * self.lambda_reg * (np.sum(self.U**2) + np.sum(self.V**2) + np.sum(self.Z**2))
            
            if (epoch + 1) % 10 == 0:
                rmse = self.evaluate(test_data) if test_data else 0.0
                print(f"Epoch {epoch+1}/{self.max_iter} - Loss: {loss:.2f} - Test RMSE: {rmse:.4f}")

    def predict(self, u, i):
        """Predict rating for user u and item i."""
        if u >= self.num_users or i >= self.num_items:
            return 3.0 
        
        dot_val = np.dot(self.U[u], self.V[i])
        # Map back from [0, 1] to [1, 5]
        # val = g(x) * 4 + 1
        val = self.sigmoid(dot_val) * 4.0 + 1.0
        
        return np.clip(val, 1, 5)

    def evaluate(self, test_data):
        """Calculate RMSE on test data."""
        error_sum = 0
        count = 0
        for u, i, r in test_data:
            pred = self.predict(u, i)
            error_sum += (r - pred) ** 2
            count += 1
        return np.sqrt(error_sum / count)
