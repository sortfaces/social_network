import numpy as np
from dataset import DataLoader
from sorec import SoRec

def main():
    # 1. Parameters
    NUM_USERS = 200
    NUM_ITEMS = 1000
    LATENT_DIM = 10
    MAX_ITER = 50
    LEARNING_RATE = 0.01
    LAMBDA_C = 10.0 # Importance of social network
    
    # 2. Load/Generate Data
    print("--- Loading Data ---")
    R, C, train_data, test_data = DataLoader.generate_synthetic_data(
        num_users=NUM_USERS, 
        num_items=NUM_ITEMS,
        rating_density=0.02,
        social_density=0.02
    )
    
    # 3. Initialize Model
    print("\n--- Initializing SoRec Model ---")
    model = SoRec(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        latent_dim=LATENT_DIM,
        learning_rate=LEARNING_RATE,
        lambda_c=LAMBDA_C,
        max_iter=MAX_ITER
    )
    
    # 4. Train
    print("\n--- Starting Training ---")
    model.fit(R, C, train_data, test_data)
    
    # 5. Final Evaluation
    print("\n--- Final Evaluation ---")
    rmse = model.evaluate(test_data)
    print(f"Final Test RMSE: {rmse:.4f}")
    
    # Example Prediction
    u, i, r = test_data[0]
    pred = model.predict(u, i)
    print(f"\nExample Prediction: User {u}, Item {i} -> Actual: {r}, Predicted: {pred:.2f}")

if __name__ == "__main__":
    main()
