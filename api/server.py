import flwr as fl
import torch
import torch.nn as nn
from flwr.common import Parameters
from flwr.common import parameters_to_ndarrays
from model import Net

# Save the global model after federated learning
class SaveModelCallback(fl.server.strategy.FedAvg):
    def __init__(self, num_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds  # Store the num_rounds value

    def aggregate_fit(self, rnd, results, failures):
        """Aggregate model updates and save the global model."""
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters = aggregated_result[0]  # Extract Parameters object
        if rnd == self.num_rounds:  # After the last round
            self.save_model(aggregated_parameters)
        return aggregated_result

    def save_model(self, parameters, path="global_model.pth"):
        print("Saving the global model...")

        # Convert Flower Parameters object to list of NumPy arrays
        parameter_list = fl.common.parameters_to_ndarrays(parameters)

        # Initialize a new model to load parameters into
        model = Net()  # Instantiate a new model
        state_dict = model.state_dict()  # Get the state_dict of the model

        # Iterate through the parameters and update the model's state_dict
        updated_state_dict = {}
        for param_name, param_value in zip(state_dict.keys(), parameter_list):
            updated_state_dict[param_name] = torch.tensor(param_value, dtype=torch.float32)

        # Load the updated state_dict into the model
        model.load_state_dict(updated_state_dict)
        
        # Save the model to a file
        torch.save(model.state_dict(), path)
        print(f"Global model saved to '{path}'.")


# Define the strategy (FedAvg with our custom callback)
strategy = SaveModelCallback(num_rounds=3)  # Pass num_rounds to the strategy

# Start the Flower server and save the model after training rounds
if __name__ == "__main__":
    print("Starting Flower server...")
    
    # Start the server
    fl.server.start_server(
        server_address="127.0.0.1:8080", 
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
