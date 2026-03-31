import torch
import torch.nn.functional as F

from .base_attack import _BaseAttacker
import matplotlib.pyplot as plt
import cvxpy as cp
from utils.breaching_utils import *
import numpy as np
import lpips
import torch
# from skimage.metrics import niqe
import piq
import torch
import torchvision.models as models
import torch.nn.functional as F
import os
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F

class AnalyticAttacker(_BaseAttacker):
    """Implements a sanity-check analytic inversion

    Only works for a torch.nn.Sequential model with input-sized FC layers."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cuda"))):
        super().__init__(model, loss_fn, cfg_attack, setup)

    def __repr__(self):
        return f"""Attacker (of type {self.__class__.__name__})."""

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        # Main reconstruction: loop starts here:
        inputs_from_queries = []
        for model, user_gradient in zip(rec_models, shared_data["gradients"]):
            idx = len(user_gradient) - 1
            for layer in list(model)[::-1]:  # Only for torch.nn.Sequential
                if isinstance(layer, torch.nn.Linear):
                    bias_grad = user_gradient[idx]
                    weight_grad = user_gradient[idx - 1]
                    layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, labels)
                    idx -= 2
                elif isinstance(layer, torch.nn.Flatten):
                    inputs = layer_inputs.reshape(shared_data["num_data_points"], *self.data_shape)
                else:
                    raise ValueError(f"Layer {layer} not supported for this sanity-check attack.")
            inputs_from_queries += [inputs]

        final_reconstruction = torch.stack(inputs_from_queries).mean(dim=0)
        reconstructed_data = dict(data=inputs, labels=labels)

        return reconstructed_data, stats

    def invert_fc_layer(self, weight_grad, bias_grad, image_positions):
        """The basic trick to invert a FC layer."""
        # By the way the labels are exactly at (bias_grad < 0).nonzero() if they are unique
        valid_classes = bias_grad != 0
        # print(valid_classes)
        intermediates = weight_grad[valid_classes, :] / bias_grad[valid_classes, None]
        # print('len(intermediates)',len(intermediates)) # determined by how many valid_class
        # print('image_positions:',image_positions)
        # print('valid_classes:',valid_classes)
        # print('weight_grad[valid_classes]:',weight_grad[valid_classes, :])

        if len(image_positions) == 0:      # this is true
            reconstruction_data = intermediates
        elif len(image_positions) == 1:
            reconstruction_data = intermediates.mean(dim=0)
        else:
            reconstruction_data = intermediates[image_positions]
        return reconstruction_data

class ImprintAttacker(AnalyticAttacker):
    # @torch.no_grad()    


    def reconstruct(self, server_payload, shared_data, server_secrets,Psi,A,defense,dryrun=False):
        """This is somewhat hard-coded for images, but that is not a necessity."""
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        if "ImprintBlock" in server_secrets.keys():
            weight_idx = server_secrets["ImprintBlock"]['weight_idx_second']
            bias_idx = server_secrets["ImprintBlock"]['bias_idx_second']
            data_shape = server_secrets["ImprintBlock"]["shape"]
        else:
            raise ValueError(f"No imprint hidden in model {rec_models[0]} according to server.")

        bias_grad = shared_data["gradients"][0][bias_idx].clone()
        weight_grad = shared_data["gradients"][0][weight_idx].clone()
       
        print('before weight_grad:',weight_grad.shape)
        print('before bias_grad:',bias_grad.shape)
        print('before sub bias_grad = 0:',(bias_grad == 0).sum().item())
        if server_secrets["ImprintBlock"]["structure"] == "cumulative":
            for i in reversed(list(range(1, weight_grad.shape[0]))):
                weight_grad[i] -= weight_grad[i - 1]
                bias_grad[i] -= bias_grad[i - 1]

        
        print('after sub bias_grad = 0:', (bias_grad == 0).sum().item())
        layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, [])
        

      
        # Compute x = W_pinv @ (z - b)
        # If W is not full-rank (i.e., some rows/columns are linearly dependent), W^+ won't recover the original x exactly
        B = layer_inputs.shape[0]
        alpha_hats = []
        z=layer_inputs


        A_cpu = A.detach().cpu().numpy()
        z_cpu = z.detach().cpu().numpy()

      
        # ##############################
        valid_alpha_hats = []
        for i in range(B):
            z = layer_inputs[i]
            z_cpu = z.detach().cpu().numpy()

            # mask = z_cpu != 0
            mask = z_cpu > 0
            z_reduced = z_cpu[mask]
            mask_A = mask.reshape(-1)
            # A_reduced = A_cpu[mask_A, :]
            A_reduced = A_cpu[:,mask_A]
            
            
            alpha = cp.Variable(A_cpu.shape[0])
            objective = cp.Minimize(cp.norm1(alpha))

                # Define constraints based on defense
            try:
                if defense == 'None':
                    constraints = [A_reduced.T @ alpha == z_reduced]
                    problem = cp.Problem(objective, constraints)
                    problem.solve()  # Use default solver
                else:
                    constraints = [A_reduced.T @ alpha == z_reduced]
                    problem = cp.Problem(objective, constraints)
                    problem.solve(solver=cp.SCS, verbose=False)  # You can try other solvers like 'OSQP', 'ECOS' etc.

                # Store result
                if alpha.value is not None:
                    alpha_hats.append(alpha.value)
                    print(f"✅ Success to find a solution at index {i}")
                else:
                    print(f"❌ Solver failed to find a solution at index {i}")
                    alpha_hats.append(None)

            except Exception as e:
                print(f"❌ Solver exception at index {i}: {e}")
                alpha_hats.append(None)


        # Convert all α̂ to a single tensor [B, D]
        print(f"Number of alpha_hats: {len(alpha_hats)}")
        for i, a in enumerate(alpha_hats):
            print(f"alpha_hats[{i}]: type={type(a)}, shape={getattr(a, 'shape', 'N/A')}")
        # Filter out invalid or mismatched alpha_hats
        
        ref_shape = None

        for i, a in enumerate(alpha_hats):
            if isinstance(a, np.ndarray):
                if ref_shape is None:
                    ref_shape = a.shape  # Set reference shape
                if a.shape == ref_shape:
                    valid_alpha_hats.append(a)
                else:
                    print(f"Skipping alpha_hats[{i}]: shape mismatch {a.shape} vs {ref_shape}")
            else:
                print(f"Skipping alpha_hats[{i}]: invalid type {type(a)}")

        # Proceed if there are valid entries
        if valid_alpha_hats:
            alpha_hats_torch = torch.tensor(np.stack(valid_alpha_hats), dtype=Psi.dtype, device=Psi.device)
            x_hat = torch.matmul(alpha_hats_torch, Psi.t())
            B = len(valid_alpha_hats)
        else:
            raise ValueError("No valid alpha_hats to process.")
        # Make sure the directory exists
        save_dir = "x_hat"
        os.makedirs(save_dir, exist_ok=True)

        # Generate a timestamp ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # File name with timestamp
        file_name = f"x_hat_{timestamp}.pt"
        file_path = os.path.join(save_dir, file_name)

        # Save the tensor
        torch.save(x_hat, file_path)

        # Print the saved file name
        print(f"Saved x_hat to: {file_path}")
        torch.save(x_hat, "x_hat.pt")
        # print(x_hat.shape)
        

        if "decoder" in server_secrets["ImprintBlock"].keys():
            inputs = server_secrets["ImprintBlock"]["decoder"](layer_inputs)
        else:
            inputs = x_hat.reshape(B, *data_shape)[:, :3, :, :]
        if weight_idx > 0:  # An imprint block later in the network:
            inputs = torch.nn.functional.interpolate(
                inputs, size=self.data_shape[1:], mode="bicubic", align_corners=False
            )
        inputs = torch.max(torch.min(inputs, (1 - self.dm.to(inputs.device)) / self.ds.to(inputs.device)), -self.dm.to(inputs.device) / self.ds.to(inputs.device))
        # inputs_denorm = undo_normalize(inputs, self.dm, self.ds).clamp(0, 1)  # In [0, 1]
        batch_size=len(labels)
        if len(labels) >= inputs.shape[0]:
            # Fill up with zero if not enough data can be found:
            missing_entries = torch.zeros(len(labels) - inputs.shape[0], *self.data_shape, **self.setup)
            inputs = torch.cat([inputs, missing_entries], dim=0)
        else:
            print(f"Initially produced {inputs.shape[0]} hits.")
            
             # Cut additional hits:
            lpips_model = lpips.LPIPS(net='alex').to(inputs.device).eval()

            threshold = 0.1  # Tune this for sensitivity

            unique_indices = []

            for idx in range(inputs.shape[0]):
                candidate_img = inputs[idx].unsqueeze(0)  # shape [1, C, H, W]
                is_duplicate = False

                for u_idx in unique_indices:
                    unique_img = inputs[u_idx].unsqueeze(0)
                    with torch.no_grad():
                        dist = lpips_model(candidate_img, unique_img).item()
                    if dist < threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_indices.append(idx)
            
            
            inputs = inputs[unique_indices]
            
            print(f"Reduced to {inputs.shape[0]} unique images after LPIPS filtering.")
    
        if inputs.shape[0] < batch_size:
            
            print(f"Found only {inputs.shape[0]} samples. Padding...")
        
            missing_entries = torch.zeros(batch_size - inputs.shape[0], *self.data_shape, **self.setup)
           
            inputs = torch.cat([inputs, missing_entries], dim=0)

            print(f"Final have {inputs.shape[0]} hits.")
            assert inputs.shape[0] == labels.shape[0], "inputs and labels length mismatch after slicing!"



        reconstructed_data = dict(data=inputs, labels=labels)
        return reconstructed_data, stats
    
