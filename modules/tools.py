import pathlib
import torch
import json
import os

class Reparameterizer:
    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        noise = torch.randn_like(sigma)

        return mu + noise * sigma

class ModuleTools:
    """Module Tools
    Make sure to create a 'self.config' object with all the '__init__' arguments in it.
    """
    def print_parameters(self):
        # === Parameter count ===
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{type(self).__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size ≈ {total_params * 4 / (1024**2):.1f} MB (in FP32)")


    def save(self, path):
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        config_path = path / "config.json"
        model_path = path / "model.pt"

        torch.save(self.state_dict(), model_path)

        with open(config_path, "w") as f:
            f.write(json.dumps(self.config))

        print(f"[{type(self).__name__} saved] {path}")

    @classmethod
    def load(cls, path, map_location=None):
        # config_path = pathlib.Path(path) / "config.json"
        # model_path = pathlib.Path(path) / "model.pt"

        # with open(config_path, "r") as f:
        #     config = json.loads(f.read())
        sd, config = cls.read_state_dict(path, map_location)

        model = cls(**config)
        model.load_state_dict(sd)
        print(f"[Model {cls.__name__} Loaded] {path}")
        return model

    def clamp_weights(self):
        for module in self.model.modules():
            if (hasattr(module, "weight") and module.kernel_size == (1, 1)):
                module.weight.data = torch.clamp(module.weight.data, min=0.0)
                
    def checkpoint_model(self, path_to_checkpoint, checkpoint_name):
        path_to_lpips = os.path.join(path_to_checkpoint, checkpoint_name)
        config_path = os.path.join(path_to_checkpoint, "config.json")

        with open(config_path, "w") as f:
            f.write(json.dumps(self.config))

        print(f"Saving Checkpoint to {path_to_lpips}")
        torch.save(self.model.state_dict(), path_to_lpips)

    @classmethod
    def load_checkpoint(cls, path_to_checkpoint, checkpoint_name):
        path_to_lpips = os.path.join(path_to_checkpoint, checkpoint_name)
        config_path = os.path.join(path_to_checkpoint, "config.json")

        with open(config_path, "r") as f:
            config = json.loads(f.read())

        sd = torch.load(path_to_lpips)
        model = cls(**config)
        model.load_state_dict(sd)
        return model
        

    @classmethod
    def read_state_dict(cls, path, map_location=None):
        config_path = pathlib.Path(path) / "config.json"
        model_path = pathlib.Path(path) / "model.pt"

        with open(config_path, "r") as f:
            config = json.loads(f.read())

        sd = torch.load(model_path, map_location=map_location)
        print(f"[Retreived {cls.__name__} State Dict] {path}")
        return sd, config     

    @classmethod
    def load(cls, path, map_location=None):
        # config_path = pathlib.Path(path) / "config.json"
        # model_path = pathlib.Path(path) / "model.pt"

        # with open(config_path, "r") as f:
        #     config = json.loads(f.read())
        sd, config = cls.read_state_dict(path, map_location)

        model = cls(**config)
        model.load_state_dict(sd)
        print(f"[Model {cls.__name__} Loaded] {path}")
        return model