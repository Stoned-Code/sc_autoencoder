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
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{type(self).__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size ≈ {total_params * 4 / (1024**2):.1f} MB (in FP32)")

    def save_checkpoint(self, path):
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        config_path = path / "config.json"
        model_path = path / "model.pt"

        torch.save(self.state_dict(), model_path)

        with open(config_path, "w") as f:
            f.write(json.dumps(self.config))

        print(f"[{type(self).__name__} saved] {path}")

    @classmethod
    def read_checkpoint_state_dict(cls, path, map_location=None):
        config_path = pathlib.Path(path) / "config.json"
        model_path = pathlib.Path(path) / "model.pt"

        with open(config_path, "r") as f:
            config = json.loads(f.read())

        sd = torch.load(model_path, map_location=map_location)
        print(f"[Retreived {cls.__name__} State Dict] {path}")
        return sd, config     

    @classmethod
    def load_checkpoint(cls, path, map_location=None):
        sd, config = cls.read_checkpoint_state_dict(path, map_location)

        model = cls(**config)
        model.load_state_dict(sd)
        print(f"[Model {cls.__name__} Loaded] {path}")
        return model



