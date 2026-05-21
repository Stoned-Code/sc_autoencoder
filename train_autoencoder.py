import os

from tqdm import tqdm

from modules.dynamic_ae import DynamicAutoencoder2D
from modules.discriminator import PatchGAN
from data.sc_wds import SCWebDatasets
from data.transforms import SquareImageTransform
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
from PIL import Image
from modules.scheduler import LRScheduler
from data.transforms import SquareMethod
from accelerate import Accelerator
from training_arguments import get_arguments

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolate on the scale given by a to b, using t as the point on that scale."""
    return (1 - t) * a + t * b


def clamp(x, a, b):
    if x < a:
        return a
    elif x > b:
        return b
    else:
        return x


def save_reconstructions(originals, recons, expected=None, output_dir = "./reconstructions", filename="reconstruction.jpg"):
    """
    originals: list of torch.Tensor  (usually shape [C, H, W] or [B, C, H, W])
    recons:    list of torch.Tensor (same shape as originals)
    expected:  list of torch.Tensor (same shape as originals and recons)
    """
    
    os.makedirs(output_dir, exist_ok=True)

    # Make sure we have batches of images in the common [C, H, W] format
    originals = [x.squeeze(0) if x.ndim == 4 else x for x in originals]  # remove batch dim if present
    recons     = [x.squeeze(0) if x.ndim == 4 else x for x in recons]
    if expected is not None:
        expected = [x.squeeze(0) if x.ndim == 4 else x for x in expected]

    # Stack images horizontally
    combined_original = torch.cat(originals, dim=2)   # → [C, H, W_total]
    combined_recons   = torch.cat(recons, dim=2)
    if expected is not None:
        combined_expected = torch.cat(expected, dim=2)

    # Stack original row above reconstruction row
    if expected is not None:
        combined = torch.cat([combined_original, combined_recons, combined_expected], dim=1)  # → [C, H×2, W_total]
    else:
        combined = torch.cat([combined_original, combined_recons], dim=1)

    img_array = (combined * 255).byte().permute(1, 2, 0).cpu().numpy()

    img = Image.fromarray(img_array)
    img.save(os.path.join(output_dir, filename))


def create_gif(images_path, output_dir="./reconstructions", filename="reconstruciont_progression.gif", duration=100, delete_images=False):
    os.makedirs(output_dir, exist_ok=True)

    paths = [ p for p in os.listdir(images_path) if p.endswith(".jpg")]
    paths = sorted(paths)
    images = [Image.open(p) for p in paths]

    images[0].save(os.path.join(output_dir, filename), save_all=True, append_images=images[1:], duration=duration, loop=0)

    if delete_images:
        [os.remove(p) for p in paths]


def create_loss_weights(recon_loss, commit_loss, noise_loss, kl_loss, adv_loss, mmd_loss):
    return {
        "recon_loss": recon_loss,
        "commit_loss": commit_loss,
        "noise_loss": noise_loss,
        "kl_loss": kl_loss,
        "mmd_loss": mmd_loss,
        "adv_loss": adv_loss
    }


def log_codebook_usage(indices, codebook_size, prefix=""):
    """Log how many unique codes are being used"""
    indices = indices.flatten()
    unique_codes = torch.unique(indices)
    usage_count = torch.bincount(indices, minlength=codebook_size)
    
    active_codes = (usage_count > 0).sum().item()
    max_usage = usage_count.max().item()
    min_usage = usage_count.min().item()
    
    accelerator.print(f"{prefix}Codebook Usage:")
    accelerator.print(f"  Unique Codes     : {len(unique_codes)}")
    accelerator.print(f"  Active codes     : {active_codes}/{codebook_size} ({active_codes/codebook_size*100:.2f}%)")
    accelerator.print(f"  Most used code   : {max_usage} times")
    accelerator.print(f"  Least used code  : {min_usage} times")
    accelerator.print(f"  Avg usage per code: {usage_count.float().mean().item():.2f}")


def train(args):
    # Set the device to be used for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.no_cuda:
        device = "cpu"
    print("Device:", device)
    # Set the discriminative and generative model paths to be loaded if they exist.
    g_model_path = args.input_g_model
    d_model_path = args.input_d_model

    # Load Generative model if it exists.
    if os.path.exists(g_model_path):
        g_model = DynamicAutoencoder2D.load_checkpoint(g_model_path)
    else:
        g_model = DynamicAutoencoder2D(args.latent_dims, args.channels, args.g_hidden_size, args.unflatten_shape, args.num_quantizers, 
                                args.codebook_size, args.no_attn, args.no_bitnet, args.unet_style, args.skip_dropout, args.num_heads, args.conv_bottleneck)

    g_model.print_parameters()

    # Load the Discriminative model if it exists.
    if os.path.exists(d_model_path):
        d_model = PatchGAN.load_checkpoint(d_model_path)
    else:
        d_model = PatchGAN(args.channels, args.d_start_dims, args.d_depth, args.d_kernel_size, args.d_padding, args.d_leaky_relu_slope)

    d_model.print_parameters()

    # Set the learning rate and weight decay.
    learning_rate = args.learning_rate 
    weight_decay = args.weight_decay

    # Create the SC_ASMR object to grab the ASMR data from.
    asmr = SCWebDatasets(args.dataset_path)

    square_method = args.square_method.upper()
    # Create a data transform object
    train_data_transform = SquareImageTransform(args.side_length, eval(f"SquareMethod.{square_method}"), denoise=args.denoise, TS=args.max_timestep)
    val_data_transform = SquareImageTransform(args.side_length, eval(f"SquareMethod.{square_method}"), denoise=args.denoise, TS=args.max_timestep, random_tile=False)
    # Create dataset splits.
    train_ds = asmr.get_datasets(args.datasets, "train").map(train_data_transform)
    val_ds = asmr.get_datasets(args.datasets, "val", False).map(val_data_transform)

    # Print the split lengths.
    print("Training Samples:", len(train_ds))
    print("Validation Samples:", len(val_ds))

    # Set the batch size.
    batch_size = args.batch_size

    # Turn datasets into dataloaders.
    train_loader = DataLoader(train_ds, batch_size, num_workers=args.train_workers, prefetch_factor=args.prefetch_factor, persistent_workers=args.train_workers > 0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, num_workers=args.val_workers, prefetch_factor=args.prefetch_factor, persistent_workers=args.val_workers > 0, pin_memory=True)

    # Grab the amount of steps for each data loader.
    train_steps = len(train_loader)
    val_steps = len(val_loader)

    # Create the optimizers for both models.
    g_optimizer = optim.AdamW(g_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    d_optimizer = optim.AdamW(d_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Create the learning rate scheduler for both models.
    g_scheduler = LRScheduler(g_optimizer, args.learning_rate, args.epochs * train_steps, 5 * train_steps)
    d_scheduler = LRScheduler(d_optimizer, args.learning_rate, args.epochs * train_steps, 5 * train_steps)

    # Prepare everything for training.
    g_model, d_model, g_optimizer, g_optimizer, g_scheduler, d_scheduler, train_loader, val_loader = accelerator.prepare(
        g_model, d_model, g_optimizer, g_optimizer, g_scheduler, d_scheduler, train_loader, val_loader
    )


    total_training_iterations = len(train_loader) * args.epochs
    accelerator.print("Training for {} iterations".format(total_training_iterations))

    # Create the loss weights for the training losses.
    loss_weights = create_loss_weights(args.recon_loss_weight, args.commit_loss_weight, args.noise_loss_weight, args.kl_loss_weight, args.adv_loss_weight, args.mmd_loss_weight)

    # Set epochs.
    epochs = args.epochs

    # Set The discriminator and generator to training model
    g_model.train()
    d_model.train()

    # Sets the patience loss key
    patience_loss = args.patience_loss

    # Sets the patience.
    # patience = args.patience

    # Sets the current patience.
    current_patience = args.patience

    # Max out the lowest loss for early stopping.
    lowest_loss = float("inf")

    # Iterrate over epochs.
    for e in range(epochs):
        try:
            if args.denoise:
                noise_sigma = lerp(clamp(args.min_noise_sigma), 1.0, (e + 1) / epochs)
                train_data_transform.set_ts_sigma(noise_sigma)
                val_data_transform.set_ts_sigma(noise_sigma)

            # Create losses dictionary for the training set.
            train_losses = {
                "total_loss": 0
            }

            # Iterrate over training data.
            for xb, yb, tb in tqdm(train_loader, desc=f"Epoch {e + 1}/{epochs} - Training", total=train_steps):
                xb = xb.to(accelerator.device) # Move 'x' values to the set device.
                yb = yb.to(accelerator.device) # Move 'y' values to the set device.

                # If all timestep values are zero, move them over to set device, otherwise set to None.
                if not torch.all(tb == 0):
                    tb = tb.to(accelerator.device)
                else:
                    tb = None
                
                # Do a forward pass with 'xb', 'yb' and 'tb' as inputs.
                _, recon, _, ae_losses = g_model(xb, yb, tb)

                # Train Discriminator
                d_optimizer.zero_grad(set_to_none=True)

                # Grab the logits for the real and generated data
                real_logits = d_model(yb)
                fake_logits = d_model(recon.detach())

                # Compute the loss for the backward pass of the Discriminator.
                d_loss = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + \
                        F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))

                # Do the backward pass.
                accelerator.backward(d_loss)
                accelerator.clip_grad_norm_(d_model.parameters(), max_norm=1.0)

                d_optimizer.step()
                d_scheduler.step()

                # Train Generative
                g_optimizer.zero_grad(set_to_none=True)

                # Grab the logits from the Discriminator for the fake values
                fake_logits = d_model(recon)

                # Compute the loss for the fake data.
                ae_losses["adv_loss"] = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

                # Sum all the losses together to create the total loss and weighting them using the loss_weights object.
                total_loss = sum([v * loss_weights[k] for k, v in ae_losses.items()])

                # Use the total loss to run the backward pass.
                accelerator.backward(total_loss)
                accelerator.clip_grad_norm_(g_model.parameters(), max_norm=1.0)

                g_optimizer.step()
                g_scheduler.step()

                # Set the add the total loss to the train_losses dictionary.
                train_losses["total_loss"] += total_loss.item()

                # Add other calculated losses to the losses dictionary.
                for k, v in ae_losses.items():
                    # Create the key in the losses dictionary if it isn't there already.
                    if k not in train_losses:
                        train_losses[k] = 0
                    
                    train_losses[k] += v.item()
            
            # Create the validation lossses dictionary.
            val_losses = {
                "total_loss": 0
            }

            # Iterrate over validation data using torch.no_grad() to avoid calculating gradients.
            with torch.no_grad():
                for xb, yb, tb in tqdm(val_loader, desc=f"Epoch {e+1}/{epochs} - Validating"):
                    # Set x, y, and t batches to the set device.
                    xb = xb.to(device)
                    yb = yb.to(device)
                    if not torch.all(tb == 0):
                        tb = tb.to(device)
                    else:
                        tb = None

                    # Do a forward pass using 'xb', 'yb', tb'
                    _, recon, indices, ae_losses = g_model(xb, yb, tb)

                    # Do a forward pass to get the logits for generated data.
                    fake_logits = d_model(recon)
                    # Compute the loss for the adversarial model.
                    ae_losses["adv_loss"] = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
                    
                    # Sum all the calculated losses and weight them using the loss_weights dictionary to create the total loss.
                    total_loss = sum([v * loss_weights[k] for k, v in ae_losses.items()])
                    # Add the total loss to the validation losses dictionary.
                    val_losses["total_loss"] += total_loss.item()

                    # Iterrate through the calculated losses to add them to the validation losses dictionary.
                    for k, v in ae_losses.items():
                        if k not in val_losses:
                            val_losses[k] = 0
                        
                        val_losses[k] += v.item()

            # Average out the losses for each set's losses.
            train_losses = {f"train_{k}": v / train_steps for k, v in train_losses.items()}
            val_losses = {f"val_{k}": v / val_steps for k, v in val_losses.items()}
            
            # Log the losses of the training set and the validation set.
            data = train_losses | val_losses
            accelerator.print({k: round(v, 5) for k, v in data.items()})

            # Log the codebook usage using the most recent returned codebook indices.
            if indices is not None:
                log_codebook_usage(indices, args.codebook_size)

            # Reconstruct an audio file then save it for listening.
            save_reconstructions(xb[-3:], recon[-3:], yb[-3:], filename=f"reconstruction_{e:06d}.jpg")
            save_reconstructions(xb[-3:], recon[-3:], yb[-3:], output_dir="./")

            # Using the set patience_loss as the key, check if the model has improved.
            if val_losses[patience_loss] < lowest_loss:
                lowest_loss = val_losses[patience_loss] # Set the new lowest loss.
                accelerator.unwrap_model(g_model).save_checkpoint(args.output_g_model) # Save Generative model if there's improvement.
                accelerator.unwrap_model(d_model).save_checkpoint(args.output_d_model) # Save the Discriminator model if there's imporovment.
                current_patience = args.patience # Reset the current patience.
            
            else:
                current_patience -= 1 # Incriment the current_patience down if there isn't improvement.
            
            # Break the training loop if there hasn't been improvements in a while.
            if current_patience <= 0:
                print(f"Patience ({args.patience}) exceeded.")
                break
        
        # Break the loop of CTRL+C has been pressed.
        except KeyboardInterrupt as ex:
            print("Stopping Training...")
            break
    
    create_gif("./reconstructions", delete_iamges=True)

    # Empty cuda cache if it's available.
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Grab arguments for training
    args = get_arguments()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    train(args)

    accelerator.wait_for_everyone()
    accelerator.end_training()