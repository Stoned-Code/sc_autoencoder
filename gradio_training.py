import os

import pandas as pd
from tqdm import tqdm

from modules.dynamic_ae import ImageAE
from modules.discriminator import PatchGAN
from data.sc_wds import Imagenet_1K
from data.transforms import SquareImageTransform
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from modules.scheduler import LRScheduler
from modules.loss import LPIPSLoss
from data.transforms import SquareMethod, TorchResize
from accelerate import Accelerator
from training_arguments import get_arguments
import gradio as gr

torch.backends.cudnn.enabled = False

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
    return img


def create_gif(images_path, output_dir="./reconstructions", filename="reconstruciont_progression.gif", duration=100, delete_images=False):
    os.makedirs(output_dir, exist_ok=True)

    paths = [ os.path.join(images_path, p) for p in os.listdir(images_path) if p.endswith(".jpg")]

    if len(paths) == 0:
        return 

    paths = sorted(paths)
    images = [Image.open(p) for p in paths]

    images[0].save(os.path.join(output_dir, filename), save_all=True, append_images=images[1:], duration=duration, loop=0)

    if delete_images:
        [os.remove(p) for p in paths]

def prepare_loss_dataframe(df: pd.DataFrame):
    """Convert your wide loss DataFrame into long format for plotting."""
    
    # Add epoch column (assuming index = epoch)
    df = df.reset_index().rename(columns={"index": "epoch"})
    df["epoch"] = df["epoch"].astype(int)
    
    # Melt into long format
    long_df = pd.melt(
        df,
        id_vars=["epoch"],
        var_name="loss_name",
        value_name="loss_value"
    )
    
    # Split into phase (train/val) and loss_type
    long_df[["phase", "loss_type"]] = long_df["loss_name"].str.split("_", n=1, expand=True)
    
    # Clean up names
    long_df["phase"] = long_df["phase"].str.capitalize()  # Train / Val
    long_df["loss_type"] = long_df["loss_type"].str.replace("_loss", "", regex=False)
    
    return long_df

def create_loss_weights(recon_loss, commit_loss, noise_loss, kl_loss, adv_loss, lpips_loss):
    loss_weights = {
        "recon_loss": recon_loss,
        "commit_loss": commit_loss,
        "noise_loss": noise_loss,
        "kl_loss": kl_loss,
        # "mmd_loss": mmd_loss,
        "adv_loss": adv_loss,
        "lpips_loss": lpips_loss
    }

    total = 0

    for k, v in loss_weights.items():
        total += v
    
    for k, v in loss_weights.items():
        loss_weights[k] = v / total if v != 0.0 else 0.0

    return loss_weights


def start_training(args, train_df, val_df, update_prompt, progress = gr.Progress()):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="bf16")
    progress(0, desc="Starting Training...")
    def train():
        # Set the device to be used for training
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.no_cuda:
            device = "cpu"
        print("Device:", device)
        # Set the discriminative and generative model paths to be loaded if they exist.
        g_model_path = args.input_g_model
        d_model_path = args.input_d_model
        if args.use_lpips:
            resize = TorchResize((64, 64))

            lpips = LPIPSLoss()
            lpips.to(device)
        
        # Load Generative model if it exists.
        if os.path.exists(g_model_path):
            g_model = ImageAE.load_checkpoint(g_model_path)
        else:
            g_model = ImageAE(args.channels, args.start_channels, args.g_depth, args.conv_bottleneck, 0.01, args.output_padding, args.unet_style, args.num_heads, args.no_bitnet)
            # g_model = ImageAE(args.latent_dims, args.channels, args.g_hidden_size, args.unflatten_shape, args.num_quantizers, 
            #                         args.codebook_size, args.no_attn, args.no_bitnet, args.unet_style, args.skip_dropout, args.num_heads, args.conv_bottleneck)

        if g_model.unet_style and not args.unet_style:
            g_model.freeze_all_unet_layers()

        g_model = torch.compile(g_model) # 

        g_model.print_parameters()

        # Load the Discriminative model if it exists.
        if os.path.exists(d_model_path):
            d_model = PatchGAN.load_checkpoint(d_model_path)
        else:
            d_model = PatchGAN(args.channels, args.d_start_dim, args.d_depth, args.d_kernel_size, args.d_padding, args.d_leaky_relu_slope)

        d_model = torch.compile(d_model)

        d_model.print_parameters()

        # Set the learning rate and weight decay.
        learning_rate = args.learning_rate 
        weight_decay = args.weight_decay


        square_method = args.square_method.upper()
        # Create a data transform object
        train_data_transform = SquareImageTransform(args.side_length, eval(f"SquareMethod.{square_method}"), denoise=args.denoise, TS=args.max_timestep)
        val_data_transform = SquareImageTransform(args.side_length, eval(f"SquareMethod.{square_method}"), denoise=args.denoise, TS=args.max_timestep, random_tile=False)
        
        # Create dataset splits.
        if args.dataset_path is None:
            train_ds = Imagenet_1K.get_from_hf("train").map(train_data_transform)
            val_ds = Imagenet_1K.get_from_hf("val", False).map(val_data_transform)
        else:
            train_ds = Imagenet_1K.get_dataset(args.dataset_path, "train").map(train_data_transform)
            val_ds = Imagenet_1K.get_dataset(args.dataset_path, "val").map(val_data_transform)
            
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

        if args.data_amt == "half":
            train_steps = int(train_steps * 0.5)
            val_steps = int(val_steps * 0.5)
        
        elif args.data_amt == "quarter":
            train_steps = int(train_steps * 0.25)
            val_steps = int(val_steps * 0.25)
        
        elif args.data_amt == "tenth":
            train_steps = int(train_steps * 0.1)
            val_steps = int(val_steps * 0.1)
        
        elif args.data_amt == "hundredth":
            train_steps = int(train_steps * 0.01)
            val_steps = int(val_steps * 0.01)

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
        loss_weights = create_loss_weights(args.recon_loss_weight, args.commit_loss_weight, args.noise_loss_weight, args.kl_loss_weight, args.adv_loss_weight, args.lpips_loss_weight)

        # Set epochs.
        epochs = args.epochs

        # Sets the patience loss key
        patience_loss = args.patience_loss

        # Sets the patience.
        # patience = args.patience

        # Sets the current patience.
        current_patience = args.patience

        # Max out the lowest loss for early stopping.
        lowest_loss = float("inf")

        noise_sigma = args.min_noise_sigma

        # Iterrate over epochs.
        for e in range(epochs):
            try:
                # Set The discriminator and generator to training model
                g_model.train()
                d_model.train()
                if args.denoise:
                    noise_sigma = lerp(clamp(args.min_noise_sigma, 0.0, 1.0), 1.0, (e + 1) / epochs)
                    train_data_transform.set_ts_sigma(noise_sigma)
                    val_data_transform.set_ts_sigma(noise_sigma)

                # Create losses dictionary for the training set.
                train_losses = {
                    "total_loss": 0
                }
                train_iteration = 0
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
                    _, recon, ae_losses = g_model(xb, yb, tb, use_skips=args.unet_style)

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
                    # accelerator.clip_grad_norm_(d_model.parameters(), max_norm=1.0)

                    d_optimizer.step()
                    d_scheduler.step()

                    # Train Generative
                    g_optimizer.zero_grad(set_to_none=True)

                    # Grab the logits from the Discriminator for the fake values
                    fake_logits = d_model(recon)

                    # Compute the loss for the fake data.
                    ae_losses["adv_loss"] = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
                    
                    # Grab LPIPS loss
                    if args.use_lpips:
                        ae_losses["lpips_loss"] = lpips(recon, yb.clone())

                    # Sum all the losses together to create the total loss and weighting them using the loss_weights object.
                    total_loss = sum([v * loss_weights[k] for k, v in ae_losses.items()])

                    # Use the total loss to run the backward pass.
                    accelerator.backward(total_loss)
                    # accelerator.clip_grad_norm_(g_model.parameters(), max_norm=1.0)

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
                    train_iteration += 1
                    yield None, None, None, None, None, train_iteration + 1, train_steps, "Training", e + 1, args.epochs
                    if train_iteration >= train_steps:
                        break

                # Create the validation lossses dictionary.
                val_losses = {
                    "total_loss": 0
                }
                
                # Set The discriminator and generator to training model
                g_model.eval()
                d_model.eval()

                # Iterrate over validation data using torch.no_grad() to avoid calculating gradients.
                with torch.no_grad():
                    val_iteration = 0

                    for xb, yb, tb in tqdm(val_loader, desc=f"Epoch {e+1}/{epochs} - Validating", total=val_steps):
                        # Set x, y, and t batches to the set device.
                        xb = xb.to(device)
                        yb = yb.to(device)
                        if not torch.all(tb == 0):
                            tb = tb.to(device)
                        else:
                            tb = None

                        # Do a forward pass using 'xb', 'yb', tb'
                        _, recon, ae_losses = g_model(xb, yb, tb, use_skips=args.unet_style)



                        # Do a forward pass to get the logits for generated data.
                        fake_logits = d_model(recon)
                        # Compute the loss for the adversarial model.
                        ae_losses["adv_loss"] = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
                        
                        # Get LPIPS loss.
                        if args.use_lpips:
                            ae_losses["lpips_loss"] = lpips(recon, yb.clone())
                        
                        # Sum all the calculated losses and weight them using the loss_weights dictionary to create the total loss.
                        total_loss = sum([v * loss_weights[k] for k, v in ae_losses.items()])
                        # Add the total loss to the validation losses dictionary.
                        val_losses["total_loss"] += total_loss.item()

                        # Iterrate through the calculated losses to add them to the validation losses dictionary.
                        for k, v in ae_losses.items():
                            if k not in val_losses:
                                val_losses[k] = 0
                            
                            val_losses[k] += v.item()

                        val_iteration += 1
                        yield None, None, None, None, None, val_iteration + 1, val_steps, "Validating", e + 1, args.epochs
                        if val_iteration >= val_steps:
                            break
                # Average out the losses for each set's losses.
                train_losses = {f"train_{k}": v / train_steps for k, v in train_losses.items()}
                val_losses = {f"val_{k}": v / val_steps for k, v in val_losses.items()}
                
                # Log the losses of the training set and the validation set.
                # data = train_losses | val_losses
                # accelerator.print({k: round(v, 5) for k, v in data.items()})

                # Log the codebook usage using the most recent returned codebook indices.
                # if indices is not None:
                #     log_codebook_usage(indices, args.codebook_size)

                # Reconstruct sample images for the Gradio UI / disk.
                img_array = None
                if args.recon_amt < 0:
                    img_array = save_reconstructions(xb[args.recon_amt:], recon[args.recon_amt:], yb[args.recon_amt:] if args.denoise else None, filename=f"reconstruction_{e:06d}.jpg")
                    save_reconstructions(xb[args.recon_amt:], recon[args.recon_amt:], yb[args.recon_amt:] if args.denoise else None, output_dir="./")

                elif args.recon_amt > 0:
                    img_array = save_reconstructions(xb[:args.recon_amt], recon[:args.recon_amt], yb[:args.recon_amt] if args.denoise else None, filename=f"reconstruction_{e:06d}.jpg")
                    save_reconstructions(xb[:args.recon_amt], recon[:args.recon_amt], yb[:args.recon_amt] if args.denoise else None, output_dir="./")
                
                is_lowest = False

                # Using the set patience_loss as the key, check if the model has improved.
                if val_losses[patience_loss] < lowest_loss:
                    lowest_loss = val_losses[patience_loss] # Set the new lowest loss.
                    accelerator.unwrap_model(g_model).save_checkpoint(args.output_g_model) # Save Generative model if there's improvement.
                    accelerator.unwrap_model(d_model).save_checkpoint(args.output_d_model) # Save the Discriminator model if there's imporovment.
                    current_patience = args.patience # Reset the current patience.
                    is_lowest = True
                
                else:
                    current_patience -= 1 # Incriment the current_patience down if there isn't improvement.

                yield train_losses, val_losses, img_array, current_patience, is_lowest, None, None, "Checking Patience", e + 1, args.epochs

                # Break the training loop if there hasn't been improvements in a while.
                if current_patience <= 0:
                    yield None, None, None, None, is_lowest, None, None, "Checking ({args.patience}) exceeded.", e + 1, args.epochs
                    print(f"Patience ({args.patience}) exceeded.")
                    break
            
            # Break the loop of CTRL+C has been pressed.
            except KeyboardInterrupt as ex:
                print("Stopping Training...")
                break
        
        create_gif("./reconstructions", delete_images=True)

        # Empty cuda cache if it's available.
        if torch.cuda.is_available() and device == "cuda":
            torch.cuda.empty_cache()

        # return df


    for t_l, v_l, i_a, c_p, i_l, step, steps, step_type, epoch, epochs in train():
        # End-of-epoch yields include real loss dicts (t_l/v_l). In-step yields pass t_l=None
        # with a step_type string like "Training" / "Validating" for progress only.
        # NOTE: step_type is never None (was the old broken check that prevented UI updates).
        if t_l is not None:
            train_df.loc[len(train_df)] = t_l
            val_df.loc[len(val_df)] = v_l

            if len(train_df.columns) != len(t_l.keys()):
                print(t_l.keys())
                print(train_df.columns)
                train_df = train_df[list(t_l.keys())]
                val_df = val_df[list(v_l.keys())]

            # Epoch complete: push plots + image; progress at 100% for this phase.
            yield prepare_loss_dataframe(train_df), prepare_loss_dataframe(val_df), train_df, val_df, i_a, update_prompt.format(step_type=step_type, epoch=epoch, epochs=epochs), 100.0
        
        else:
            yield None, None, None, None, None, update_prompt.format(step_type=step_type, epoch=epoch, epochs=epochs), (step / steps) * 100 if step is not None and steps else 0.0
        # yield t_l, v_l, i_a, c_p, i_l

    accelerator.wait_for_everyone()
    accelerator.end_training()

def create_dummy_data(columns, amt):
    import random
    df = pd.DataFrame(columns=columns)
    for i in range(amt):
        data = {c: random.random() for c in columns}
        df.loc[len(df)] = data
    
    return df

if __name__ == "__main__":

    import threading
    import time

    args = get_arguments()

    lw = create_loss_weights(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    val_cols = ["val_" + k for k in lw.keys()] + ["val_total_loss"]
    train_cols = ["train_" + k for k in lw.keys()] + ["train_total_loss"]

    raw_train_df = pd.DataFrame(columns=train_cols)
    raw_val_df = pd.DataFrame(columns=val_cols)

    # Shared state between training thread and Gradio UI
    latest_train = None
    latest_val = None
    latest_img = None
    training_started = False
    update_prompt = "# SC Autoencoder"
    progress = 0.0
    lock = threading.Lock()


    def training_worker():
        """This runs only once in a background thread"""
        global latest_train, latest_val, latest_img, raw_train_df, raw_val_df, update_prompt, progress

        print(">>> Starting training (this will only happen once)")
        for train_data, val_data, raw_train, raw_val, sample_images, prompt, step_progress in start_training(args, raw_train_df, raw_val_df, "# SC Autoencoder\n## Epoch: {epoch}/{epochs} - {step_type}"):
            if train_data is not None:
                latest_train = train_data
                latest_val = val_data
                latest_img = sample_images
                raw_train_df = raw_train
                raw_val_df = raw_val
            
            update_prompt = prompt
            progress = round(step_progress, 2)

    def start_training_once():
        global training_started
        with lock:
            if not training_started:
                training_started = True
                t = threading.Thread(target=training_worker, daemon=True)
                t.start()

    def get_latest():
        """Just returns the latest data for the UI to display.

        Use gr.skip() when a value is not ready yet so Gradio does not clear
        existing plot/image components on intermediate timer ticks.
        """
        global latest_train, latest_val, latest_img, update_prompt, progress
        return (
            latest_train if latest_train is not None else gr.skip(),
            latest_val if latest_val is not None else gr.skip(),
            latest_img if latest_img is not None else gr.skip(),
            update_prompt,
            progress if progress is not None else gr.skip(),
        )

    # ====================== UI ======================
    with gr.Blocks(title="SC Autoencoder Training") as demo:
        md = gr.Markdown("# SC Autoencoder Training")
        
        
        
        with gr.Row():
            with gr.Column():
                progress_slider = gr.Slider(label="Progress", minimum=0, maximum=100, interactive=False, value=0)
                img = gr.Image(format="jpg", label="Sample Reconstructions", height=616)
            with gr.Column():
                y_limiter = gr.Slider(label="Max Y", minimum=0, maximum=10, value=args.y_lim)

                train_plot = gr.LinePlot(
                    x="epoch",
                    y="loss_value",
                    color="loss_type",
                    title="Training Losses",
                    x_title="Epoch",
                    y_title="Loss",
                    height=300,
                    y_lim=[0.0, args.y_lim]
                )
                val_plot = gr.LinePlot(
                    x="epoch",
                    y="loss_value",
                    color="loss_type",
                    title="Validation Losses",
                    x_title="Epoch",
                    y_title="Loss",
                    height=300,
                    y_lim=[0.0, args.y_lim]
                )

                def update_y_limit(y_max):
                    if y_max is None or y_max <= 0:
                        y_max = 0.0000001
                    
                    return gr.LinePlot(
                    x="epoch",
                    y="loss_value",
                    color="loss_type",
                    title="Training Losses",
                    x_title="Epoch",
                    y_title="Loss",
                    height=300,
                    y_lim=[0.0, y_max]), gr.LinePlot(x="epoch",
                    y="loss_value",
                    color="loss_type",
                    title="Validation Losses",
                    x_title="Epoch",
                    y_title="Loss",
                    height=300,
                    y_lim=[0.0, y_max])

                y_limiter.change(
                    fn=update_y_limit,
                    inputs=y_limiter,
                    outputs=[train_plot, val_plot]
                )

        # Auto-refresh the plots + image every 10 seconds
        timer = gr.Timer(args.update_interval)
        timer.tick(fn=get_latest, outputs=[train_plot, val_plot, img, md, progress_slider])

        # Also update once when page first loads
        demo.load(fn=get_latest, outputs=[train_plot, val_plot, img, md, progress_slider])

    # === Start training exactly once when the script is launched ===
    start_training_once()

    demo.launch(server_name=args.host, server_port=args.port)#, favicon="")

    raw_train_df.to_csv("train_stats.csv", index=False)
    raw_val_df.to_csv("val_stats.csv", index=False)