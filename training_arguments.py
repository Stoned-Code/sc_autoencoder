import argparse

def get_arguments():
    p = argparse.ArgumentParser()

    # Data
    data = p.add_argument_group("Data", description="Arguments that changes how the data is handled.")
    data.add_argument("--side_length", type=int, default=224)
    data.add_argument("--denoise", action="store_true")
    data.add_argument("--dataset_path", type=str, default=None)
    data.add_argument("--square_method", type=str, default="CROP")
    data.add_argument("--max_timestep", type=int, default=1000)
    data.add_argument("--train_workers", type=int, default=0)
    data.add_argument("--val_workers", type=int, default=0)
    data.add_argument("--prefetch_factor", type=int, default=None)
    data.add_argument("--min_noise_sigma", type=float, default=0.01)
    data.add_argument("--recon_amt", type=int, default=3)

    # Models
    models = p.add_argument_group("Both Models", description="Arguments that modify both models' behavour.")
    models.add_argument("--channels", type=int, default=3)
    models.add_argument("--no_attn", action="store_false")
    models.add_argument("--no_bitnet", action="store_false")
    models.add_argument("--unet_style", action="store_true")
    models.add_argument("--no_cuda", action="store_true")
    models.add_argument("--conv_bottleneck", type=int, default=None)

    # Generative Model
    gen = p.add_argument_group("Generative Model", description="Arguments that modify how the generative model behaves.")
    gen.add_argument("--input_g_model", type=str, default="models/img_ae")
    gen.add_argument("--output_g_model", type=str, default="models/img_ae")
    # gen.add_argument("--latent_dims", type=int, default=512)
    # gen.add_argument("--g_hidden_size", type=int, default=25088)
    # gen.add_argument("--num_quantizers", type=int, default=8)
    # gen.add_argument("--codebook_size", type=int, default=1024)
    gen.add_argument("--num_heads", type=int, default=4)
    # gen.add_argument("--unflatten_shape", nargs="+", type=tuple, default=(128, 14, 14))
    # gen.add_argument("--skip_dropout", type=float, default=0.0)
    gen.add_argument("--start_channels", type=int, default=16)
    gen.add_argument("--g_depth", type=int, default=6)
    gen.add_argument("--output_padding", type=int, nargs="+", default=[1, 1, 1, 1, 0, 0])

    # Discriminative
    disc = p.add_argument_group("Discriminative Model", description="Arguments that modify how the discriminative model behaves.")
    disc.add_argument("--input_d_model", type=str, default="models/synthetic_prediction")
    disc.add_argument("--output_d_model", type=str, default="models/synthetic_prediction")
    disc.add_argument("--d_start_dim", type=int, default=64)
    disc.add_argument("--d_depth", type=int, default=3)
    disc.add_argument("--d_kernel_size", type=int, default=4)
    disc.add_argument("--d_padding", type=int, default=1)
    disc.add_argument("--d_leaky_relu_slope", type=float, default=0.2)

    # Training
    training = p.add_argument_group("Training", description="Arguments that modify the training behaviour.")
    training.add_argument("--learning_rate", "-lr", type=float, default=0.0001)
    training.add_argument("--weight_decay", type=float, default=0.00001)
    training.add_argument("--batch_size", "-b", type=int, default=32)
    training.add_argument("--epochs", "-e", type=int, default=10)
    training.add_argument("--patience", type=int, default=10)
    training.add_argument("--patience_loss", type=str, default="val_recon_loss")
    training.add_argument("--data_amt", type=str, choices=["full", "half", "quarter", "tenth", "hundredth"], default="full")
    training.add_argument("--gradient_accumulation_steps", type=int, default=2)

    # Loss Weights
    loss_weights = p.add_argument_group("Loss Weights", description="A set of weights to decide how each loss type affects how much each loss affects the backpropogation.")
    loss_weights.add_argument("--recon_loss_weight", type=float, default=1.0)
    loss_weights.add_argument("--commit_loss_weight", type=float, default=0.05)
    loss_weights.add_argument("--noise_loss_weight", type=float, default=0.2)
    loss_weights.add_argument("--kl_loss_weight", type=float, default=1e-5)
    # loss_weights.add_argument("--mmd_loss_weight", type=float, default=1.1)
    loss_weights.add_argument("--adv_loss_weight", type=float, default=0.1)
    loss_weights.add_argument("--lpips_loss_weight", type=float, default=0.2)
    loss_weights.add_argument("--use_lpips", action="store_true")

    # Host
    host = p.add_argument_group("Host Arguments", description="A set of arguments for the gradio server.")
    host.add_argument("--host", type=str, default=None)
    host.add_argument("--port", type=int, default=7860)
    host.add_argument("--y_lim", type=float, default=1.0)
    host.add_argument("--update_interval", type=float, default=1.0)

    args = p.parse_args()

    return args