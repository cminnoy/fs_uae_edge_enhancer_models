import argparse
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


def load_image_tensor(path, device):
    img = Image.open(path).convert("RGB")
    tensor = ToTensor()(img).unsqueeze(0).to(device)
    return tensor


def compute_metrics(output_tensor, target_tensor):
    # clamp to [0,1] and detach from graph
    out = output_tensor.clamp(0, 1).cpu().squeeze(0).detach()
    tgt = target_tensor.clamp(0, 1).cpu().squeeze(0).detach()
    # L1 loss
    l1 = F.l1_loss(out, tgt).item()
    # SSIM: compute per-channel and average
    out_np = out.permute(1, 2, 0).numpy()
    tgt_np = tgt.permute(1, 2, 0).numpy()
    ssim_vals = []
    for ch in range(out_np.shape[2]):
        val = compare_ssim(
            tgt_np[:, :, ch], out_np[:, :, ch], data_range=1.0
        )
        ssim_vals.append(val)
    ssim = float(np.mean(ssim_vals))
    return l1, ssim


def process_image(model, input_path, output_path, expected_path=None, num_passes=1, device='cuda'):
    # load input
    input_tensor = load_image_tensor(input_path, device)
    # iterative inference
    out_tensor = input_tensor
    for _ in range(num_passes):
        with torch.no_grad():
            out_tensor = model(out_tensor)
    out_tensor = out_tensor.clamp(0, 1)
    # save output
    output_image = ToPILImage()(out_tensor.cpu().squeeze(0))
    output_image.save(output_path)

    # metrics
    if expected_path:
        if os.path.isfile(expected_path):
            target_tensor = load_image_tensor(expected_path, device)
            l1, ssim = compute_metrics(out_tensor, target_tensor)
            print(f"Metrics for {os.path.basename(input_path)}: L1={l1:.6f}, SSIM={ssim:.6f}")
        else:
            print(f"Expected file '{expected_path}' not found or not a file.")


def infer(model_path, input_path, output_path, expected_path=None, num_passes=1, device='cuda'):
    # load model
    model = torch.load(model_path, weights_only=False)
    model = torch.compile(model)
    model = model.to(device)
    model.eval()

    # single file
    if os.path.isfile(input_path):
        if expected_path and os.path.isdir(expected_path):
            exp_file = os.path.join(expected_path, os.path.basename(input_path))
        else:
            exp_file = expected_path
        process_image(model, input_path, output_path, exp_file, num_passes, device)

    # directory
    elif os.path.isdir(input_path):
        # prepare output directory
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        elif not os.path.isdir(output_path):
            raise ValueError(
                f"Output path '{output_path}' must be a directory when input is a directory."
            )
        # expected directory if provided
        exp_dir = expected_path if expected_path and os.path.isdir(expected_path) else None
        # process each image
        for filename in os.listdir(input_path):
            if filename.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
            ):
                in_file = os.path.join(input_path, filename)
                out_file = os.path.join(output_path, filename)
                exp_file = os.path.join(exp_dir, filename) if exp_dir else None
                print(f"Processing: {in_file} -> {out_file}")
                process_image(
                    model, in_file, out_file, exp_file, num_passes, device
                )
    else:
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on one image or a directory of images, with optional metrics."
    )
    parser.add_argument(
        "model", help="Path to the model file (e.g., full_picomodel.pt)"
    )
    parser.add_argument(
        "input", help="Path to input image or directory"
    )
    parser.add_argument(
        "output", help="Path to save output image or directory"
    )
    parser.add_argument(
        "-e", "--expected",
        help="Path to expected output image or directory",
        default=None
    )
    parser.add_argument(
        "-p", "--passes",
        type=int,
        default=1,
        help="Number of inference passes per image (default: 1)"
    )
    args = parser.parse_args()
    infer(
        args.model,
        args.input,
        args.output,
        expected_path=args.expected,
        num_passes=args.passes
    )

