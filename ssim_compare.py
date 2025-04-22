import argparse
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]

def compare_images(img1, img2, title1, title2):
    ssim_score, ssim_map = ssim(img1, img2, full=True, channel_axis=2, data_range=1.0)
    psnr_score = psnr(img1, img2, data_range=1.0)

    print(f"\nComparison: {title1} vs {title2}")
    print(f"  SSIM: {ssim_score:.4f}")
    print(f"  PSNR: {psnr_score:.2f} dB")

    return ssim_map, ssim_score, psnr_score

def visualize_comparison(img1, img2, ssim_map, title1, title2):
    diff = np.abs(img1 - img2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(np.clip(diff * 5, 0, 1))
    axes[0].set_title(f'|{title1} - {title2}| (Amplified)')
    axes[1].imshow(np.clip(ssim_map, 0, 1), cmap='viridis')
    axes[1].set_title('SSIM Map')
    axes[2].imshow(img2)
    axes[2].set_title(title2)
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare three images: input, expected, predicted")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("expected_image", help="Path to expected output image (ground truth)")
    parser.add_argument("predicted_image", help="Path to predicted output image")
    args = parser.parse_args()

    img_input = load_image(args.input_image)
    img_expected = load_image(args.expected_image)
    img_predicted = load_image(args.predicted_image)

    # 1. Predicted vs Expected
    ssim_map1, ssim1, psnr1 = compare_images(img_predicted, img_expected, "Predicted", "Expected")
    visualize_comparison(img_expected, img_predicted, ssim_map1, "Expected", "Predicted")

    # 2. Input vs Expected
    ssim_map2, ssim2, psnr2 = compare_images(img_input, img_expected, "Input", "Expected")
    visualize_comparison(img_expected, img_input, ssim_map2, "Expected", "Input")

    # 3. Input vs Predicted
    ssim_map3, ssim3, psnr3 = compare_images(img_input, img_predicted, "Input", "Predicted")
    visualize_comparison(img_predicted, img_input, ssim_map3, "Predicted", "Input")

if __name__ == "__main__":
    main()
