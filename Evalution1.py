import numpy as np
import math
from PIL import Image
import sys
import os

# --- Compatibility wrapper for skimage PSNR & SSIM ---
_psnr_func = None
_ssim_func = None
_ssim_uses_channel_axis = False  # whether structural_similarity accepts channel_axis

try:
    # Try modern alias first (skimage >= 0.20+)
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    _psnr_func = peak_signal_noise_ratio
    _ssim_func = structural_similarity
    _ssim_uses_channel_axis = True
except Exception:
    try:
        # Older name (some versions)
        from skimage.metrics import peak_signal_to_noise_ratio, structural_similarity
        _psnr_func = peak_signal_to_noise_ratio
        _ssim_func = structural_similarity
        _ssim_uses_channel_axis = False
    except Exception:
        _psnr_func = None
        _ssim_func = None
        _ssim_uses_channel_axis = False


# --- Fallback PSNR implementation ---

def psnr_fallback(a: np.ndarray, b: np.ndarray, data_range: float = 255.0):
    """Manual PSNR calculation (dB) using MSE. Works for uint8 arrays."""
    if a.shape != b.shape:
        raise ValueError("Input shapes must match for PSNR calculation.")
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_val = float(data_range)
    return 10.0 * math.log10((max_val ** 2) / mse)


# --- Image Metrics (PSNR & SSIM) ---

def calculate_psnr(img1_path, img2_path):
    """Calculates the PSNR between two images (in dB)."""
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if img1.size != img2.size:
            print(f"Warning: Image sizes differ. Resizing '{img2_path}' to match '{img1_path}'.")
            img2 = img2.resize(img1.size)

        img1_arr = np.asarray(img1).astype(np.uint8)
        img2_arr = np.asarray(img2).astype(np.uint8)

        if _psnr_func is not None:
            try:
                psnr_value = _psnr_func(img1_arr, img2_arr, data_range=255)
            except TypeError:
                psnr_value = _psnr_func(img1_arr, img2_arr)
        else:
            psnr_value = psnr_fallback(img1_arr, img2_arr, data_range=255)

        if math.isinf(psnr_value):
            return float('inf')
        return float(psnr_value)

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during PSNR calculation: {e}")
        return None


def calculate_ssim(img1_path, img2_path):
    """Calculates the SSIM between two images."""
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if img1.size != img2.size:
            print(f"Warning: Image sizes differ. Resizing '{img2_path}' to match '{img1_path}'.")
            img2 = img2.resize(img1.size)

        img1_arr = np.asarray(img1).astype(np.uint8)
        img2_arr = np.asarray(img2).astype(np.uint8)

        if _ssim_func is None:
            print("Warning: structural_similarity not available.")
            return None

        try:
            if _ssim_uses_channel_axis:
                ssim_value = _ssim_func(img1_arr, img2_arr, channel_axis=2, data_range=255)
            else:
                ssim_value = _ssim_func(img1_arr, img2_arr, multichannel=True, data_range=255)
        except TypeError:
            try:
                ssim_value = _ssim_func(img1_arr, img2_arr, channel_axis=2, data_range=255)
            except Exception:
                ssim_value = _ssim_func(img1_arr, img2_arr, multichannel=True, data_range=255)

        return float(ssim_value)

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during SSIM calculation: {e}")
        return None


# --- Data Metric (BER) ---

def message_to_binary(message):
    """Converts a string message to a binary string using UTF-8 bytes."""
    if isinstance(message, str):
        b = message.encode('utf-8')
    elif isinstance(message, (bytes, bytearray)):
        b = bytes(message)
    else:
        raise TypeError("message must be str, bytes, or bytearray")
    return ''.join(format(byte, '08b') for byte in b)


def calculate_ber(original_message, extracted_message):
    """Calculates Bit Error Rate (BER) between two text messages."""
    if original_message is None:
        original_message = ''
    if extracted_message is None:
        extracted_message = ''

    if original_message == '' and extracted_message == '':
        print("Both messages are empty. BER = 0.0")
        return 0.0
    if original_message == '':
        print("Original message empty; treating all extracted bits as errors.")
        return 1.0
    if extracted_message == '':
        print("Extracted message empty; treating all original bits as errors.")
        return 1.0

    bin_orig = message_to_binary(original_message)
    bin_ext = message_to_binary(extracted_message)

    len_orig = len(bin_orig)
    len_ext = len(bin_ext)
    max_len = max(len_orig, len_ext)

    if len_orig < max_len:
        bin_orig = bin_orig.ljust(max_len, '0')
    if len_ext < max_len:
        bin_ext = bin_ext.ljust(max_len, '0')

    error_bits = sum(1 for i in range(max_len) if bin_orig[i] != bin_ext[i])

    ber = error_bits / max_len if max_len else 0.0
    return ber


# --- MAIN MENU ---

def main_menu():
    """Interactive menu for metrics."""
    while True:
        print("\n" + "="*45)
        print("           Metrics Calculator")
        print("="*45)
        print("1. Calculate PSNR & SSIM (Image vs Image)")
        print("2. Calculate BER (Text file vs Text file)")
        print("3. Exit")
        print("="*45)

        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            try:
                img1 = input("Enter original image path (e.g., cover.png): ").strip()
                img2 = input("Enter stego/modified image path (e.g., stego.png): ").strip()

                if not os.path.isfile(img1) or not os.path.isfile(img2):
                    print("Error: One or both image files not found.")
                    continue

                print("\nCalculating PSNR and SSIM...")
                psnr_val = calculate_psnr(img1, img2)
                ssim_val = calculate_ssim(img1, img2)

                print("\n--- Image Quality Metrics ---")
                if psnr_val is not None:
                    print(f"PSNR: {'Infinite' if psnr_val == float('inf') else f'{psnr_val:.2f} dB'} (higher is better)")
                else:
                    print("PSNR: Calculation failed.")

                if ssim_val is not None:
                    print(f"SSIM: {ssim_val:.6f} (closer to 1 is better)")
                else:
                    print("SSIM: Calculation failed.")
                print("-----------------------------")

            except Exception as e:
                print(f"An error occurred: {e}")

        elif choice == '2':
            try:
                file1 = input("Enter original text file path (e.g., original.txt): ").strip()
                file2 = input("Enter extracted text file path (e.g., extracted.txt): ").strip()

                if not os.path.isfile(file1):
                    print(f"Error: File not found: {file1}")
                    continue
                if not os.path.isfile(file2):
                    print(f"Error: File not found: {file2}")
                    continue

                with open(file1, 'r', encoding='utf-8', errors='ignore') as f1:
                    msg1 = f1.read()
                with open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
                    msg2 = f2.read()

                ber_val = calculate_ber(msg1, msg2)

                print("\n--- Data Integrity Metric (BER) ---")
                print(f"BER: {ber_val:.8f}")
                print(f"({ber_val * 100:.4f}% of bits are different)")
                if ber_val == 0.0:
                    print("Messages are identical (BER = 0).")
                print("-----------------------------------")

            except Exception as e:
                print(f"An error occurred: {e}")

        elif choice == '3':
            print("Exiting. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2 or 3.")


if __name__ == "__main__":
    main_menu()
