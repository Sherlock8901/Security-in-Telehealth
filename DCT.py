import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from math import log10, sqrt
from skimage.metrics import structural_similarity  # Added for SSIM

# -------------------- DCT Helpers --------------------
def dct2(block):
    """2D DCT (orthonormal) on block"""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """2D inverse DCT (orthonormal) on block"""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def text_to_bits(text):
    b = text.encode('utf-8')
    bits = []
    for byte in b:
        bits.extend([int(x) for x in format(byte, '08b')])
    return bits

def bits_to_text(bits):
    bytes_out = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        if len(chunk) < 8:
            break
        val = 0
        for bit in chunk:
            val = (val << 1) | int(bit)
        bytes_out.append(val)
    # Use errors='replace' to handle potential decoding errors
    return bytes(bytes_out).decode('utf-8', errors='replace')

# JPEG-like luminance quantization table
Q_LUMINANCE = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float32)

# -------------------- Embedding (Color Images) --------------------
def embed_message_dct_color(cover_img, message, block_size=8, coeff_pos=(4,3), Q=Q_LUMINANCE):
    """
    Embed message into color image using Y channel of YCrCb
    """
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(cover_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # Crop all channels to same divisible size
    h, w = y.shape
    h_new = h - (h % block_size)
    w_new = w - (w % block_size)
    
    y = y[:h_new, :w_new]
    cr = cr[:h_new, :w_new]
    cb = cb[:h_new, :w_new]
    
    # Embed in Y channel (luminance)
    y_stego = embed_message_dct_grayscale(y.astype(np.float32), message, block_size, coeff_pos, Q)
    
    # Merge back
    ycrcb_stego = cv2.merge([y_stego.astype(np.uint8), cr, cb])
    stego_bgr = cv2.cvtColor(ycrcb_stego, cv2.COLOR_YCrCb2BGR)
    
    return stego_bgr

def embed_message_dct_grayscale(cover_img, message, block_size=8, coeff_pos=(4,3), Q=Q_LUMINANCE):
    """
    Embed message into grayscale image
    """
    cover_float = cover_img.astype(np.float32)
    h, w = cover_float.shape
    
    # Ensure divisible by block size
    h_new = h - (h % block_size)
    w_new = w - (w % block_size)
    cover_float = cover_float[:h_new, :w_new]
    
    capacity_bits = (h_new // block_size) * (w_new // block_size)

    payload_bits = text_to_bits(message)
    payload_len = len(payload_bits)
    # Header is 32 bits, storing the length of the payload
    header_bits = [int(x) for x in format(payload_len, '032b')]
    bits = header_bits + payload_bits

    if len(bits) > capacity_bits:
        raise ValueError(f"Message too long: need {len(bits)} bits, capacity is {capacity_bits}")

    stego = np.zeros_like(cover_float, dtype=np.float32)
    bit_idx = 0
    
    for by in range(0, h_new, block_size):
        for bx in range(0, w_new, block_size):
            block = cover_float[by:by+block_size, bx:bx+block_size]
            D = dct2(block)
            
            # Quantize
            Qblock = Q.copy()
            qD = np.round(D / Qblock).astype(np.int32)

            # Embed bit
            if bit_idx < len(bits):
                target = bits[bit_idx]
                r, c = coeff_pos
                current = int(qD[r, c]) & 1 # Get LSB
                
                # Change LSB if it doesn't match target bit
                if current != target:
                    if qD[r, c] >= 0:
                        qD[r, c] += 1
                    else:
                        qD[r, c] -= 1
                    # Ensure LSB is now correct (handles qD[r,c] == 0 case)
                    if (int(qD[r, c]) & 1) != target:
                         qD[r, c] = 1 if target == 1 else 0
                         
                bit_idx += 1

            # Dequantize and inverse DCT
            D_mod = qD.astype(np.float32) * Qblock
            idct_block = idct2(D_mod)
            stego[by:by+block_size, bx:bx+block_size] = idct_block

    return np.clip(np.round(stego), 0, 255).astype(np.uint8)

# -------------------- Extraction (Color Images) --------------------
def extract_message_dct_color(stego_img, block_size=8, coeff_pos=(4,3), Q=Q_LUMINANCE):
    """
    Extract message from color image using Y channel
    """
    ycrcb = cv2.cvtColor(stego_img, cv2.COLOR_BGR2YCrCb)
    y, _, _ = cv2.split(ycrcb)
    
    # Crop to divisible size
    h, w = y.shape
    h_new = h - (h % block_size)
    w_new = w - (w % block_size)
    y = y[:h_new, :w_new]
    
    return extract_message_dct_grayscale(y.astype(np.float32), block_size, coeff_pos, Q)

def extract_message_dct_grayscale(stego_img, block_size=8, coeff_pos=(4,3), Q=Q_LUMINANCE):
    stego_float = stego_img.astype(np.float32)
    h, w = stego_float.shape
    
    # Ensure divisible by block size
    h_new = h - (h % block_size)
    w_new = w - (w % block_size)
    stego_float = stego_float[:h_new, :w_new]
    
    bits = []
    
    for by in range(0, h_new, block_size):
        for bx in range(0, w_new, block_size):
            block = stego_float[by:by+block_size, bx:bx+block_size]
            D = dct2(block)
            
            Qblock = Q.copy()
            qD = np.round(D / Qblock).astype(np.int32)
            
            r, c = coeff_pos
            bits.append(int(qD[r, c]) & 1) # Extract LSB

    if len(bits) < 32:
        return "" # Not enough bits for a header

    # Extract header (message length)
    header_bits = bits[:32]
    payload_len_str = "".join(map(str, header_bits))
    if not payload_len_str: # Handle empty string case
        return ""
        
    payload_len = int(payload_len_str, 2)
    
    # Extract message
    if 32 + payload_len > len(bits):
        # Data is shorter than header indicates, possibly truncated
        payload_len = len(bits) - 32
    
    payload_bits = bits[32:32+payload_len]
    return bits_to_text(payload_bits)

# -------------------- Evaluation Metrics --------------------
def calculate_psnr(img1, img2):
    # Ensure both images have same dimensions
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
    
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * log10(255.0 / sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculates SSIM for color or grayscale images."""
    # Ensure both images have same dimensions
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]

    if img1.ndim == 2:
        # Grayscale image
        return structural_similarity(img1, img2, data_range=255)
    elif img1.ndim == 3:
        # Color image
        # channel_axis=-1 for (H, W, C) format used by cv2
        return structural_similarity(img1, img2, data_range=255, multichannel=True, channel_axis=-1)
    else:
        raise ValueError("Unsupported image dimension for SSIM.")

def calculate_ber(original_msg, extracted_msg):
    """Calculates Bit Error Rate (BER) between two strings."""
    orig_bits = text_to_bits(original_msg)
    extr_bits = text_to_bits(extracted_msg)
    
    orig_len = len(orig_bits)
    extr_len = len(extr_bits)
    
    # Use the length of the longer list for total bit count
    max_len = max(orig_len, extr_len)
    if max_len == 0:
        if orig_len == 0 and extr_len == 0:
            return 0.0, 0, 0 # Both empty, no errors
        else:
            # This case should be covered by max_len > 0
            pass 
            
    if max_len == 0:
       return 0.0, 0, 0 # Handle case where both messages are empty

    error_bits = 0
    # Compare bits up to the minimum length
    min_len = min(orig_len, extr_len)
    for i in range(min_len):
        if orig_bits[i] != extr_bits[i]:
            error_bits += 1
            
    # Any extra bits in one list vs the other are errors
    error_bits += abs(orig_len - extr_len)
    
    ber = error_bits / float(max_len)
    return ber, error_bits, max_len

# -------------------- Visualization --------------------
def plot_psnr_histogram(original, stego, psnr_value, output_path="psnr_histogram.png"):
    """Create PSNR histogram and comparison plot"""
    # Ensure both images have same dimensions
    h = min(original.shape[0], stego.shape[0])
    w = min(original.shape[1], stego.shape[1])
    original = original[:h, :w]
    stego = stego[:h, :w]
    
    plt.figure(figsize=(15, 10))
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    stego_rgb = cv2.cvtColor(stego, cv2.COLOR_BGR2RGB)
    
    # 1. Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(original_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # 2. Stego Image
    plt.subplot(2, 3, 2)
    plt.imshow(stego_rgb)
    plt.title(f'Stego Image\nPSNR: {psnr_value:.2f} dB')
    plt.axis('off')
    
    # 3. Difference Image
    plt.subplot(2, 3, 3)
    diff = cv2.absdiff(original, stego)
    # Enhance difference for visibility if needed
    diff_visible = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff_rgb = cv2.cvtColor(diff_visible, cv2.COLOR_BGR2RGB)
    plt.imshow(diff_rgb)
    plt.title('Difference Image (Normalized)')
    plt.axis('off')
    
    # 4. Histogram of Original Image
    plt.subplot(2, 3, 4)
    colors = ('b', 'g', 'r')
    for i, col in enumerate(colors):
        histr = cv2.calcHist([original], [i], None, [256], [0, 256])
        plt.plot(histr, color=col, alpha=0.7)
    plt.title('Original Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    
    # 5. Histogram of Stego Image
    plt.subplot(2, 3, 5)
    for i, col in enumerate(colors):
        histr = cv2.calcHist([stego], [i], None, [256], [0, 256])
        plt.plot(histr, color=col, alpha=0.7)
    plt.title('Stego Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    
    # 6. PSNR Value Bar Chart
    plt.subplot(2, 3, 6)
    plt.bar(['PSNR'], [psnr_value], color='green')
    plt.title(f'PSNR Value: {psnr_value:.2f} dB')
    plt.ylabel('dB')
    plt.ylim(0, max(60, psnr_value + 10)) # Adjusted y-limit
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\n✓ Histogram saved to {output_path}")
    plt.show()
    
    return output_path

# -------------------- Main Application --------------------
class DCTSteganography:
    def __init__(self):
        self.original = None
        self.stego_img = None
        self.original_message = None # Added to store message for BER
        self.psnr = None
        self.ssim = None # Added for SSIM
    
    def embed(self, image_path, message, output_path):
        """Embed message into color image"""
        # Read color image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.original = img.copy()
        self.original_message = message # Store original message
        
        self.stego_img = embed_message_dct_color(img, message)
        
        # Calculate metrics
        self.psnr = calculate_psnr(self.original, self.stego_img)
        self.ssim = calculate_ssim(self.original, self.stego_img)
        
        # Save as PNG to avoid compression
        cv2.imwrite(output_path, self.stego_img)
        
        # Return all relevant info
        return self.stego_img, self.psnr, self.ssim, self.original
    
    def extract(self, image_path):
        """Extract message from color image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        return extract_message_dct_color(img)

# -------------------- Menu-Driven Interface --------------------
def main():
    # Note: Requires 'pip install scikit-image' for SSIM
    stego = DCTSteganography()
    
    while True:
        print("\n" + "=" * 60)
        print("DCT STEGANOGRAPHY SYSTEM - MENU")
        print("=" * 60)
        print("1. Embed message in image (Calculates PSNR & SSIM)")
        print("2. Extract message from image (Calculates BER if 1 was run)")
        print("3. View PSNR histogram (after embedding)")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Embed message
            image_path = input("Enter input image path: ").strip().strip('"')
            if not os.path.exists(image_path):
                print(f"Error: Image file not found at '{image_path}'")
                continue
            
            message = input("Enter secret message: ").strip()
            if not message:
                print("Error: Message cannot be empty!")
                continue
            
            output_path = input("Enter output path (default: stego.png): ").strip().strip('"')
            if not output_path:
                output_path = "stego.png"
            
            try:
                print("\nEmbedding message and calculating metrics...")
                stego_img, psnr, ssim, original = stego.embed(image_path, message, output_path)
                
                print(f"\n--- Embedding Complete ---")
                print(f"✓ Message embedded successfully!")
                print(f"✓ Saved as: {output_path}")
                print(f"✓ Original size: {original.shape}")
                print(f"✓ Stego size: {stego_img.shape}")
                
                print(f"\n--- Evaluation Metrics ---")
                print(f"✓ PSNR: {psnr:.2f} dB")
                print(f"✓ SSIM: {ssim:.4f} (Value closer to 1 is better)")
                
            except Exception as e:
                print(f"\nAn error occurred during embedding: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '2':
            # Extract message
            stego_path = input("Enter stego image path: ").strip().strip('"')
            if not os.path.exists(stego_path):
                print(f"Error: Stego image not found at '{stego_path}'")
                continue
            
            try:
                print("\nExtracting message...")
                extracted = stego.extract(stego_path)
                print(f"\n--- Extraction Complete ---")
                print(f"✓ Extracted message: '{extracted}'")
                
                # Calculate BER if original message is available from step 1
                if stego.original_message is not None:
                    print(f"\n--- BER Calculation (comparing with message from step 1) ---")
                    ber, errors, total = calculate_ber(stego.original_message, extracted)
                    print(f"✓ Original Message:  '{stego.original_message}'")
                    print(f"✓ Extracted Message: '{extracted}'")
                    print(f"✓ Bit Error Rate (BER): {ber:.6f}")
                    print(f"  ({errors} error bits out of {total} total bits)")
                else:
                    print("\n(Note: Run '1. Embed message' in this same session to also calculate BER)")

            except Exception as e:
                print(f"\nAn error occurred during extraction: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '3':
            # Show PSNR histogram
            if stego.original is None or stego.stego_img is None:
                print("Error: No embedding data found. Please run '1. Embed message' first.")
            else:
                print("Generating PSN-R and Histogram plot...")
                plot_psnr_histogram(stego.original, stego.stego_img, stego.psnr)
        
        elif choice == '4':
            print("Exiting program. Goodbye!")
            break
        
        else:
            print("Invalid choice! Please enter 1-4.")

if __name__ == "__main__":
    print("Welcome! This script requires: opencv-python, numpy, matplotlib, scipy, scikit-image")
    print("Please ensure they are installed (e.g., 'pip install scikit-image')")
    main()