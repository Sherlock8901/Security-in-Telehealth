import numpy as np
from PIL import Image
import sys
import os

# --- UTILITIES ---

def message_to_binary_bytes(message: str) -> str:
    """Convert a string to a binary string representing UTF-8 bytes."""
    if not isinstance(message, (str, bytes, bytearray)):
        raise TypeError("message must be str or bytes")
    if isinstance(message, str):
        b = message.encode('utf-8')
    else:
        b = bytes(message)
    return ''.join(format(byte, '08b') for byte in b)

def binary_to_bytes(bin_str: str) -> bytes:
    """Convert a binary string (length multiple of 8) to bytes."""
    if len(bin_str) % 8 != 0:
        raise ValueError("binary string length must be multiple of 8")
    return bytes(int(bin_str[i:i+8], 2) for i in range(0, len(bin_str), 8))

# --- ENCODING ---

def hide_message(image_path: str, secret_file: str, output_path: str):
    """Hide the contents of a text file inside an image using 1-bit LSB."""
    print("Starting encoding process...")
    if not os.path.exists(secret_file):
        print(f"Error: Secret file '{secret_file}' not found.")
        return

    # Read text file contents
    with open(secret_file, 'r', encoding='utf-8') as f:
        secret_message = f.read()

    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        return
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    if image.mode != 'RGB':
        print(f"Image mode is '{image.mode}'. Converting to 'RGB'.")
        image = image.convert('RGB')

    width, height = image.size
    binary_message = message_to_binary_bytes(secret_message)
    header = format(len(binary_message), '032b')
    payload = header + binary_message

    capacity = width * height * 3
    if len(payload) > capacity:
        print(f"Error: Message too large. Image can hold {capacity} bits, message requires {len(payload)} bits.")
        return

    print(f"Embedding {len(binary_message)} bits into image...")

    new_img = image.copy()
    pixels = new_img.load()
    data_index = 0

    for y in range(height):
        for x in range(width):
            if data_index >= len(payload):
                break
            r, g, b = pixels[x, y]
            if data_index < len(payload):
                r = (r & ~1) | int(payload[data_index]); data_index += 1
            if data_index < len(payload):
                g = (g & ~1) | int(payload[data_index]); data_index += 1
            if data_index < len(payload):
                b = (b & ~1) | int(payload[data_index]); data_index += 1
            pixels[x, y] = (r, g, b)
        if data_index >= len(payload):
            break

    out_root, out_ext = os.path.splitext(output_path)
    if out_ext.lower() not in ['.png', '.bmp', '.tiff', '.tif']:
        print("Warning: Using PNG for lossless saving.")
        output_path = out_root + '.png'

    new_img.save(output_path, format='PNG')
    print(f"Message embedded successfully. Stego image saved as '{output_path}'")

# --- DECODING ---

def reveal_message(image_path: str, output_text_path: str):
    """Extract hidden text data from a stego image and save it as a .txt file."""
    print("Starting decoding process...")
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: '{image_path}' not found.")
        return
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    if image.mode != 'RGB':
        print(f"Image mode is '{image.mode}'. Converting to 'RGB'.")
        image = image.convert('RGB')

    width, height = image.size
    pixels = image.load()

    # Read header first (32 bits = message bit length)
    header_bits = ''
    count = 0
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            for channel_lsb in (r & 1, g & 1, b & 1):
                header_bits += str(channel_lsb)
                count += 1
                if count == 32:
                    break
            if count == 32:
                break
        if count == 32:
            break

    if len(header_bits) != 32:
        print("Error: Could not read header.")
        return

    msg_length_bits = int(header_bits, 2)
    print(f"Message bit length: {msg_length_bits}")

    # Flatten LSBs for full payload
    all_bits = []
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            all_bits.extend([r & 1, g & 1, b & 1])

    message_bits = all_bits[32:32 + msg_length_bits]
    if len(message_bits) < msg_length_bits:
        print("Warning: message incomplete or corrupted.")
        return

    binary_string = ''.join(str(bit) for bit in message_bits)
    message_bytes = binary_to_bytes(binary_string)
    message = message_bytes.decode('utf-8', errors='replace')

    # Save to text file
    with open(output_text_path, 'w', encoding='utf-8') as out_f:
        out_f.write(message)
    print(f"Hidden message extracted and saved as '{output_text_path}'")

# --- MAIN MENU ---

def main_menu():
    while True:
        print("\n" + "="*40)
        print("     File-based LSB Steganography")
        print("="*40)
        print("1. Embed text file into image")
        print("2. Extract text file from image")
        print("3. Exit")
        print("="*40)
        choice = input("Enter choice (1-3): ").strip()

        if choice == '1':
            img_in = input("Enter cover image path (e.g., cover.png): ").strip()
            txt_in = input("Enter secret text file path (e.g., secret.txt): ").strip()
            img_out = input("Enter output stego image path (e.g., stego.png): ").strip()
            hide_message(img_in, txt_in, img_out)

        elif choice == '2':
            img_in = input("Enter stego image path (e.g., stego.png): ").strip()
            txt_out = input("Enter output text file name (e.g., extracted.txt): ").strip()
            reveal_message(img_in, txt_out)

        elif choice == '3':
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main_menu()
