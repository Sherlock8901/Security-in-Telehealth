from PIL import Image
import math
import os
import sys

def get_lsb(value):
    """Returns the Least Significant Bit (1 or 0) of a byte value."""
    return value & 1

def bits_to_int(bit_list):
    """Converts a list of 0s and 1s to an integer."""
    result = 0
    for bit in bit_list:
        result = (result << 1) | bit
    return result

def decode_pixel_map(stego_image_path):
    """
    Analyzes a stego-image to find which pixels were used for hiding.
    
    Returns:
        - A new PIL Image object (the pixel map).
        - The number of pixels used.
        - The length of the hidden message.
    """
    try:
        # 1. Open the stego-image
        img = Image.open(stego_image_path).convert('RGB')
        width, height = img.size
        
        # 2. Read the 32-bit header
        header_bits = []
        pixel_count = 0
        pixels = list(img.getdata())
        
        # We need 32 bits for the header
        while len(header_bits) < 32:
            if pixel_count >= len(pixels):
                print("Error: Reached end of image before reading 32-bit header.")
                return None, 0, 0
                
            r, g, b = pixels[pixel_count]
            
            header_bits.append(get_lsb(r))
            if len(header_bits) == 32: break
                
            header_bits.append(get_lsb(g))
            if len(header_bits) == 32: break
                
            header_bits.append(get_lsb(b))
            
            pixel_count += 1
        
        # 3. Convert header to message length
        # This is the length of the *data payload* in bits
        message_length_bits = bits_to_int(header_bits)
        
        # 4. Calculate total bits and total pixels
        total_bits = 32 + message_length_bits # Header + Message
        total_pixels_used = math.ceil(total_bits / 3.0)
        
        if total_pixels_used > len(pixels):
            print(f"Warning: Header indicates {total_pixels_used} pixels,")
            print(f"but image only has {len(pixels)} pixels.")
            total_pixels_used = len(pixels) # Cap at max pixels
            
        print("\n--- Analysis Complete ---")
        print(f"Header found: {message_length_bits} bits")
        print(f"Total bits embedded (header + data): {total_bits}")
        print(f"Total pixels used for hiding: {total_pixels_used}")
        print("---------------------------\n")

        # 5. Create the pixel map
        map_image = Image.new('RGB', (width, height), 'black')
        map_pixels = map_image.load()
        
        for i in range(total_pixels_used):
            x = i % width
            y = i // width
            map_pixels[x, y] = (255, 0, 0) # Set pixel to bright Red
            
        return map_image, total_pixels_used, message_length_bits

    except FileNotFoundError:
        print(f"Error: Stego-image not found at '{stego_image_path}'")
        return None, 0, 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, 0, 0

def main_menu():
    while True:
        print("\n" + "="*45)
        print("      LSB HIDDEN PIXEL MAP GENERATOR")
        print("="*45)
        print("This tool shows exactly which pixels were")
        print("used to hide data, based on the LSB")
        print("algorithm from your paper.")
        print("="*45)
        print("1. Analyze Stego-Image and Show Pixel Map")
        print("2. Exit")
        print("="*45)
        
        choice = input("Enter your choice (1-2): ").strip()

        if choice == '1':
            stego_path = input("Enter path to your Stego-Image (e.g., stego.png): ").strip()
            
            if not os.path.exists(stego_path):
                print(f"\n[!] Error: File not found: '{stego_path}'")
                continue
            
            print("Analyzing image...")
            map_img, pixels, bits = decode_pixel_map(stego_path)
            
            if map_img:
                print("Showing pixel map... (Close the image window to continue)")
                map_img.show(title="Map of Hidden Pixels (in Red)")
        
        elif choice == '2':
            print("Exiting...")
            sys.exit()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()


