import hashlib
import os

def sha256_hash_of_file(path: str) -> str:
    """Calculates the SHA-256 hash of a file."""
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        print(f"Error: File not found for hashing: {path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def handle_sender_hash():
    """
    SENDER'S ACTION:
    Generates a hash from an image and saves it to a file.
    """
    print("\n--- 1. Sender: Generate Hash ---")
    
    # 1. Get Input Image
    while True:
        input_image_file = input("   Enter path to your ORIGINAL image: ")
        if os.path.exists(input_image_file):
            break
        print(f"  File not found: '{input_image_file}'. Please try again.")

    # 2. Hash Output
    image_hash = sha256_hash_of_file(input_image_file)
    hash_file = "image_hash.txt"
    
    if image_hash:
        print(f"\n  Generated SHA-256 hash: {image_hash}")
        try:
            with open(hash_file, 'w') as f:
                f.write(image_hash)
            print(f"  Hash saved to '{hash_file}'. This file is now 'sent' to the receiver.")
        except Exception as e:
            print(f"  WARNING: Could not save hash file! {e}")
            
    print("\n--- Sender Workflow Complete ---")

def handle_receiver_verify():
    """
    RECEIVER'S ACTION:
    Compares a local file's hash against a trusted hash
    that is AUTOMATICALLY read from 'image_hash.txt'.
    """
    print("\n--- 2. Receiver: Verify Image Hash ---")
    
    hash_file = "image_hash.txt"
    
    # --- THIS IS THE MODIFIED SECTION ---
    # 1. Get Trusted Hash (Automatically from file)
    print(f"\n  Attempting to read trusted hash from '{hash_file}'...")
    try:
        with open(hash_file, 'r') as f:
            trusted_hash = f.read().strip()
        
        if not trusted_hash:
            raise ValueError("Hash file is empty.")
        
        print(f"  Successfully read trusted hash from file.")
        
        if len(trusted_hash) != 64:
            print(f"  WARNING: Hash in {hash_file} is not 64 characters! File might be corrupt.")
    
    except FileNotFoundError:
        print(f"  Error: Could not find '{hash_file}'.")
        print("  Please run Option 1 (Sender) first to generate the hash file.")
        return
    except Exception as e:
        print(f"  Error reading hash file: {e}")
        return
    # --- END OF MODIFIED SECTION ---

    # 2. Get Received Image
    while True:
        received_image_file = input("\n  Enter path to the IMAGE you received: ")
        if os.path.exists(received_image_file):
            break
        print(f"  File not found: '{received_image_file}'. Please try again.")

    # --- Execute ---
    print("\nStarting process...")
    # 3. Compute Local Hash
    print("1) Computing local hash of the received file...")
    receiver_hash = sha256_hash_of_file(received_image_file)
    if receiver_hash is None: 
        return
    
    print(f"   Local Hash (from file):     {receiver_hash}")
    print(f"   Trusted Hash (from {hash_file}): {trusted_hash}")

    # 4. VERIFY
    if receiver_hash == trusted_hash:
        print("\n2) ✅ VERIFIED: Hashes match! Image is authentic and unmodified.")
    else:
        print("\n2) ❌ VERIFICATION FAILED: Hashes do NOT match!")
        print("   DO NOT TRUST THIS FILE. It is tampered or incorrect.")
    
    print("\n--- Receiver Workflow Complete ---")

def main_menu():
    """Displays the main menu and handles user choices."""
    while True:
        print("\n" + "="*50)
        print("     Automatic Image Hash Verification Tool")
        print(" (The core integrity-check principle from Blockchain)")
        print("="*50)
        print("1. Generate Hash (Sender)")
        print("2. Verify Hash (Receiver)")
        print("3. Exit")
        
        choice = input("Enter your choice (1, 2, or 3): ")
        
        if choice == '1':
            handle_sender_hash()
        elif choice == '2':
            handle_receiver_verify()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main_menu()