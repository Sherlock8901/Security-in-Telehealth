import os
import base64
import getpass  # Kept though we use input() for visibility
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# --- Configuration ---
SALT_SIZE = 16
BLOCK_SIZE = 16
KEY_SIZE = 32
PBKDF2_ITERATIONS = 1000000
CHUNK_SIZE = 1024 * 1024  # 1 MB


def get_derived_key(password: str, salt: bytes) -> bytes:
    """Derives a 32-byte (256-bit) key from the password and salt using PBKDF2."""
    return PBKDF2(password, salt, dkLen=KEY_SIZE, count=PBKDF2_ITERATIONS)


def encrypt_file(in_filepath: str, out_filepath: str, password: str):
    """
    Encrypts a file using AES-256 in CBC mode.
    Output file is a single Base64 line: [SALT + IV + ENCRYPTED_DATA]
    """
    print(f"Starting encryption of '{in_filepath}'...")

    try:
        salt = get_random_bytes(SALT_SIZE)
        iv = get_random_bytes(BLOCK_SIZE)
        key = get_derived_key(password, salt)
        cipher = AES.new(key, AES.MODE_CBC, iv)

        encrypted_data = b""

        with open(in_filepath, 'rb') as in_file:
            while True:
                chunk = in_file.read(CHUNK_SIZE)
                if len(chunk) == 0:
                    break
                elif len(chunk) < CHUNK_SIZE:
                    chunk = pad(chunk, BLOCK_SIZE)
                    encrypted_data += cipher.encrypt(chunk)
                    break
                else:
                    encrypted_data += cipher.encrypt(chunk)

        # Combine all parts and encode as one Base64 line
        final_data = salt + iv + encrypted_data
        encoded_line = base64.b64encode(final_data)

        with open(out_filepath, 'wb') as out_file:
            out_file.write(encoded_line)

        print(f"✅ Encryption successful. Output file (single line): '{out_filepath}'")

    except FileNotFoundError:
        print(f"❌ Error: Input file '{in_filepath}' not found.")
    except Exception as e:
        print(f"⚠️ An error occurred during encryption: {e}")
        if os.path.exists(out_filepath):
            os.remove(out_filepath)


def decrypt_file(in_filepath: str, out_filepath: str, password: str):
    """
    Decrypts a Base64 single-line encrypted file.
    Expects format: Base64([SALT + IV + ENCRYPTED_DATA])
    """
    print(f"Starting decryption of '{in_filepath}'...")

    try:
        with open(in_filepath, 'rb') as in_file:
            encoded_data = in_file.read().strip()
            data = base64.b64decode(encoded_data)

        # Extract salt and IV
        salt = data[:SALT_SIZE]
        iv = data[SALT_SIZE:SALT_SIZE + BLOCK_SIZE]
        encrypted_data = data[SALT_SIZE + BLOCK_SIZE:]

        key = get_derived_key(password, salt)
        cipher = AES.new(key, AES.MODE_CBC, iv)

        decrypted = cipher.decrypt(encrypted_data)

        try:
            final_data = unpad(decrypted, BLOCK_SIZE)
        except ValueError:
            print("-" * 50)
            print("❌ DECRYPTION FAILED!")
            print("Likely due to an incorrect password or corrupted file.")
            print("-" * 50)
            if os.path.exists(out_filepath):
                os.remove(out_filepath)
            return

        with open(out_filepath, 'wb') as out_file:
            out_file.write(final_data)

        print(f"✅ Decryption successful. Output file: '{out_filepath}'")

    except FileNotFoundError:
        print(f"❌ Error: Input file '{in_filepath}' not found.")
    except Exception as e:
        print(f"⚠️ An error occurred during decryption: {e}")
        if os.path.exists(out_filepath):
            os.remove(out_filepath)


def get_password(confirm: bool = True) -> str:
    """
    Prompts user for a password.
    WARNING: Uses input() — password will be visible on screen.
    """
    while True:
        password = input("Enter password: ")
        if not password:
            print("Password cannot be empty.")
            continue

        if confirm:
            password_confirm = input("Confirm password: ")
            if password == password_confirm:
                return password
            else:
                print("Passwords do not match. Try again.")
        else:
            return password


def get_filepath(prompt: str) -> str:
    """Prompts user for a file path and validates its existence."""
    while True:
        filepath = input(prompt).strip().strip("'\"")
        if not filepath:
            print("File path cannot be empty.")
            continue
        if "encrypt" in prompt or "decrypt" in prompt:
            if not os.path.exists(filepath):
                print(f"Error: File not found at '{filepath}'. Try again.")
                continue
        return filepath


def main_menu():
    """Displays the main menu and handles user input."""
    while True:
        print("\n--- AES-256 File Encryption Utility ---")
        print("1. Encrypt a file")
        print("2. Decrypt a file")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            print("\n[ ENCRYPT FILE ]")
            in_file = get_filepath("Enter the full path of the file to encrypt: ")
            out_file = input(f"Enter output file path (default: {in_file}.enc): ").strip().strip("'\"")
            if not out_file:
                out_file = in_file + ".enc"

            print("\n⚠️ WARNING: Your password WILL be visible.")
            password = get_password(confirm=True)
            encrypt_file(in_file, out_file, password)

        elif choice == '2':
            print("\n[ DECRYPT FILE ]")
            in_file = get_filepath("Enter the full path of the file to decrypt: ")
            default_out = in_file.rsplit('.enc', 1)[0]
            if default_out == in_file:
                default_out = in_file + ".dec"
            out_file = input(f"Enter output file path (default: {default_out}): ").strip().strip("'\"")
            if not out_file:
                out_file = default_out

            print("\n⚠️ WARNING: Your password WILL be visible.")
            password = get_password(confirm=False)
            decrypt_file(in_file, out_file, password)

        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main_menu()
