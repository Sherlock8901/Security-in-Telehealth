README - Secure Steganography and Blockchain Verification System
=============================================================

This project combines three main security modules to provide an end-to-end secure telehealth data transmission pipeline:
1. AES Encryption (AES1.py)
2. DCT-Based Steganography (DCT.py)
3. Blockchain-Style Hash Verification (Blockchain.py)

-------------------------------------------------------------
MODULE 1: AES ENCRYPTION (AES1.py)
-------------------------------------------------------------
Purpose:
- AES1.py is used to ENCRYPT and DECRYPT confidential data (e.g., patient health records, reports, or sensitive metadata).
- It ensures that even if intercepted, the data remains unreadable.

How it works:
1. Input a file (e.g., "patient_data.txt") and choose a password.
2. The program uses AES-256 encryption in CBC mode to produce an encrypted file (*.enc).
3. The same password is required to decrypt it back into readable form.

Example workflow:
- Encrypt:  AES1.py → patient_data.txt → patient_data.txt.enc
- Decrypt:  AES1.py → patient_data.txt.enc → patient_data_decrypted.txt

-------------------------------------------------------------
MODULE 2: DCT STEGANOGRAPHY (DCT.py)
-------------------------------------------------------------
Purpose:
- Embeds the AES-encrypted data into a cover image using the Discrete Cosine Transform (DCT).
- This hides the existence of the data itself within an image (cover image → stego image).

How it works:
1. Takes a cover image (e.g., "cover.png") and secret text or encrypted data.
2. Uses DCT-based frequency-domain embedding to hide the message inside the Y-channel of the image.
3. Produces a stego image (e.g., "stego.png") that visually looks identical to the cover image.
4. Allows extraction and quality analysis using metrics such as PSNR, SSIM, and BER.

Integration with AES:
- Before embedding, encrypt your message or data using AES1.py.
- Then use DCT.py to embed the encrypted output into the image.
- Example:
  1. Run AES1.py → "patient_data.txt" → "patient_data.txt.enc"
  2. Run DCT.py → Embed "patient_data.txt.enc" into "cover.png" → "stego.png"

-------------------------------------------------------------
MODULE 3: BLOCKCHAIN VERIFICATION (Blockchain.py)
-------------------------------------------------------------
Purpose:
- Ensures integrity and authenticity of the transmitted image using SHA-256 hashing.
- Simulates blockchain verification principles for image authenticity.

How it works:
1. The sender generates a hash of the stego image (stego.png).
   → Saves it in a "image_hash.txt" file.
2. The receiver uses the same tool to compute the hash of the received stego image.
3. If both hashes match, the image is verified as authentic and untampered.

Integration with DCT:
- After embedding the encrypted data using DCT.py, run Blockchain.py to generate a hash of the stego image.
- The receiver uses Blockchain.py to verify that the received stego image has not been altered.

-------------------------------------------------------------
FULL PIPELINE OVERVIEW
-------------------------------------------------------------
Below is the step-by-step integration process showing how all modules work together:

1️⃣ **Encryption**
   - Use AES1.py to encrypt your sensitive data.
   - Output: Encrypted file (.enc)

2️⃣ **Data Hiding**
   - Use DCT.py to embed the encrypted data into an image.
   - Output: Stego image (stego.png)

3️⃣ **Integrity Hash Generation**
   - Use Blockchain.py (Option 1) to generate a SHA-256 hash of stego.png.
   - Output: image_hash.txt

4️⃣ **Transmission**
   - Send the stego image (stego.png) and hash (image_hash.txt) to the receiver.

5️⃣ **Verification**
   - Receiver uses Blockchain.py (Option 2) to verify the integrity of stego.png.
   - If verified, use DCT.py to extract the embedded message.
   - Finally, decrypt it with AES1.py using the shared password.

-------------------------------------------------------------
EVALUATION METRICS
-------------------------------------------------------------
- PSNR (Peak Signal-to-Noise Ratio): Measures image quality (higher is better).
- SSIM (Structural Similarity Index): Measures similarity between cover and stego image.
- BER (Bit Error Rate): Measures extraction accuracy (lower is better).

-------------------------------------------------------------
DEPENDENCIES
-------------------------------------------------------------
Ensure the following Python packages are installed:

pip install numpy opencv-python matplotlib scipy scikit-image pycryptodome

-------------------------------------------------------------
PROJECT STRUCTURE
-------------------------------------------------------------
├── AES1.py              # AES encryption & decryption utility
├── DCT.py               # DCT-based steganography system
├── Blockchain.py        # Hash verification system (blockchain principle)
├── README.txt           # Documentation file

-------------------------------------------------------------
AUTHOR
-------------------------------------------------------------
Developed for secure telehealth data transmission research.
Combines cryptography, steganography, and blockchain verification principles.
