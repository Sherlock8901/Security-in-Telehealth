README – Secure Telehealth Data Transmission (AES + DCT + LSB + Blockchain)

==========================================================

This project presents a complete & secure pipeline for medical data transmission using a combination of:

Security Layer	Technique Used
Encryption	AES-256 (CBC Mode)
Steganography – Method 1	DCT-based
Steganography – Method 2	LSB-based (New) 

LSB1


Integrity Verification	Blockchain-style SHA-256 hash
Quality Evaluation	PSNR, SSIM, BER (New) 

Evalution1


Pixel Used Detection	Hidden Pixel Visualization (New) 

Change1

🔐 MODULE 1: AES ENCRYPTION – AES1.py
Purpose:

Encrypt and decrypt confidential health data (e.g. medical reports, patient history).

Workflow:
Step	Input	Output
Encrypt	patient_data.txt	patient_data.enc
Decrypt	patient_data.enc	patient_data_decrypted.txt
Highlights:

AES-256 CBC Mode

Password Protected

Secure Against Eavesdropping

🧠 MODULE 2A: DCT-BASED STEGANOGRAPHY – DCT.py
Purpose:

Hide AES-encrypted file inside frequency domain (Y-channel of image).

Flow:
AES1.py  → Encrypt Data  
DCT.py   → Embed into Image (cover.png → stego.png)

Also Supports:

✔ Extraction
✔ PSNR / SSIM / BER evaluation
✔ Security + invisibility

🧬 MODULE 2B (NEW): LSB-BASED STEGANOGRAPHY – LSB1.py

LSB1

Simple & fast 1-bit LSB File-based steganography

Features:

Fully supports text file embedding/extraction

Stores length in 32-bit header

Lossless output (.png / .bmp / .tiff)

Handles UTF-8 file contents

Menu Flow:
1. Embed text file → image
2. Extract text file from stego image

🧾 MODULE 3: BLOCKCHAIN HASH VERIFICATION – Blockchain.py
Purpose:

Ensure image authenticity (NO tampering during transmission).

Process:
Task	Output
Generate Hash	image_hash.txt
Verify Hash	MATCH / MISMATCH

Uses SHA-256 (Blockchain principle)
Useful for telehealth authentication & legal validation.

🧪 (NEW) MODULE 4: EVALUATION SYSTEM – Evalution1.py

Evalution1

Supported Metrics:
Metric	Purpose
PSNR	Image quality (higher = better)
SSIM	Structural similarity
BER	Extraction error rate (lower = better)
Menu Options:
1. PSNR + SSIM   → Compare cover.png & stego.png  
2. BER           → Compare original.txt & extracted.txt


Supports old & new skimage versions → Highly compatible!

🔍 (NEW) MODULE 5: LSB HIDDEN PIXEL ANALYZER – Change1.py

Change1

Shows exact pixels used for data hiding!
Outputs a visual red heatmap of modified pixels.

Feature	Benefit
Visual proof of embedding	Good for research papers 📑
Pixel count & message size	Good for security analysis
Safe for publication	No data leak
📌 FULL PIPELINE (UPDATED)
1️⃣ AES1.py       → Encrypt          → data.enc
2️⃣ DCT.py OR LSB1.py  → Embed into cover image → stego.png
3️⃣ Blockchain.py → Generate image_hash.txt
4️⃣ Send stego.png + image_hash.txt
5️⃣ Verification:
     ✔ Validate hash
     ✔ Extract message
     ✔ Decrypt via AES1.py
6️⃣ Evalution1.py → Calculate PSNR / SSIM / BER
7️⃣ (Optional) Change1.py → Visualize used pixels

📦 PROJECT STRUCTURE (UPDATED)
├── AES1.py              # AES-256 Encryption/Decryption
├── DCT.py               # Frequency-Domain Steganography
├── LSB1.py              # NEW 1-bit LSB Steganography :contentReference[oaicite:9]{index=9}
├── Change1.py           # NEW Pixel Usage Visualizer  :contentReference[oaicite:10]{index=10}
├── Evalution1.py        # NEW Metrics: PSNR, SSIM, BER  :contentReference[oaicite:11]{index=11}
├── Blockchain.py        # SHA-256 Integrity Verification
├── README.txt           # Documentation

📦 DEPENDENCIES
pip install numpy opencv-python matplotlib scipy scikit-image pycryptodome pillow

👨‍⚕️ AUTHOR

Developed for Secure Telehealth Data Transmission and Research
Combining Cryptography, Steganography & Blockchain Security.
Author Name: Dip Patra
College: NIT Agartala
