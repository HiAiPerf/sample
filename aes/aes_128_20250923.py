from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64

# --- Step 1: Generate a random 128-bit key (16 bytes) ---
key = get_random_bytes(16)  # AES-128 requires 16-byte key
print("Key (base64):", base64.b64encode(key).decode())

# --- Step 2: Create a new AES cipher (CBC mode with random IV) ---
cipher = AES.new(key, AES.MODE_CBC)
iv = cipher.iv
print("IV (base64):", base64.b64encode(iv).decode())

# --- Step 3: Encrypt some plaintext ---
plaintext = b"Hello, AES-128 encryption!"
# AES requires blocks of 16 bytes → pad the plaintext if needed
padding_length = 16 - len(plaintext) % 16
plaintext_padded = plaintext + bytes([padding_length]) * padding_length

ciphertext = cipher.encrypt(plaintext_padded)
print("Ciphertext (base64):", base64.b64encode(ciphertext).decode())

# --- Step 4: Decrypt the ciphertext ---
decipher = AES.new(key, AES.MODE_CBC, iv)
decrypted_padded = decipher.decrypt(ciphertext)

# Remove padding
padding_length = decrypted_padded[-1]
decrypted = decrypted_padded[:-padding_length]

print("Decrypted text:", decrypted.decode())
