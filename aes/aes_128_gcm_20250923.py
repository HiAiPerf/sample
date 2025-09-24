from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64

# --- Step 1: Generate a 128-bit key (16 bytes) ---
key = get_random_bytes(16)  # AES-128
print("Key (base64):", base64.b64encode(key).decode())

# --- Step 2: Create AES cipher in GCM mode ---
cipher = AES.new(key, AES.MODE_GCM)

# GCM uses a random nonce (like an IV, but unique per encryption)
nonce = cipher.nonce
print("Nonce (base64):", base64.b64encode(nonce).decode())

# --- Step 3: Encrypt ---
plaintext = b"Hello, AES-128 GCM encryption!"
ciphertext, tag = cipher.encrypt_and_digest(plaintext)

print("Ciphertext (base64):", base64.b64encode(ciphertext).decode())
print("Tag (base64):", base64.b64encode(tag).decode())  # Authentication tag

# --- Step 4: Decrypt ---
decipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
decrypted = decipher.decrypt_and_verify(ciphertext, tag)

print("Decrypted text:", decrypted.decode())
