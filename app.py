import streamlit as st
import numpy as np 
import cv2
from cv2 import dnn
import os
from io import BytesIO
import base64
import json
import requests
import time
import shutil

# --- NEW IMPORTS FOR BACKGROUND TRAINING AND MODEL DEFINITION ---
import multiprocessing
# We only import the torch components to define the structure and logic.
# NOTE: Running the actual training requires installing PyTorch and having a GPU.
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from torchvision import transforms # Added transforms import
    # Check if a model checkpoint exists for potential future inference
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    class DummyModule:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return self
        def to(self, *args, **kwargs): return self
        def load_state_dict(self, *args, **kwargs): return self
    # Placeholder classes if PyTorch is not available
    nn = DummyModule()
    nn.Module = object
    nn.Conv2d = DummyModule
    nn.BatchNorm2d = DummyModule
    nn.ReLU = DummyModule
    nn.MaxPool2d = DummyModule
    nn.ConvTranspose2d = DummyModule
    nn.Tanh = DummyModule
    Dataset = object
    DataLoader = object
    torch = DummyModule()

# --- NEW IMPORTS FOR AUDIO WAVE VISUALIZATION AND MANIPULATION ---
# Using a try block for these as well, since they are visualization dependencies.
try:
    import matplotlib.pyplot as plt
    from scipy.io import wavfile as wav
    AUDIO_PLOTTING_AVAILABLE = True
except ImportError:
    AUDIO_PLOTTING_AVAILABLE = False
    # If not available, we'll gracefully skip the visualization.

# --- CONFIGURATION AND FILE SYSTEM SETUP ---
PROTO_FILE = 'Model/colorization_deploy_v2.prototxt'
MODEL_FILE = 'Model/colorization_release_v2.caffemodel'
HULL_PTS = 'Model/pts_in_hull.npy'

# Data and Training Configuration
DATA_DIR = 'ModelData'
TRAINING_LOCK_FILE = os.path.join(DATA_DIR, 'training_in_progress.lock')
TRAINING_LOG_FILE = os.path.join(DATA_DIR, 'training_log.txt')
MODEL_SAVE_PATH = os.path.join(DATA_DIR, 'custom_unet_colorizer.pth')
# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Gemini API Configuration
# THE USER HAS CONFIRMED THIS IS THEIR ACTUAL KEY AND WANTS TO USE IT.
API_KEY = "AIzaSyDxVW1Fijgrh44O4zAsOA0jt4C_4ahhEyM" 
API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"

# Check if model files exist
if not (os.path.exists(PROTO_FILE) and os.path.exists(MODEL_FILE) and os.path.exists(HULL_PTS)):
    st.error(
        f"Model files not found. Please ensure the required files are located in the './Model/' directory: "
        f"{PROTO_FILE}, {MODEL_FILE}, and {HULL_PTS}."
    )
    st.stop()

# --- PYTORCH MODEL BLUEPRINT (TRAINING BACKEND) ---

# This structure mirrors the U-Net architecture from the previous response.
# It is defined here so the Python process can access it.

if PYTORCH_AVAILABLE:
    
    # Custom Data Handler for LAB conversion
    class ColorizationDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            # Look for pairs: input_{idx}.png (L channel) and target_{idx}.png (RGB for ground truth)
            self.image_files = [f for f in os.listdir(root_dir) if f.startswith('input_')]
            self.transform = transform
            self.image_size = 256
            # transforms was not imported globally, so it's defined here for clarity
            try:
                self.default_transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
                ])
            except NameError:
                 # Fallback if Image or transforms is not defined
                 self.default_transform = lambda x: x # Identity function as fallback

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            input_name = os.path.join(self.root_dir, f'input_{idx}.png')
            target_name = os.path.join(self.root_dir, f'target_{idx}.png')

            # Load the original color image (target)
            img_rgb = cv2.imread(target_name)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            
            # Apply transformation (PIL for consistency)
            img_pil = Image.fromarray(img_rgb)
            img_pil = self.default_transform(img_pil)
            img_rgb_final = np.array(img_pil)
            
            # Convert to LAB
            img_lab = cv2.cvtColor(img_rgb_final, cv2.COLOR_RGB2LAB)

            # Normalization
            L = img_lab[:, :, 0] / 100.0          # L channel is normalized to [0, 1]
            ab = img_lab[:, :, 1:3] / 128.0      # a and b channels are normalized to [-1, 1]

            # Convert to PyTorch tensors
            L_tensor = torch.from_numpy(L).float().unsqueeze(0)  # (1, H, W)
            ab_tensor = torch.from_numpy(ab).float().permute(2, 0, 1) # (2, H, W)

            return L_tensor, ab_tensor

    # U-Net Model Blocks (Simplified for brevity, but the structure is here)
    class ConvBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_c)
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x): return self.relu(self.bn(self.conv(x)))

    class DecoderBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
            self.conv = ConvBlock(out_c * 2, out_c) 
        def forward(self, x, skip_connection):
            x = self.up(x)
            x = torch.cat([skip_connection, x], dim=1) 
            return self.conv(x)

    class ColorizationUNet(nn.Module):
        def __init__(self, in_channels=1, out_channels=2):
            super().__init__()
            self.initial_conv = ConvBlock(in_channels, 64)
            self.down1 = ConvBlock(64, 128); self.pool1 = nn.MaxPool2d(2) 
            self.down2 = ConvBlock(128, 256); self.pool2 = nn.MaxPool2d(2)
            self.down3 = ConvBlock(256, 512); self.pool3 = nn.MaxPool2d(2)
            self.down4 = ConvBlock(512, 1024); self.pool4 = nn.MaxPool2d(2)
            self.bridge = ConvBlock(1024, 2048)
            self.up1 = DecoderBlock(2048, 1024)
            self.up2 = DecoderBlock(1024, 512)
            self.up3 = DecoderBlock(512, 256)
            self.up4 = DecoderBlock(256, 128)
            self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)
            self.final_activation = nn.Tanh()

        def forward(self, x):
            x0 = self.initial_conv(x)
            x1 = self.pool1(x0); x1 = self.down1(x1)
            x2 = self.pool2(x1); x2 = self.down2(x2)
            x3 = self.pool3(x2); x3 = self.down3(x3)
            x4 = self.pool4(x3); x4 = self.down4(x4)
            bridge = self.bridge(x4)
            u1 = self.up1(bridge, x4)
            u2 = self.up2(u1, x3)
            u3 = self.up3(u2, x2)
            u4 = self.up4(u3, x1)
            return self.final_activation(self.final_conv(u4))


def train_model_process(data_dir, model_save_path, lock_file, log_file):
    """
    The actual training function designed to run in a separate process.
    """
    try:
        if not PYTORCH_AVAILABLE:
            with open(log_file, "a") as f:
                f.write(f"\n--- Simulation training started at {time.ctime()} ---\n")
                f.write("PyTorch not installed. Simulating 10 second training.\n")
            time.sleep(10) # Simulate training time
            
            # Simulated cleanup
            for filename in os.listdir(data_dir):
                file_path = os.path.join(data_dir, filename)
                if os.path.isfile(file_path) and filename not in [os.path.basename(model_save_path), os.path.basename(log_file), os.path.basename(lock_file)]:
                    os.unlink(file_path)

            with open(log_file, "a") as f: f.write(f"Simulation training finished. Data deleted.\n")
            return
            
        # 1. Setup Logging
        with open(log_file, "a") as f:
            f.write(f"\n--- Training started at {time.ctime()} ---\n")
            
        # 2. Check Data
        dataset = ColorizationDataset(root_dir=data_dir)
        if len(dataset) < 1:
             with open(log_file, "a") as f: f.write("No data found to train. Aborting.\n")
             return

        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        
        # 3. Model, Loss, Optimizer Initialization
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ColorizationUNet().to(DEVICE)
        criterion = nn.SmoothL1Loss().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 4. Training Loop (Simulated as 3 epochs for responsiveness)
        NUM_EPOCHS = 3 
        
        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            
            for batch_idx, (L_input, ab_target) in enumerate(dataloader):
                L_input = L_input.to(DEVICE)
                ab_target = ab_target.to(DEVICE)

                optimizer.zero_grad()
                ab_pred = model(L_input)
                loss = criterion(ab_pred, ab_target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_epoch_loss = running_loss / len(dataloader)
            
            with open(log_file, "a") as f: 
                f.write(f"Epoch {epoch+1}/{NUM_EPOCHS} completed. Avg Loss: {avg_epoch_loss:.6f}\n")
            
        # 5. Final Save and Cleanup
        torch.save(model.state_dict(), model_save_path)
        
        # CLEANUP: Delete the data folder contents after successful training
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path) and filename not in [os.path.basename(model_save_path), os.path.basename(log_file), os.path.basename(lock_file)]:
                os.unlink(file_path)
        
        with open(log_file, "a") as f: f.write(f"Training finished successfully. Data deleted. Model saved to {model_save_path}.\n")
        
    except Exception as e:
        with open(log_file, "a") as f: f.write(f"Training failed: {e}\n")
    finally:
        # IMPORTANT: Remove the lock file so the process can be restarted
        if os.path.exists(lock_file):
            os.remove(lock_file)


def start_background_training():
    """
    Checks for lock file and starts the training process if not already running.
    """
    if os.path.exists(TRAINING_LOCK_FILE):
        return "Training is already in progress.", False
    
    # Check if there is data to train on (must have at least one input file)
    if not any(f.startswith('input_') for f in os.listdir(DATA_DIR)):
         return "No new input/output pairs found in ModelData/ to start training.", False

    # Create lock file
    with open(TRAINING_LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))

    # Start the training process in the background
    p = multiprocessing.Process(
        target=train_model_process, 
        args=(DATA_DIR, MODEL_SAVE_PATH, TRAINING_LOCK_FILE, TRAINING_LOG_FILE)
    )
    p.start()
    return f"Model training (PID: {p.pid}) started in the background. Check the log file for updates.", True


def save_image_data(original_image_bgr, colorized_image_bgr):
    """Saves B&W input and Color RGB output pairs to the data directory."""
    
    # 1. Find next index for image pair
    existing_files = [f for f in os.listdir(DATA_DIR) if f.startswith('input_')]
    next_idx = len(existing_files)
    
    # 2. Convert original BGR image to B&W L channel (input)
    # The U-Net input is the L channel, but saving BGR/RGB helps visualize the pair
    original_grayscale = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2GRAY)
    original_grayscale_bgr = cv2.cvtColor(original_grayscale, cv2.COLOR_GRAY2BGR)

    # 3. Save input (grayscale BGR for simple viewing) and target (color BGR)
    input_path = os.path.join(DATA_DIR, f'input_{next_idx}.png')
    target_path = os.path.join(DATA_DIR, f'target_{next_idx}.png')
    
    cv2.imwrite(input_path, original_grayscale_bgr)
    cv2.imwrite(target_path, colorized_image_bgr)
    
    return f"Data pair saved: {input_path} and {target_path}"

# --- Utility Functions for TTS Conversion (MODIFIED) ---
def base64ToArrayBuffer(b64):
    """Converts a base64 string to a NumPy array buffer."""
    return np.frombuffer(base64.b64decode(b64), dtype=np.int16)

# MODIFIED: Added speed_factor parameter and adjusted sample_rate calculation
def pcmToWav(pcm16, base_sample_rate=16000, speed_factor=1.0):
    """
    Converts 16-bit signed PCM data (NumPy array) into a WAV byte array (BytesIO buffer).
    The effective sample rate is adjusted by the speed_factor for playback speed control.
    """
    buffer = BytesIO()
    
    effective_sample_rate = int(base_sample_rate * speed_factor)

    # WAV file headers
    buffer.write(b'RIFF')
    buffer.write((36 + len(pcm16) * 2).to_bytes(4, 'little'))
    buffer.write(b'WAVE')
    buffer.write(b'fmt ')
    buffer.write((16).to_bytes(4, 'little'))    # Subchunk1Size (16 for PCM)
    buffer.write((1).to_bytes(2, 'little'))     # AudioFormat (1 for PCM)
    buffer.write((1).to_bytes(2, 'little'))     # NumChannels (Mono)
    # Use the calculated effective sample rate for playback speed
    buffer.write(effective_sample_rate.to_bytes(4, 'little')) # SampleRate
    # ByteRate calculation: SampleRate * NumChannels * BitsPerSample/8
    buffer.write((effective_sample_rate * 2).to_bytes(4, 'little')) # ByteRate
    buffer.write((2).to_bytes(2, 'little'))     # BlockAlign
    buffer.write((16).to_bytes(2, 'little'))    # BitsPerSample
    buffer.write(b'data')
    buffer.write((len(pcm16) * 2).to_bytes(4, 'little')) # Subchunk2Size
    buffer.write(pcm16.tobytes())
    buffer.seek(0)
    return buffer

def call_api_with_retry(url, payload, max_retries=3, initial_delay=1):
    """Handles API calls with exponential backoff for transient 5xx errors."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if 500 <= e.response.status_code < 600 and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise e
        except requests.exceptions.RequestException as e:
            raise e
            
    raise requests.exceptions.RequestException("Max retries exceeded without successful response.")


# --- Caching the Model Loading (Unchanged) ---
@st.cache_resource
def load_colorization_model():
    """Loads the pre-trained Caffe model."""
    try:
        net = dnn.readNetFromCaffe(PROTO_FILE, MODEL_FILE)
        kernel = np.load(HULL_PTS)
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = kernel.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        return net
    except Exception as e:
        st.error(f"Error loading model files. Check the paths and file integrity: {e}")
        st.stop()


def colorize_image(net, input_image_bgr):
    """Performs the core image colorization logic (Unchanged)."""
    scaled = input_image_bgr.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (input_image_bgr.shape[1], input_image_bgr.shape[0]))
    L = cv2.split(lab_img)[0]
    colorized_lab = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    colorized_bgr = np.clip(colorized_bgr, 0, 1)
    colorized_bgr = (255 * colorized_bgr).astype("uint8")
    return colorized_bgr

# --- NEW FUNCTION FOR PYTORCH INFERENCE (FEATURE 1) ---
def pytorch_colorize_image(model_path, input_image_bgr):
    """Performs colorization using the custom PyTorch U-Net model."""
    if not PYTORCH_AVAILABLE:
        st.error("PyTorch is not installed. Cannot run custom model inference. Returning grayscale.")
        return cv2.cvtColor(cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    if not os.path.exists(model_path):
        st.error(f"Custom model checkpoint not found at: {model_path}. Please train the model first. Returning grayscale.")
        return cv2.cvtColor(cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        
    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ColorizationUNet().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        # 1. Prepare Input (similar to Dataset preparation, but for a single image)
        # Resize to 256x256 for U-Net input
        img_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Consistent transformation size
        transform = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC)])
        img_pil_resized = transform(img_pil)
        
        img_lab_resized = cv2.cvtColor(np.array(img_pil_resized), cv2.COLOR_RGB2LAB)

        L_256 = img_lab_resized[:, :, 0] / 100.0  # L channel normalized to [0, 1]
        L_tensor = torch.from_numpy(L_256).float().unsqueeze(0).unsqueeze(0).to(DEVICE) # (1, 1, 256, 256)

        # 2. Inference
        with torch.no_grad():
            ab_pred_tensor = model(L_tensor).cpu() # (1, 2, 256, 256)
        
        # 3. Post-processing and upsampling
        ab_pred = ab_pred_tensor.squeeze(0).permute(1, 2, 0).numpy() # (256, 256, 2)
        ab_pred = (ab_pred * 128.0).astype(np.int8) # Scale ab back to [-128, 127]
        
        # Resize the predicted ab channels back to the original image size
        ab_channel = cv2.resize(ab_pred, 
                                (input_image_bgr.shape[1], input_image_bgr.shape[0]), 
                                interpolation=cv2.INTER_CUBIC)

        # 4. Re-assemble LAB and convert to BGR
        scaled = input_image_bgr.astype("float32") / 255.0
        lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        L_orig = cv2.split(lab_img)[0] # Get original L channel
        
        # Note: Concatenating np.int8 (ab_channel) with np.uint8 (L_orig) requires type casting.
        # We ensure L_orig is scaled back to LAB range (0-100)
        # However, for cv2.cvtColor to work correctly on LAB space, the L channel must be correctly defined.
        # The L channel from the scaled input (0-1 range) needs to be rescaled or used from the original input
        
        # Let's use the L channel directly from the original LAB conversion
        L_orig_255 = cv2.split(cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2LAB))[0]
        
        # cv2.cvtColor expects L to be 0-255 if the input type is uint8 (which we need for BGR conversion at the end)
        colorized_lab = np.concatenate((L_orig_255[:, :, np.newaxis], ab_channel.astype(np.int8)), axis=2)
        
        # Convert back to BGR and clip
        colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
        colorized_bgr = np.clip(colorized_bgr, 0, 255).astype("uint8")
        
        return colorized_bgr

    except Exception as e:
        st.error(f"Error during PyTorch custom model inference: {e}")
        return cv2.cvtColor(cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
# --- END NEW FUNCTION ---


def apply_post_processing(image_bgr, selected_filter):
    """Applies selected post-processing filter (Unchanged)."""
    if selected_filter == "Sharpen":
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(image_bgr, -1, kernel)
    
    elif selected_filter == "Detail Enhancement":
        blurred = cv2.GaussianBlur(image_bgr, (0, 0), 5)
        return cv2.addWeighted(image_bgr, 1.5, blurred, -0.5, 0)
    
    return image_bgr

# MODIFIED: Returns raw PCM data array instead of WAV BytesIO
def get_image_description_and_tts(colorized_image_bgr):
    """Uses Gemini Vision to caption the image and Gemini TTS to generate audio."""
    is_success, buffer = cv2.imencode(".png", colorized_image_bgr)
    if not is_success:
        return "Could not encode image for API.", None
        
    base64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")

    caption_prompt = "Describe the image concisely, focusing on the main objects and colors."
    vision_model = "gemini-2.5-flash-preview-09-2025"
    vision_url = f"{API_URL_BASE}{vision_model}:generateContent?key={API_KEY}"
    
    vision_payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": caption_prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64_image
                        }
                    }
                ]
            }
        ],
    }
    
    caption_text = "Error fetching caption or API call skipped."
    
    # Check if API_KEY is set (i.e., not an empty string)
    if API_KEY: 
        try:
            result = call_api_with_retry(vision_url, vision_payload)
            caption_text = result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            st.error(f"Error during Vision API call: {e}")
            caption_text = "Error fetching caption. Check API key and network."
    else:
        st.warning("API key is not set. Using placeholder caption.")
        caption_text = "API key not set. Using placeholder caption."


    tts_model = "gemini-2.5-flash-preview-tts"
    tts_url = f"{API_URL_BASE}{tts_model}:generateContent?key={API_KEY}"
    
    tts_payload = {
        "contents": [{"parts": [{"text": caption_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
            }
        },
        "model": tts_model
    }

    audio_pcm_data = None # This will hold the raw NumPy array
    
    if API_KEY:
        try:
            result = call_api_with_retry(tts_url, tts_payload)
            part = result['candidates'][0]['content']['parts'][0]
            audio_data_b64 = part['inlineData']['data']
            # Only convert B64 to raw PCM data array here
            audio_pcm_data = base64ToArrayBuffer(audio_data_b64) 
            
        except Exception as e:
            st.error(f"Error during TTS API call (audio playback disabled): {e}")
    else:
        st.warning("API key is missing. Text-to-Speech synthesis is disabled.")


    # Return raw PCM data array
    return caption_text, audio_pcm_data

# --- MODIFIED: Increased minimum requested colors to 5 ---
def get_style_analysis(caption_text):
    """Uses Gemini to perform structured analysis of the image's style and era."""
    vision_model = "gemini-2.5-flash-preview-09-2025"
    analysis_url = f"{API_URL_BASE}{vision_model}:generateContent?key={API_KEY}"

    # MODIFIED: Requesting a list of 5 colors
    analysis_prompt = f"Based on this description of a colorized vintage photograph: '{caption_text}', provide a detailed analysis of its likely artistic style and historical era. Infer the color palette based on the description. For the suggested colors, provide a list of 5 dominant colors, including a descriptive name and a corresponding hex code for each."

    style_payload = {
        "contents": [{"parts": [{"text": analysis_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "dominant_style": {"type": "STRING", "description": "The likely artistic style or movement of the photograph (e.g., photojournalism, portraiture, impressionistic, early color photography)."},
                    "historical_era": {"type": "STRING", "description": "The estimated historical period this image depicts or was taken in (e.g., Early 20th Century, Post-War era)."},
                    "suggested_colors": {
                        "type": "ARRAY",
                        # MODIFIED: Explicitly ask for 5 colors in the schema description
                        "description": "A list of 5 dominant color descriptions/names and their corresponding hex codes. Ensure exactly 5 items are returned in this array.",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "name": {"type": "STRING"},
                                "hex_code": {"type": "STRING", "description": "The hex code for the color, e.g., #FFFFFF."}
                            }
                        }
                    }
                }
            }
        },
        "model": vision_model
    }
    
    # MODIFIED: Updated placeholder analysis to include 5 colors
    placeholder_analysis = {
        "dominant_style": "Vintage Photography",
        "historical_era": "Unknown (API key not used)",
        "suggested_colors": [
            {"name": "Sepia Tone", "hex_code": "#704214"}, 
            {"name": "Muted Brown", "hex_code": "#A0522D"}, 
            {"name": "Soft Green", "hex_code": "#8FBC8F"},
            {"name": "Dusty Blue", "hex_code": "#6A8B9A"},
            {"name": "Cream White", "hex_code": "#F5F5DC"}
        ]
    }
    
    if API_KEY:
        try:
            result = call_api_with_retry(analysis_url, style_payload)
            json_string = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(json_string)
            
        except Exception as e:
            st.error(f"Error during Structured Analysis API call. Using placeholder data: {e}")
            return placeholder_analysis
    else:
        st.warning("API key is missing. Style analysis is disabled.")
        return placeholder_analysis

# --- NEW FUNCTION TO VISUALIZE AUDIO WAVE ---
def visualize_audio_wave(audio_bytes_io):
    """
    Reads the WAV file data from the BytesIO buffer and generates a Matplotlib plot
    of the audio waveform. The plot's x-axis automatically reflects the effective
    sample rate (playback speed) set in the WAV header.
    """
    if not AUDIO_PLOTTING_AVAILABLE:
        st.error("Audio visualization requires `matplotlib` and `scipy` to be installed.")
        return

    try:
        # Rewind the buffer to the start to read the WAV data
        audio_bytes_io.seek(0)
        
        # Read WAV data from the BytesIO buffer
        # This function reads the sample rate *written in the header*, which is
        # the base rate multiplied by the speed factor.
        sample_rate, data = wav.read(audio_bytes_io)

        # For stereo, take one channel (it's mono in this case, but good practice)
        if len(data.shape) > 1:
            data = data[:, 0]
        
        # Time array for the x-axis: len(data) / sample_rate = duration at that speed
        time_data = np.linspace(0., len(data) / sample_rate, len(data))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(time_data, data, color='blue')
        ax.set_title("Generated Audio Waveform")
        ax.set_xlabel(f"Time (seconds) - Scale adjusted for playback speed: {16000/sample_rate:.2f}x")
        ax.set_ylabel("Amplitude")
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0)) # Clean up y-axis labels
        plt.tight_layout()
        
        # Display the plot in Streamlit (CSS now targets this component container)
        st.pyplot(fig)
        
        plt.close(fig) # Close the figure to free memory

    except Exception as e:
        st.error(f"Error generating audio waveform visualization: {e}")


# --- MODIFIED: Removed placeholder check for API_KEY ---
def home_page():
    st.title("🏡 Welcome to the Advanced Colorization & AI Analysis App")
    st.markdown("This application is a powerful demonstration of combining **traditional Computer Vision (OpenCV/Caffe)** for a core task (colorization) with **modern Generative AI (Google Gemini)** for advanced analysis, all while integrating **live model fine-tuning (PyTorch U-Net)**.")
    
    st.markdown("---")
    
    st.header("🔑 **Setup Status**")
    # Removed the check for the specific placeholder key and replaced it with a simple check.
    if API_KEY:
        st.success("✅ **API Key is Set.** You are ready to use all features!")
    else:
        st.warning("⚠️ **API Key is Missing.** Sections 2 and 3 may be disabled.")

    st.markdown(
        """
        ### Project Architecture Overview
        * **Core Colorization:** Uses a pre-trained **Caffe deep learning model**.
        * **Custom Fine-Tuning:** Uses a **PyTorch U-Net** structure, enabling you to save results as training data and kick off a non-blocking background process to train a custom colorizer. You can then switch to this custom model in Section 1.
        * **Acoustic Description (Section 2):** Employs the **Gemini Vision model** for image captioning and the **Gemini TTS model** for generating spoken audio.
        * **Style Analysis (Section 3):** Uses the **Gemini Structured Output model** for analyzing the image's style, era, and color palette.
        
        ### Start Here
        Move to **'Section 1: Core Colorization'** in the sidebar to upload an image and begin.
        """
    )
    # --- MODIFIED: Removed manual HTML wrapper. CSS now targets the st.image container for styling. ---
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image("Home_img.png")
    # ---------------------------------------------------
    
def section_1_colorization(net):
    st.title("1️⃣ Core Colorization & Training Data Capture")
    st.markdown("This section executes the main black & white to color conversion and allows you to capture the result to improve a custom model.")
    
    # --- NEW: Model Selection UI (FEATURE 1) ---
    st.subheader("Colorization Model Selection")
    
    # Check if custom model exists
    custom_model_exists = os.path.exists(MODEL_SAVE_PATH)
    
    model_options = {
        "Caffe (Pre-trained)": "caffe",
        "Custom PyTorch U-Net": "pytorch"
    }
    
    # Default to Caffe if the custom model hasn't been trained yet
    if 'model_type' not in st.session_state:
        st.session_state['model_type'] = 'caffe'

    selected_model_name = st.radio(
        "Choose Colorization Engine:",
        options=list(model_options.keys()),
        index=0,
        disabled=not custom_model_exists and st.session_state['model_type'] == 'pytorch',
        key="model_selector"
    )
    
    st.session_state['model_type'] = model_options[selected_model_name]

    if not custom_model_exists and st.session_state['model_type'] == 'pytorch':
        st.warning("Custom PyTorch model is not yet trained! Please save data and start training below.")
        st.session_state['model_type'] = 'caffe'
    
    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Choose a B&W image (.jpg, .jpeg, or .png)", 
        type=["jpg", "jpeg", "png"],
        key="section1_uploader"
    )

    if uploaded_file is not None:
        # Load and process the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Store original image for subsequent steps
        st.session_state['original_image_bgr'] = original_image_bgr

        st.subheader("Processing...")
        
        colorized_image_bgr = None
        
        model_name_for_spinner = selected_model_name if st.session_state['model_type'] == 'pytorch' else "Caffe (Pre-trained)"
        
        with st.spinner(f'Applying {model_name_for_spinner} colorization magic...'):
            # --- MODIFIED: Conditional Colorization Call ---
            if st.session_state['model_type'] == 'caffe':
                colorized_image_bgr = colorize_image(net, original_image_bgr)
            elif st.session_state['model_type'] == 'pytorch':
                colorized_image_bgr = pytorch_colorize_image(MODEL_SAVE_PATH, original_image_bgr)
            # -----------------------------------------------
            
        # --- Post-Colorization Filter Selection ---
        st.subheader("Post-Colorization Filters")
        st.markdown("Apply a computer vision filter to enhance or sharpen the final output.")
        
        filter_options = ["None", "Sharpen", "Detail Enhancement"]
        selected_filter = st.selectbox(
            "Select an enhancement filter to apply:", 
            options=filter_options,
            key="post_filter_select"
        )
        
        final_image_bgr = colorized_image_bgr
        if selected_filter != "None":
             with st.spinner(f"Applying {selected_filter} filter..."):
                 final_image_bgr = apply_post_processing(colorized_image_bgr, selected_filter)
        
        # Store final image for subsequent steps
        st.session_state['final_image_bgr'] = final_image_bgr
        st.session_state['caption_text'] = None # Clear old captions
        st.session_state['analysis_data'] = None # Clear old analysis
        st.session_state['audio_pcm_data'] = None # Clear old audio data
        
        # --- Image Comparison ---
        st.subheader("Results: Visual Output")
        col1, col2 = st.columns(2)
        
        with col1:
            # --- MODIFIED: Removed manual HTML wrapper. CSS now targets the st.image container for styling. ---
            st.image(
                cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB), 
                caption="Original Image (Input)", 
                use_container_width=True,
                width='stretch'
            )
            # --------------------------------------------
        
        with col2:
            # --- MODIFIED: Removed manual HTML wrapper. CSS now targets the st.image container for styling. ---
            st.image(
                cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB),
                caption=f"Colorized Image (Output with {selected_filter} via {model_name_for_spinner})", 
                use_container_width=True,
                width='stretch'
            )
            # --------------------------------------------
            
        # --- DOWNLOAD BUTTON SECTION (Centered across full width) ---
        st.subheader("Download Image")
        
        is_success, buffer = cv2.imencode(".png", final_image_bgr)
        if is_success:
            io_buf = BytesIO(buffer)
            st.download_button(
                label="Download Colorized Image",
                data=io_buf,
                file_name="colorized_image.png",
                mime="image/png",
                key="download_button_final_fix" 
            )
        
        st.markdown("---")
        
        # --- DATA CAPTURE AND BACKGROUND TRAINING UI ---
        st.header("🧠 Background Model Training")
        st.markdown("Save this result as a new input/output pair to fine-tune a custom PyTorch model based on your usage data. Training runs in a non-blocking process.")
        
        # Save the Input/Output Pair
        save_col, status_col = st.columns(2)
        
        if save_col.button("💾 Save this Result for Training Data", key="save_data_btn"):
            with st.spinner('Saving input/output pair...'):
                # Pass original_image_bgr from session state/local variable
                save_msg = save_image_data(st.session_state['original_image_bgr'], final_image_bgr)
                st.session_state['save_msg'] = save_msg
            save_col.success(st.session_state.get('save_msg', 'Data saved successfully!'))

        # Check for training status and provide button
        if 'training_status_msg' not in st.session_state:
             st.session_state['training_status_msg'] = ""
             
        # Check lock file status
        is_training = os.path.exists(TRAINING_LOCK_FILE)
        
        if status_col.button("🚀 Start Custom Model Training (Non-Blocking)", disabled=is_training):
            st.session_state['training_status_msg'], started = start_background_training()
            if started:
                status_col.info(st.session_state['training_status_msg'])
        
        # Display Status
        data_count = len([f for f in os.listdir(DATA_DIR) if f.startswith('input_')])
        
        if is_training:
            status_col.warning(f"Training in progress! Check '{TRAINING_LOG_FILE}' for details.")
        elif custom_model_exists:
             status_col.success(f"Custom model trained and ready to use! You have **{data_count}** new image pairs saved.")
        elif data_count > 0:
            status_col.info(f"Ready to train! You have **{data_count}** new image pairs saved.")
        else:
            status_col.markdown(f"**Data Status:** **0 pairs**. Save some results to begin!")
            
        st.markdown(f"Training Data Folder: `./{DATA_DIR}`. Data will be deleted when training completes.")
        st.markdown("---")
        
    else:
        st.info("Upload a black & white image above to begin the colorization process.")
        # Ensure final image is cleared if a new upload starts or none is present
        st.session_state['final_image_bgr'] = None
        st.session_state['caption_text'] = None
        st.session_state['analysis_data'] = None
        st.session_state['audio_pcm_data'] = None


def section_2_acoustic_description():
    st.title("2️⃣ Acoustic Description (Vision + TTS)")
    st.markdown("This step uses the **Gemini Vision model** to generate a concise caption of the **colorized image**, and then uses the **Gemini Text-to-Speech (TTS) model** to turn that caption into a playable audio file. This feature requires your API key to be set.")
    
    final_image_bgr = st.session_state.get('final_image_bgr')
    
    if final_image_bgr is None:
        st.warning("⚠️ **Section 1: Core Colorization** must be completed first. Please upload and process an image there.")
        return
        
    st.markdown("---")
    
    caption_text = st.session_state.get('caption_text')
    
    # Disable the button if the API key is empty/not set
    run_disabled = not API_KEY
    if run_disabled:
         st.error("Cannot run: API Key is not set or empty. Please check the `API_KEY` variable at the top of the code.")

    if st.button("▶️ Generate Description and Audio", key="run_section_2_btn", disabled=run_disabled):
        st.session_state['caption_text'] = None # Reset for fresh run
        st.session_state['audio_pcm_data'] = None # MODIFIED: Reset the PCM data
        
        with st.spinner('Generating image caption and synthesizing speech...'):
            # MODIFIED: Receives raw PCM data
            caption_text, audio_pcm_data = get_image_description_and_tts(final_image_bgr)
            st.session_state['caption_text'] = caption_text
            # MODIFIED: Store raw PCM data
            st.session_state['audio_pcm_data'] = audio_pcm_data
    
    # Display results if available
    caption_text = st.session_state.get('caption_text')
    # MODIFIED: Retrieve raw PCM data
    audio_pcm_data = st.session_state.get('audio_pcm_data') 
    
    if caption_text:
        st.subheader("Results")
        st.markdown(f"**Generated Caption:** *{caption_text}*")
        
        if audio_pcm_data is not None:
            
            # --- NEW ADDITION: AUDIO SPEED SELECTOR ---
            st.markdown("---")
            speed_options = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
            speed_factor = st.selectbox(
                "Select Playback Speed:", 
                options=speed_options, 
                format_func=lambda x: f"{x}x",
                index=3 # Default to 1.0x
            )
            
            # Generate the final WAV BytesIO buffer with the selected speed
            # The modified pcmToWav uses the speed_factor to adjust the sample rate in the header.
            audio_bytes_io = pcmToWav(audio_pcm_data, speed_factor=speed_factor)
            
            # Display Audio Player (CSS now targets this component container)
            st.subheader("Synthesized Audio")
            
            # --- MODIFIED: Removed manual HTML wrapper. CSS now targets the st.audio container for styling. ---
            audio_bytes_io.seek(0)
            st.audio(audio_bytes_io.read(), format='audio/wav')
            # ----------------------------------------------------

            # --- NEW ADDITION: AUDIO WAVE VISUALIZATION (Placed just below Audio displayed) ---
            st.subheader("Audio Wave Visualization")
            # The visualization function now directly calls st.pyplot, which is targeted by CSS
            visualize_audio_wave(audio_bytes_io)
            # ----------------------------------------------

        else:
            if not run_disabled:
                 st.info("Audio generation failed or was skipped.")
            
def section_3_style_analysis():
    st.title("3️⃣ AI Art & Historical Style Analysis (Structured LLM)")
    st.markdown("This step uses the **Gemini Structured Output model** to perform a detailed analysis of the image. It generates a clean JSON object containing the **Artistic Style**, **Historical Era**, and a **Suggested Color Palette** based on the generated caption from Section 2 (if run, otherwise it analyzes the final image). This feature requires your API key to be set.")
    
    final_image_bgr = st.session_state.get('final_image_bgr')
    
    if final_image_bgr is None:
        st.warning("⚠️ **Section 1: Core Colorization** must be completed first. Please upload and process an image there.")
        return
        
    st.markdown("---")
    
    # Disable the button if the API key is empty/not set
    run_disabled = not API_KEY
    if run_disabled:
         st.error("Cannot run: API Key is not set or empty. Please check the `API_KEY` variable at the top of the code.")

    if st.button("▶️ Perform Style Analysis", key="run_section_3_btn", disabled=run_disabled):
        st.session_state['analysis_data'] = None # Reset for fresh run
        
        # Use the existing caption if available, otherwise get a quick placeholder caption for analysis
        caption_text = st.session_state.get('caption_text')
        if not caption_text:
            with st.spinner('First, generating a temporary caption for structured analysis...'):
                # Note: This calls the function, which now uses the API key without the placeholder check
                # We don't need the audio PCM data here, so we ignore the second return value
                caption_text, _ = get_image_description_and_tts(final_image_bgr) 
                # Do NOT store this temporary caption in session state as a final result
        
        with st.spinner('Analyzing style, era, and suggested palette...'):
            analysis_data = get_style_analysis(caption_text)
            st.session_state['analysis_data'] = analysis_data

    # Display results if available
    analysis_data = st.session_state.get('analysis_data')
    
    if analysis_data:
        st.subheader("Results")
        
        style_col, era_col = st.columns(2)

        with style_col:
            st.markdown(
                '<p style="font-size: 1.1em; font-weight: bold; margin-bottom: 0;">🖼️ Artistic Style</p>', 
                unsafe_allow_html=True
            )
            st.markdown(
                f'<p style="font-size: 0.9em; word-wrap: break-word;">{analysis_data.get("dominant_style", "N/A")}</p>',
                unsafe_allow_html=True
            )

        with era_col:
            st.markdown(
                '<p style="font-size: 1.1em; font-weight: bold; margin-bottom: 0;">🕰️ Historical Era</p>', 
                unsafe_allow_html=True
            )
            st.markdown(
                f'<p style="font-size: 0.9em; word-wrap: break-word;">{analysis_data.get("historical_era", "N/A")}</p>',
                unsafe_allow_html=True
            )
        
        st.markdown("**Suggested Color Palette:**")
        
        suggested_colors = analysis_data.get('suggested_colors', [])
        
        if not isinstance(suggested_colors, list) or (suggested_colors and not isinstance(suggested_colors[0], dict)):
             # Handle cases where the structured output might be slightly malformed
             suggested_colors = [{"name": c, "hex_code": "#f0f0f0"} for c in suggested_colors if isinstance(c, str)]
             if not suggested_colors: 
                 suggested_colors = [{"name": "No Data", "hex_code": "#f0f0f0"}]

        # Ensure we have at least 1 color column for display
        if suggested_colors:
            color_cols = st.columns(len(suggested_colors))
            
            for i, color_obj in enumerate(suggested_colors):
                color_name = color_obj.get('name', 'N/A')
                hex_code = color_obj.get('hex_code', '#f0f0f0')
                
                text_color = "black"
                try:
                    # Simplified luminance calculation for text contrast
                    r = int(hex_code.lstrip('#')[0:2], 16)
                    g = int(hex_code.lstrip('#')[2:4], 16)
                    b = int(hex_code.lstrip('#')[4:6], 16)
                    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                    text_color = "white" if luminance < 0.5 else "black"
                except:
                    pass
                    
                # MODIFIED: Added the 'palette-block' class for the CSS hover effect
                color_html = f'<div class="palette-block" style="background-color: {hex_code}; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #ccc; color: {text_color}; font-weight: bold;">{color_name}<br><span style="font-size: 0.8em; font-weight: normal;">{hex_code}</span></div>'
                
                if len(color_cols) > i:
                    color_cols[i].markdown(color_html, unsafe_allow_html=True)
        else:
             st.info("No color palette data was returned from the analysis.")


# --- Streamlit App UI (Main Execution Block) ---
def main():
    
    st.set_page_config(
        page_title="BW Image Colorizer + Acoustic Description",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # --- CUSTOM CSS STYLING ---
    custom_css = """
    <style>
    /* Center Download Button */
    div.stDownloadButton {
        display: flex;
        justify-content: center;
        width: 100%; 
    }
    
    /* ---------------------------------------------------- */
    /* --- COMMON STYLING FOR IMAGES, AUDIO, & PLOTS --- */
    /* ---------------------------------------------------- */
    
    /* Target all visual elements that need the border/hover effect:
       1. st.image wrapper: div[data-testid^="stImage"] > div:first-child
       2. st.audio wrapper: div[data-testid="stAudio"]
       3. st.pyplot/Matplotlib plot wrapper (the block containing the plot) */
    
    div[data-testid^="stImage"] > div:first-child,
    div[data-testid="stAudio"],
    /* The plot container is tricky. We target the vertical block containing the Matplotlib plot (which is inside stPlotlyChart) */
    div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stPlotlyChart"]) 
    { 
        /* Section (Curved and marked edges) */
        border: 2px solid #ccc;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px; 
        
        /* Hover preparation */
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        box-shadow: none; /* Default state: no shadow */
        overflow: hidden; /* Important for containing children elements */
    }
    
    /* Hover effect: lift-up and shadow */
    div[data-testid^="stImage"] > div:first-child:hover,
    div[data-testid="stAudio"]:hover,
    div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stPlotlyChart"]):hover
    {
        transform: translateY(-5px); /* Lift-up effect */
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2); /* Shadow effect */
    }

    /* ---------------------------------------------------- */
    /* --- NEW: STYLING FOR COLOR PALETTE BLOCKS --- */
    /* ---------------------------------------------------- */
    
    .palette-block {
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        box-shadow: none;
    }
    .palette-block:hover {
        transform: translateY(-5px); /* Lift-up effect */
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2); /* Shadow effect */
    }
    
    /* ---------------------------------------------------- */
    /* --- SPECIFIC FIXES --- */
    /* ---------------------------------------------------- */

    /* Ensure audio player fits inside the container */
    div[data-testid="stAudio"] audio {
        width: 100%;
    }
    
    /* Center the home image which is inside the column (Home_img.png) */
    /* This targets the div that contains the image column content to center it */
    .st-emotion-cache-1cpx4g8 { 
        display: flex;
        justify-content: center;
        padding-top: 0px !important; 
    }
    
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    # --------------------------
    
    st.title("🎨 Advanced Colorization with AI Analysis")
    
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state['page'] = 'Home'
        
    # Initialize session state for image data
    if 'final_image_bgr' not in st.session_state:
        st.session_state['final_image_bgr'] = None
    if 'original_image_bgr' not in st.session_state:
        st.session_state['original_image_bgr'] = None
    if 'caption_text' not in st.session_state:
         st.session_state['caption_text'] = None
    if 'analysis_data' not in st.session_state:
         st.session_state['analysis_data'] = None
    # NEW/MODIFIED: State variable to hold the raw audio data array
    if 'audio_pcm_data' not in st.session_state: 
         st.session_state['audio_pcm_data'] = None
    # NEW: State variable for model type selection
    if 'model_type' not in st.session_state:
         st.session_state['model_type'] = 'caffe'


    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    
    pages = {
        "Home": home_page,
        "Section 1: Core Colorization": lambda: section_1_colorization(load_colorization_model()),
        "Section 2: Acoustic Description": section_2_acoustic_description,
        "Section 3: Style Analysis": section_3_style_analysis,
    }

    selected_page = st.sidebar.radio("Select Feature:", list(pages.keys()), index=list(pages.keys()).index(st.session_state['page']))
    
    # Update state and run the selected function
    st.session_state['page'] = selected_page
    pages[selected_page]()


if __name__ == "__main__":
    # Ensure multiprocessing starts correctly in a Streamlit environment
    multiprocessing.freeze_support()
    main()