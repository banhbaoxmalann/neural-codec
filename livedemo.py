import os
import time
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
from glob import glob

# ==========================================
# 1. CORE ARCHITECTURE (PURE NUMPY)
# ==========================================

class Conv1d:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, lr=0.001):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr = lr

        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.kernel = np.random.randn(out_channels, in_channels, kernel_size) * scale
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        batch, in_c, length = x.shape
        out_len = (length + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = np.zeros((batch, self.out_channels, out_len))
        x_pad = np.pad(x, ((0,0), (0,0), (self.padding, self.padding)), mode='constant')

        for b in range(batch):
            for o in range(self.out_channels):
                for i in range(out_len):
                    start = i * self.stride
                    window = x_pad[b, :, start:start+self.kernel_size]
                    out[b, o, i] = np.sum(window * self.kernel[o]) + self.bias[o]
        return out

class ConvTranspose1d:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0, lr=0.001):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.lr = lr

        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.kernel = np.random.randn(in_channels, out_channels, kernel_size) * scale
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        batch, in_c, in_len = x.shape
        out_len = (in_len - 1) * self.stride + self.kernel_size - 2 * self.padding + self.output_padding
        out = np.zeros((batch, self.out_channels, out_len))

        x_dilated = np.zeros((batch, in_c, (in_len-1)*self.stride + 1))
        for b in range(batch):
            for c in range(in_c):
                x_dilated[b, c, ::self.stride] = x[b, c]

        for b in range(batch):
            for o in range(self.out_channels):
                for i in range(in_len):
                    start = i * self.stride
                    for k in range(self.kernel_size):
                        pos = start + k - self.padding
                        if 0 <= pos < out_len:
                            out[b, o, pos] += np.sum(x_dilated[b, :, start] * self.kernel[:, o, k]) + self.bias[o]
        return out

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

class Encoder:
    def __init__(self, lr=0.001):
        self.conv1 = Conv1d(1, 32, kernel_size=7, stride=2, padding=1, lr=lr)
        self.relu1 = ReLU()
        self.conv2 = Conv1d(32, 64, kernel_size=7, stride=2, padding=1, lr=lr)
        self.relu2 = ReLU()
        self.conv3 = Conv1d(64, 128, kernel_size=7, stride=2, padding=1, lr=lr)
        self.relu3 = ReLU()

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.conv3.forward(out)
        out = self.relu3.forward(out)
        return out

class VectorQuantization:
    def __init__(self, num_embeddings, embedding_dim, lr=0.01):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.codebook = np.random.randn(embedding_dim, num_embeddings)
        self.codebook /= np.linalg.norm(self.codebook, axis=0, keepdims=True)

    def forward(self, z_e):
        batch, dim, length = z_e.shape
        z_e_flat = z_e.transpose(0,2,1).reshape(-1, dim)

        z_e_sq = np.sum(z_e_flat**2, axis=1, keepdims=True)
        codebook_sq = np.sum(self.codebook**2, axis=0, keepdims=True)
        dist = z_e_sq + codebook_sq - 2 * z_e_flat @ self.codebook

        indices = np.argmin(dist, axis=1)
        z_q = self.codebook[:, indices].T
        z_q = z_q.reshape(batch, length, dim).transpose(0,2,1)
        return z_q, indices

class Decoder:
    def __init__(self, lr=0.001):
        self.conv1 = ConvTranspose1d(128, 64, kernel_size=7, stride=2, padding=1, output_padding=0, lr=lr)
        self.relu1 = ReLU()
        self.conv2 = ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=1, output_padding=1, lr=lr)
        self.relu2 = ReLU()
        self.conv3 = ConvTranspose1d(32, 1,  kernel_size=7, stride=2, padding=1, output_padding=1, lr=lr)
        self.relu3 = ReLU()

    def forward(self, z_q):
        out = self.conv1.forward(z_q)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.conv3.forward(out)
        out = self.relu3.forward(out)
        return out

class NeuralCodec:
    def __init__(self, num_embeddings=512, embedding_dim=128, lr=0.001):
        self.encoder = Encoder(lr=lr)
        self.decoder = Decoder(lr=lr)
        self.vq = VectorQuantization(num_embeddings, embedding_dim, lr=lr)

    def forward(self, x):
        z_e = self.encoder.forward(x)
        z_q, indices = self.vq.forward(z_e)
        recon = self.decoder.forward(z_q)
        return recon, z_q, z_e, indices

    def load(self, save_dir):
        enc_file = os.path.join(save_dir, f"encoder_final.npz")
        dec_file = os.path.join(save_dir, f"decoder_final.npz")
        vq_file = os.path.join(save_dir, f"vq_codebook.npy")

        if os.path.exists(enc_file):
            data = np.load(enc_file)
            self.encoder.conv1.kernel = data['conv1_kernel']
            self.encoder.conv1.bias = data['conv1_bias']
            self.encoder.conv2.kernel = data['conv2_kernel']
            self.encoder.conv2.bias = data['conv2_bias']
            self.encoder.conv3.kernel = data['conv3_kernel']
            self.encoder.conv3.bias = data['conv3_bias']

        if os.path.exists(dec_file):
            data = np.load(dec_file)
            self.decoder.conv1.kernel = data['conv1_kernel']
            self.decoder.conv1.bias = data['conv1_bias']
            self.decoder.conv2.kernel = data['conv2_kernel']
            self.decoder.conv2.bias = data['conv2_bias']
            self.decoder.conv3.kernel = data['conv3_kernel']
            self.decoder.conv3.bias = data['conv3_bias']

        if os.path.exists(vq_file):
            self.vq.codebook = np.load(vq_file)

# ==========================================
# 2. INIT & QUANTITATIVE METRICS
# ==========================================

@st.cache_resource
def load_pretrained_model():
    m = NeuralCodec(num_embeddings=128, embedding_dim=128, lr=0.001)
    # UPDATED: Assuming the weights folder is in the same directory as app.py
    WEIGHTS_PATH = "/home/anh_202414610/Desktop/nen1/weight" 
    try:
        m.load(WEIGHTS_PATH)
        return m, True
    except Exception as e:
        return m, False

model, load_success = load_pretrained_model()

ORIG_SAMPLE_RATE = 8000
AUDIO_BIT_DEPTH = 16
AUDIO_CHANNELS = 1
VQ_NUM_EMBEDDINGS = 128
VQ_EMBEDDING_DIM = 128
ENCODER_TOTAL_STRIDE = 8

def calculate_codec_metrics(original_sample_rate, bit_depth, num_audio_channels,
                            vq_num_embeddings, vq_embedding_dim,
                            encoder_total_stride, bitrate_retention_pct,
                            orig_audio=None, recon_audio=None):

    original_bps = original_sample_rate * bit_depth * num_audio_channels
    latent_frame_rate = original_sample_rate / encoder_total_stride
    bits_per_latent_dimension = np.log2(max(2, vq_num_embeddings))
    compressed_bps = latent_frame_rate * bits_per_latent_dimension

    compression_ratio = original_bps / compressed_bps if compressed_bps > 0 else float('inf')

    snr = None
    mse = None
    if orig_audio is not None and recon_audio is not None:
        sig_pwr = np.mean(orig_audio ** 2)
        noise_pwr = np.mean((orig_audio - recon_audio) ** 2)
        mse = noise_pwr
        snr = 10 * np.log10(sig_pwr / noise_pwr) if noise_pwr > 0 else float('inf')

    comp_ratio_str = f"{round(compression_ratio)}:1" if compression_ratio != float('inf') else "N/A (0 bps)"
    bitrate_str = f"{compressed_bps:.0f} bps"

    return comp_ratio_str, bitrate_str, compressed_bps, snr, mse

# ==========================================
# 3. STREAMLIT WEB UI CONTROLS
# ==========================================

st.set_page_config(layout="wide")
st.title("🎛️ Neural Audio Codec Interactive Demo")

if load_success:
    st.success(" SUCCESS: Pre-trained NumPy weights loaded globally!")
else:
    st.error(" WARNING: Failed to load weights. Running with random initialization.")

# UPDATED: Paths adjusted to point to local relative dataset folder
CAT_FOLDER_PATH = '/home/anh_202414610/Desktop/nen1/DvC/Cats'
DOG_FOLDER_PATH = '/home/anh_202414610/Desktop/nen1/DvC/Dogs'

cat_files = []
if os.path.exists(CAT_FOLDER_PATH):
    cat_files = [os.path.join(CAT_FOLDER_PATH, f) for f in os.listdir(CAT_FOLDER_PATH) if f.lower().endswith('.wav')]

dog_files = []
if os.path.exists(DOG_FOLDER_PATH):
    dog_files = [os.path.join(DOG_FOLDER_PATH, f) for f in os.listdir(DOG_FOLDER_PATH) if f.lower().endswith('.wav')]

cat_options = {}
for i in range(min(15, len(cat_files))):
    cat_options[f"🐈 Cat Sample {i+1}"] = cat_files[i]

dog_options = {}
for i in range(min(15, len(dog_files))):
    dog_options[f"🐕 Dog Sample {i+1}"] = dog_files[i]

col1, col2 = st.columns(2)
with col1:
    audio1_choice = st.selectbox(
        "Select Audio 1:", 
        list(cat_options.keys()), 
        index=0
    )
    path1 = cat_options[audio1_choice] if cat_options else None
with col2:
    audio2_choice = st.selectbox(
        "Select Audio 2 (Dog):", 
        list(dog_options.keys()), 
        index=0)
    path2 = dog_options[audio2_choice] if dog_options else None

bitrate_retention = st.slider("Bitrate Retention Simulation (%)", min_value=0, max_value=100, value=100, step=10)

def process_audio(filepath, retention_pct):
    if not filepath or not os.path.exists(filepath):
         st.error(f"File not found: {filepath}. Please ensure your dataset is properly extracted.")
         return None, None, 0, 8000
         
    sr = 8000
    length = 2048
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-8)
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    start = len(y) // 2 - length // 2
    orig_audio = y[start:start+length]

    audio_input = np.expand_dims(orig_audio, axis=(0, 1))

    z_e = model.encoder.forward(audio_input)
    z_q, _ = model.vq.forward(z_e)

    channels_to_keep = int(VQ_EMBEDDING_DIM * (retention_pct / 100.0))
    z_q_degraded = z_q.copy()
    if channels_to_keep < VQ_EMBEDDING_DIM:
        z_q_degraded[:, :, channels_to_keep:] = 0

    recon_audio = model.decoder.forward(z_q_degraded).squeeze()
    return orig_audio, recon_audio, channels_to_keep, sr

# ==========================================
# 4. EXECUTION & VISUALIZATION PIPELINE
# ==========================================

if st.button("Run Encode / Decode Comparison"):
    with st.spinner("Executing mathematical computations on CPU..."):
        start_time = time.time()

        orig1, recon1, ch_kept1, sr = process_audio(path1, bitrate_retention)
        orig2, recon2, ch_kept2, _ = process_audio(path2, bitrate_retention)

        if orig1 is None or orig2 is None:
             st.stop()

        process_time_ms = ((time.time() - start_time) / 2) * 1000

        comp_ratio_display1, bitrate_display, _, snr1, mse1 = calculate_codec_metrics(
            ORIG_SAMPLE_RATE, AUDIO_BIT_DEPTH, AUDIO_CHANNELS,
            VQ_NUM_EMBEDDINGS, VQ_EMBEDDING_DIM,
            ENCODER_TOTAL_STRIDE, bitrate_retention,
            orig_audio=orig1, recon_audio=recon1
        )
        _, _, _, snr2, mse2 = calculate_codec_metrics(
            ORIG_SAMPLE_RATE, AUDIO_BIT_DEPTH, AUDIO_CHANNELS,
            VQ_NUM_EMBEDDINGS, VQ_EMBEDDING_DIM,
            ENCODER_TOTAL_STRIDE, bitrate_retention,
            orig_audio=orig2, recon_audio=recon2
        )

        st.markdown("---")
        st.subheader("📊 Experimental Results (Quantitative)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(label="Compression Ratio", value=comp_ratio_display1, delta="Calculated")
        m2.metric(label="Target Bitrate", value=bitrate_display, delta="Calculated")
        m3.metric(label="Latency (CPU)", value=f"{process_time_ms:.0f} ms", delta="NumPy Backend", delta_color="off")
        m4.metric(label="Average SNR", value=f"{(snr1+snr2)/2:.2f} dB", delta="Negative (Needs GPU)", delta_color="inverse")
        m5.metric(label="Average MSE", value=f"{(mse1+mse2)/2:.4f}", delta="Lower is better", delta_color="inverse")
        st.markdown("---")

        st.info(f"⚙️ Active Latent Channels: {ch_kept1}/{VQ_EMBEDDING_DIM}")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(f"**Audio 1 ({audio1_choice})**")
            st.write("Original Audio")
            st.audio(orig1, sample_rate=sr)
            st.write("Reconstructed Audio")
            st.audio(recon1, sample_rate=sr)

        with res_col2:
            st.markdown(f"**Audio 2 ({audio2_choice})**")
            st.write("Original Audio")
            st.audio(orig2, sample_rate=sr)
            st.write("Reconstructed Audio")
            st.audio(recon2, sample_rate=sr)

        st.subheader("Waveforms & Spectrograms (Qualitative Analysis)")
        fig, axs = plt.subplots(4, 2, figsize=(15, 12))

        axs[0, 0].plot(orig1, color='blue')
        axs[0, 0].set_title("Audio 1 - Original Signal")
        axs[0, 1].plot(recon1, color='orange', linestyle='--')
        axs[0, 1].set_title(f"Audio 1 - Reconstructed ({bitrate_retention}%)")

        axs[1, 0].plot(orig2, color='blue')
        axs[1, 0].set_title("Audio 2 - Original Signal")
        axs[1, 1].plot(recon2, color='orange', linestyle='--')
        axs[1, 1].set_title(f"Audio 2 - Reconstructed ({bitrate_retention}%)")

        D1_orig = librosa.amplitude_to_db(np.abs(librosa.stft(orig1)), ref=np.max)
        librosa.display.specshow(D1_orig, sr=sr, ax=axs[2, 0], cmap='magma')
        axs[2, 0].set_title("Audio 1 - Original Spectrogram")

        D1_recon = librosa.amplitude_to_db(np.abs(librosa.stft(recon1)), ref=np.max)
        librosa.display.specshow(D1_recon, sr=sr, ax=axs[2, 1], cmap='magma')
        axs[2, 1].set_title("Audio 1 - Reconstructed Spectrogram")

        D2_orig = librosa.amplitude_to_db(np.abs(librosa.stft(orig2)), ref=np.max)
        librosa.display.specshow(D2_orig, sr=sr, ax=axs[3, 0], cmap='magma')
        axs[3, 0].set_title("Audio 2 - Original Spectrogram")

        D2_recon = librosa.amplitude_to_db(np.abs(librosa.stft(recon2)), ref=np.max)
        librosa.display.specshow(D2_recon, sr=sr, ax=axs[3, 1], cmap='magma')
        axs[3, 1].set_title("Audio 2 - Reconstructed Spectrogram")

        plt.tight_layout()
        st.pyplot(fig)
