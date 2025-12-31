# Parakeet-RS Test Project

This project tests the [parakeet-rs](https://github.com/altunenes/parakeet-rs) speech recognition library with both file transcription and microphone streaming.

## Setup

### 1. Download Model Files

Use the provided download script to get the models:

```bash
./download_models.sh
```

This interactive script lets you choose which models to download:

- **Option 1**: End-of-Utterance (EOU) Model (~120MB) - For streaming speech recognition
- **Option 2**: Speaker Diarization Models - For identifying different speakers (v1: 514MB, v2/v2.1: 492MB each)
- **Option 3**: All models (~5.2GB) - Complete repository including TDT models
- **Option 4**: Minimal setup (~612MB) - EOU + latest diarization (recommended for testing)

Models are downloaded from: `https://huggingface.co/altunenes/parakeet-rs`

Alternatively, you can manually download using git-lfs:
```bash
git lfs install
git clone https://huggingface.co/altunenes/parakeet-rs models
```

### 2. Build the Project

```bash
cargo build --release
```

**Note:** This project uses `parakeet-rs v0.2.x` which is compatible with the models from `altunenes/parakeet-rs` repository.

## Usage

### Test File Transcription (Measure Speed)

```bash
cargo run --release -- file path/to/audio.wav
```

This will:
- Load the Parakeet model
- Transcribe the audio file
- Display the transcription with token-level timestamps
- Show processing time to measure speed

### Test Microphone Streaming

```bash
cargo run --release -- mic [duration_seconds]
```

Examples:
```bash
cargo run --release -- mic 5    # Record for 5 seconds
cargo run --release -- mic 10   # Record for 10 seconds (default)
```

This will:
- Capture audio from your default microphone
- Convert to 16kHz mono WAV format (required by Parakeet)
- Transcribe the recorded audio
- Display the transcription

## Features

- **File transcription** with timing benchmarks
- **Microphone streaming** with automatic resampling to 16kHz mono
- **Token-level timestamps** for detailed timing info
- **Multi-format support** for both F32 and I16 audio samples
- **Error handling** with helpful messages

## Requirements

- Audio files must be 16kHz mono WAV format (the mic mode handles conversion automatically)
- Model files must be downloaded from HuggingFace
- Microphone access for streaming mode

## Performance Tips

The library supports GPU acceleration with:
- CUDA
- TensorRT
- WebGPU
- DirectML
- ROCm

With automatic CPU fallback if GPU is not available.

## Troubleshooting

If you encounter issues:

1. **Model not found**: Make sure model files are in the current directory or specify the correct path
2. **No microphone detected**: Check your system's default input device
3. **Audio format errors**: Ensure WAV files are 16kHz mono for file mode
