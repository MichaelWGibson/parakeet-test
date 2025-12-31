#!/bin/bash

set -e

REPO_URL="https://huggingface.co/altunenes/parakeet-rs/resolve/main"
MODEL_DIR="./models"

echo "Parakeet-RS Model Downloader"
echo "============================="
echo ""
echo "This script will download ONNX models from:"
echo "https://huggingface.co/altunenes/parakeet-rs"
echo ""
echo "Models will be saved to: $MODEL_DIR"
echo ""

mkdir -p "$MODEL_DIR"

download_file() {
    local url=$1
    local output=$2
    local description=$3

    echo ""
    echo "Downloading: $description"
    echo "URL: $url"
    echo "Saving to: $output"

    if command -v wget &> /dev/null; then
        wget -c "$url" -O "$output"
    elif command -v curl &> /dev/null; then
        curl -L -C - "$url" -o "$output"
    else
        echo "Error: Neither wget nor curl found. Please install one of them."
        exit 1
    fi

    echo "Downloaded: $description"
}

echo "Which models would you like to download?"
echo ""
echo "1) End-of-Utterance (EOU) Model - For streaming speech recognition"
echo "   Size: ~120MB"
echo ""
echo "2) TDT Model (int8 quantized) - Multilingual with language detection"
echo "   Size: ~710 MB (encoder, decoder, vocab)"
echo ""
echo "3) Speaker Diarization Models - For identifying different speakers"
echo "   - v1: 514 MB"
echo "   - v2: 492 MB (streaming)"
echo "   - v2.1: 492 MB (streaming, latest)"
echo ""
echo "4) All models"
echo "   Total: ~5.2 GB (includes all TDT models)"
echo ""
echo "5) Minimal setup (EOU + latest diarization)"
echo "   Size: ~612 MB"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Downloading End-of-Utterance (EOU) model..."
        mkdir -p "$MODEL_DIR/realtime_eou_120m-v1-onnx"
        download_file \
            "$REPO_URL/realtime_eou_120m-v1-onnx/decoder_joint.onnx" \
            "$MODEL_DIR/realtime_eou_120m-v1-onnx/decoder_joint.onnx" \
            "EOU Decoder Joint Model"
        ;;

    2)
        echo ""
        echo "Downloading TDT Model (int8 quantized)..."
        echo "This multilingual model supports 25 languages with automatic detection."
        mkdir -p "$MODEL_DIR/tdt"

        download_file \
            "$REPO_URL/tdt/encoder-model.int8.onnx" \
            "$MODEL_DIR/tdt/encoder-model.onnx" \
            "TDT Encoder (int8 quantized)"

        download_file \
            "$REPO_URL/tdt/decoder_joint-model.int8.onnx" \
            "$MODEL_DIR/tdt/decoder_joint-model.onnx" \
            "TDT Decoder/Joint (int8 quantized)"

        download_file \
            "$REPO_URL/tdt/vocab.txt" \
            "$MODEL_DIR/tdt/vocab.txt" \
            "TDT Vocabulary"
        ;;

    3)
        echo ""
        echo "Downloading Speaker Diarization models..."

        echo ""
        read -p "Download v1 (514 MB)? [y/N]: " dl_v1
        if [[ "$dl_v1" =~ ^[Yy]$ ]]; then
            download_file \
                "$REPO_URL/diar_sortformer_4spk-v1.onnx" \
                "$MODEL_DIR/diar_sortformer_4spk-v1.onnx" \
                "Diarization Sortformer v1"
        fi

        echo ""
        read -p "Download v2 streaming (492 MB)? [y/N]: " dl_v2
        if [[ "$dl_v2" =~ ^[Yy]$ ]]; then
            download_file \
                "$REPO_URL/diar_streaming_sortformer_4spk-v2.onnx" \
                "$MODEL_DIR/diar_streaming_sortformer_4spk-v2.onnx" \
                "Diarization Streaming Sortformer v2"
        fi

        echo ""
        read -p "Download v2.1 streaming (492 MB, recommended)? [Y/n]: " dl_v21
        if [[ ! "$dl_v21" =~ ^[Nn]$ ]]; then
            download_file \
                "$REPO_URL/diar_streaming_sortformer_4spk-v2.1.onnx" \
                "$MODEL_DIR/diar_streaming_sortformer_4spk-v2.1.onnx" \
                "Diarization Streaming Sortformer v2.1"
        fi
        ;;

    4)
        echo ""
        echo "Downloading ALL models (this will take a while - 5.2 GB)..."
        echo ""
        read -p "Are you sure? This is a large download. [y/N]: " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            if command -v git-lfs &> /dev/null; then
                echo "Using git-lfs for full repository clone..."
                git lfs install
                git clone https://huggingface.co/altunenes/parakeet-rs "$MODEL_DIR"
            else
                echo "git-lfs not found. Downloading individual files..."

                # EOU Model
                mkdir -p "$MODEL_DIR/realtime_eou_120m-v1-onnx"
                download_file \
                    "$REPO_URL/realtime_eou_120m-v1-onnx/decoder_joint.onnx" \
                    "$MODEL_DIR/realtime_eou_120m-v1-onnx/decoder_joint.onnx" \
                    "EOU Decoder Joint Model"

                # TDT Model (int8)
                mkdir -p "$MODEL_DIR/tdt"
                download_file \
                    "$REPO_URL/tdt/encoder-model.int8.onnx" \
                    "$MODEL_DIR/tdt/encoder-model.onnx" \
                    "TDT Encoder (int8 quantized)"

                download_file \
                    "$REPO_URL/tdt/decoder_joint-model.int8.onnx" \
                    "$MODEL_DIR/tdt/decoder_joint-model.onnx" \
                    "TDT Decoder/Joint (int8 quantized)"

                download_file \
                    "$REPO_URL/tdt/vocab.txt" \
                    "$MODEL_DIR/tdt/vocab.txt" \
                    "TDT Vocabulary"

                # Diarization models
                download_file \
                    "$REPO_URL/diar_sortformer_4spk-v1.onnx" \
                    "$MODEL_DIR/diar_sortformer_4spk-v1.onnx" \
                    "Diarization Sortformer v1"

                download_file \
                    "$REPO_URL/diar_streaming_sortformer_4spk-v2.onnx" \
                    "$MODEL_DIR/diar_streaming_sortformer_4spk-v2.onnx" \
                    "Diarization Streaming Sortformer v2"

                download_file \
                    "$REPO_URL/diar_streaming_sortformer_4spk-v2.1.onnx" \
                    "$MODEL_DIR/diar_streaming_sortformer_4spk-v2.1.onnx" \
                    "Diarization Streaming Sortformer v2.1"

                echo ""
                echo "All models downloaded successfully!"
            fi
        else
            echo "Download cancelled."
            exit 0
        fi
        ;;

    5)
        echo ""
        echo "Downloading minimal setup (EOU + latest diarization)..."

        mkdir -p "$MODEL_DIR/realtime_eou_120m-v1-onnx"
        download_file \
            "$REPO_URL/realtime_eou_120m-v1-onnx/decoder_joint.onnx" \
            "$MODEL_DIR/realtime_eou_120m-v1-onnx/decoder_joint.onnx" \
            "EOU Decoder Joint Model"

        download_file \
            "$REPO_URL/diar_streaming_sortformer_4spk-v2.1.onnx" \
            "$MODEL_DIR/diar_streaming_sortformer_4spk-v2.1.onnx" \
            "Diarization Streaming Sortformer v2.1"
        ;;

    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "============================="
echo "Download complete!"
echo ""
echo "Models saved to: $MODEL_DIR"
echo ""
echo "To use these models with parakeet-rs, you may need to:"
echo "1. Update the model path in your code"
echo "2. Check the parakeet-rs documentation for model loading"
echo ""
echo "Run your test with:"
echo "  cargo run --release -- mic 10"
echo ""
