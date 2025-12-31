use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{WavSpec, WavWriter};
use parakeet_rs::ParakeetEOU;
use rubato::Resampler;
use std::env;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Parakeet-RS Speech Recognition Test\n");
        println!("Usage:");
        println!("  {} file <audio.wav>     - Transcribe audio file and measure speed", args[0]);
        println!("  {} mic [duration_secs]  - Stream from microphone (default: 10 seconds)", args[0]);
        println!("  {} mic_streaming        - Real-time streaming transcription from mic", args[0]);
        println!("\nNote: You need to download the model files from HuggingFace first!");
        println!("      Place them in the current directory or specify path.");
        return Ok(());
    }

    match args[1].as_str() {
        "file" => {
            if args.len() < 3 {
                println!("Error: Please provide audio file path");
                println!("Usage: {} file <audio.wav>", args[0]);
                return Ok(());
            }
            transcribe_file(&args[2])?;
        }
        "mic" => {
            let duration = if args.len() >= 3 {
                args[2].parse::<u64>().unwrap_or(10)
            } else {
                10
            };
            stream_from_mic(duration)?;
        }
        "mic_streaming" => {
            stream_from_mic_realtime()?;
        }
        _ => {
            println!("Unknown command: {}", args[1]);
            println!("Use 'file', 'mic', or 'mic_streaming'");
        }
    }

    Ok(())
}

fn get_model_path() -> &'static str {
    if Path::new("./models").exists() {
        "./models/realtime_eou_120m-v1-onnx/"
    } else {
        "."
    }
}

fn prepare_audio_file(file_path: &str) -> Result<(String, bool)> {
    let mut reader = hound::WavReader::open(file_path)?;
    let spec = reader.spec();

    println!("Input audio: {} Hz, {} channels, {} bits",
             spec.sample_rate, spec.channels, spec.bits_per_sample);

    if spec.sample_rate == 16000 && spec.channels == 1 {
        println!("Audio already in correct format (16kHz mono)");
        Ok((file_path.to_string(), false))
    } else {
        println!("Need to resample to 16kHz mono...");

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.samples::<f32>().map(|s| s.unwrap()).collect()
            }
            hound::SampleFormat::Int => {
                reader.samples::<i16>().map(|s| {
                    s.unwrap() as f32 / i16::MAX as f32
                }).collect()
            }
        };

        let temp_file = "temp_resampled.wav".to_string();
        save_wav(&samples, spec.sample_rate, spec.channels as usize, &temp_file)?;

        Ok((temp_file, true))
    }
}

fn transcribe_file(file_path: &str) -> Result<()> {
    let model_path = get_model_path();
    println!("Loading Parakeet EOU model from: {}", model_path);
    let model_load_start = Instant::now();

    let mut parakeet = ParakeetEOU::from_pretrained(model_path, None)?;

    let model_load_time = model_load_start.elapsed();
    println!("Model loaded in {:.2}s\n", model_load_time.as_secs_f64());

    let (audio_file, is_temp) = prepare_audio_file(file_path)?;

    println!("Loading audio file: {}", audio_file);
    let mut reader = hound::WavReader::open(&audio_file)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>().map(|s| s.unwrap()).collect()
        }
        hound::SampleFormat::Int => {
            reader.samples::<i16>().map(|s| {
                s.unwrap() as f32 / i16::MAX as f32
            }).collect()
        }
    };

    println!("Transcribing {} samples in streaming chunks...", samples.len());
    let transcribe_start = Instant::now();

    const CHUNK_SIZE: usize = 2560;
    let mut full_transcription = String::new();

    for chunk in samples.chunks(CHUNK_SIZE) {
        let text = parakeet.transcribe(chunk, false)?;
        full_transcription.push_str(&text);
    }

    let transcribe_time = transcribe_start.elapsed();

    println!("\n=== Results ===");
    println!("Transcription: {}", full_transcription);
    println!("\nProcessing time: {:.2}s", transcribe_time.as_secs_f64());

    if is_temp {
        // std::fs::remove_file(&audio_file).ok();
    }

    Ok(())
}

fn stream_from_mic(duration_secs: u64) -> Result<()> {
    println!("Streaming from microphone for {} seconds...", duration_secs);
    println!("Speak clearly into your microphone!\n");

    let host = cpal::default_host();

    println!("Available input devices:");
    for (i, device) in host.input_devices()?.enumerate() {
        if let Ok(name) = device.name() {
            println!("  [{}] {}", i, name);
        }
    }
    println!();

    let device = host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

    println!("Using default input device: {}", device.name()?);

    let config = device.default_input_config()?;
    println!("Sample rate: {} Hz", config.sample_rate().0);
    println!("Channels: {}", config.channels());

    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    let audio_data = Arc::new(Mutex::new(Vec::new()));
    let audio_data_clone = audio_data.clone();

    let err_fn = |err| eprintln!("Error in audio stream: {}", err);

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &_| {
                let mut buffer = audio_data_clone.lock().unwrap();
                buffer.extend_from_slice(data);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data: &[i16], _: &_| {
                let mut buffer = audio_data_clone.lock().unwrap();
                let float_data: Vec<f32> = data.iter()
                    .map(|&s| s as f32 / i16::MAX as f32)
                    .collect();
                buffer.extend_from_slice(&float_data);
            },
            err_fn,
            None,
        )?,
        _ => return Err(anyhow::anyhow!("Unsupported sample format")),
    };

    stream.play()?;

    println!("Recording... ({}s)", duration_secs);

    for i in 0..duration_secs {
        std::thread::sleep(std::time::Duration::from_secs(1));
        let sample_count = audio_data.lock().unwrap().len();
        println!("  {}s: {} samples captured", i + 1, sample_count);
    }

    drop(stream);
    println!("Recording finished!\n");

    let audio_samples = audio_data.lock().unwrap().clone();

    println!("DEBUG: Raw captured samples: {}", audio_samples.len());
    println!("DEBUG: Sample rate: {} Hz, Channels: {}", sample_rate, channels);

    if audio_samples.is_empty() {
        return Err(anyhow::anyhow!("No audio data captured"));
    }

    println!("Captured {} samples ({:.2} seconds of audio)",
             audio_samples.len(),
             audio_samples.len() as f32 / (sample_rate * channels as u32) as f32);

    let non_zero_samples = audio_samples.iter().filter(|&&s| s.abs() > 0.001).count();
    println!("Non-silent samples: {} ({:.1}%)",
             non_zero_samples,
             (non_zero_samples as f32 / audio_samples.len() as f32) * 100.0);

    if non_zero_samples == 0 {
        println!("\nWARNING: All samples are silent!");
        println!("Check that:");
        println!("  1. Your microphone is not muted");
        println!("  2. The correct input device is selected as default");
        println!("  3. Microphone permissions are granted");
    }

    let temp_file = "temp_recording.wav";
    save_wav(&audio_samples, sample_rate, channels, temp_file)?;

    println!("Saved temporary file: {}\n", temp_file);

    let model_path = get_model_path();
    println!("Loading Parakeet EOU model from: {}", model_path);

    let mut parakeet = ParakeetEOU::from_pretrained(model_path, None)?;

    println!("Transcribing recorded audio...");

    let mut reader = hound::WavReader::open(temp_file)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>().map(|s| s.unwrap()).collect()
        }
        hound::SampleFormat::Int => {
            reader.samples::<i16>().map(|s| {
                s.unwrap() as f32 / i16::MAX as f32
            }).collect()
        }
    };

    const CHUNK_SIZE: usize = 2560;
    let mut full_transcription = String::new();

    for chunk in samples.chunks(CHUNK_SIZE) {
        let text = parakeet.transcribe(chunk, false)?;
        full_transcription.push_str(&text);
    }

    println!("\n=== Transcription Result ===");
    println!("{}", full_transcription);

    // std::fs::remove_file(temp_file).ok();

    Ok(())
}

fn stream_from_mic_realtime() -> Result<()> {
    println!("Real-time microphone streaming transcription");
    println!("Press Ctrl+C to stop\n");

    // Load model first
    let model_path = get_model_path();
    println!("Loading Parakeet EOU model from: {}", model_path);
    let model_load_start = Instant::now();

    let mut parakeet = ParakeetEOU::from_pretrained(model_path, None)?;

    let model_load_time = model_load_start.elapsed();
    println!("Model loaded in {:.2}s\n", model_load_time.as_secs_f64());

    // Set up audio device
    let host = cpal::default_host();
    let device = host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

    println!("Using input device: {}", device.name()?);

    let config = device.default_input_config()?;
    println!("Sample rate: {} Hz", config.sample_rate().0);
    println!("Channels: {}\n", config.channels());

    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    // Create channel for passing audio chunks
    let (tx, rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = mpsc::channel();

    // Audio buffer that will be shared with the audio thread
    let audio_buffer = Arc::new(Mutex::new(Vec::new()));
    let audio_buffer_clone = audio_buffer.clone();

    // Chunk size in samples for the original sample rate
    // We want to send chunks frequently enough for real-time feel
    // At 48kHz, 4800 samples = 100ms
    let chunk_duration_ms = 100;
    let chunk_size = (sample_rate * chunk_duration_ms / 1000) as usize * channels;

    let tx_clone = tx.clone();

    let err_fn = |err| eprintln!("Error in audio stream: {}", err);

    // Build audio stream
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &_| {
                let mut buffer = audio_buffer_clone.lock().unwrap();
                buffer.extend_from_slice(data);

                // Send chunks when we have enough data
                while buffer.len() >= chunk_size {
                    let chunk: Vec<f32> = buffer.drain(..chunk_size).collect();
                    tx_clone.send(chunk).ok();
                }
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data: &[i16], _: &_| {
                let mut buffer = audio_buffer_clone.lock().unwrap();
                let float_data: Vec<f32> = data.iter()
                    .map(|&s| s as f32 / i16::MAX as f32)
                    .collect();
                buffer.extend_from_slice(&float_data);

                // Send chunks when we have enough data
                while buffer.len() >= chunk_size {
                    let chunk: Vec<f32> = buffer.drain(..chunk_size).collect();
                    tx_clone.send(chunk).ok();
                }
            },
            err_fn,
            None,
        )?,
        _ => return Err(anyhow::anyhow!("Unsupported sample format")),
    };

    stream.play()?;
    drop(tx); // Drop the original sender so the channel closes when stream ends

    println!("Listening... (speak into your microphone)\n");

    // Process audio chunks as they arrive
    for chunk in rx {
        // Resample to 16kHz mono if needed
        let resampled = if sample_rate != 16000 || channels > 1 {
            resample_to_16khz_mono(&chunk, sample_rate, channels)
        } else {
            chunk
        };

        // The model expects chunks of 2560 samples (160ms at 16kHz)
        // Process the resampled chunk in sub-chunks if needed
        const MODEL_CHUNK_SIZE: usize = 2560;
        for model_chunk in resampled.chunks(MODEL_CHUNK_SIZE) {
            if let Ok(text) = parakeet.transcribe(model_chunk, false) {
                if !text.trim().is_empty() {
                    print!("{}", text);
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                }
            }
        }
    }

    Ok(())
}

fn save_wav(samples: &[f32], sample_rate: u32, channels: usize, filename: &str) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(filename, spec)?;

    let resampled = if sample_rate != 16000 || channels > 1 {
        println!("Resampling: {} Hz ({}ch) -> 16000 Hz (mono)", sample_rate, channels);
        println!("Input samples: {} -> ", samples.len());
        let result = resample_to_16khz_mono(samples, sample_rate, channels);
        println!("Output samples: {}", result.len());
        println!("Duration: {:.2}s", result.len() as f32 / 16000.0);
        result
    } else {
        samples.to_vec()
    };

    for sample in resampled {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    Ok(())
}

fn resample_to_16khz_mono(samples: &[f32], original_rate: u32, channels: usize) -> Vec<f32> {
    let mono_samples = if channels > 1 {
        samples.chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect::<Vec<f32>>()
    } else {
        samples.to_vec()
    };

    if original_rate != 16000 {
        let mut resampler = rubato::SincFixedIn::<f32>::new(
            16000.0 / original_rate as f64,
            2.0,
            rubato::SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: rubato::SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: rubato::WindowFunction::BlackmanHarris2,
            },
            mono_samples.len(),
            1,
        ).expect("Failed to create resampler");

        let waves_in = vec![mono_samples];
        let waves_out = resampler.process(&waves_in, None)
            .expect("Resampling failed");

        waves_out[0].clone()
    } else {
        mono_samples
    }
}
