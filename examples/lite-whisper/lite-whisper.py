import argparse
import ctranslate2
import librosa
import transformers
import torch

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper CTranslate2 model.")
    parser.add_argument(
        "--audio_path",
        type=str,
        default="../../../../LiteASR/harvard.wav",
        help="Path to the audio file."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="whisper-tiny",
        help="CTranslate2 Whisper model name."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the model on (default: cpu)."
    )
    args = parser.parse_args()

    print(f"==== Lite Whisper ====")
    audio, _ = librosa.load(args.audio_path, sr=16000)

    # Load processor
    processor = transformers.WhisperProcessor.from_pretrained(f"openai/{args.model_name}")
    input_features = processor(audio, sampling_rate=16000, return_tensors="np").input_features
    features = ctranslate2.StorageView.from_array(input_features)

    # Load the CTranslate2 model
    model = ctranslate2.models.Whisper(f"lite-{args.model_name}-ct2", device=args.device)

    # Detect language
    results = model.detect_language(features)
    language, probability = results[0][0]
    print(f"Detected language {language} with probability {probability:.4f}")

    # Prepare prompt
    prompt = processor.tokenizer.convert_tokens_to_ids(
        [
            "<|startoftranscript|>",
            language,
            "<|transcribe|>",
            "<|notimestamps|>",  # Remove if timestamps are needed
        ]
    )

    # Run generation
    results = model.generate(features, [prompt])
    transcription = processor.decode(results[0].sequences_ids[0])
    print(f"Transcription:\n{transcription}")
    print(f"=======================================")

if __name__ == "__main__":
    main()
