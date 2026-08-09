from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class KokoroTTSHandlerArguments:
    kokoro_model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The Kokoro TTS model to use. Auto-selects based on device: 'mlx-community/Kokoro-82M-bf16' for MPS, 'hexgrad/Kokoro-82M' for CUDA/CPU."
        },
    )
    kokoro_device: str = field(
        default="auto",
        metadata={
            "help": "The device to run Kokoro TTS on. Options: 'auto', 'cuda', 'cpu', 'mps'. Default is 'auto' (MPS on Mac, CUDA on GPU systems)."
        },
    )
    kokoro_voice: str = field(
        default="bm_fable",
        metadata={
            "help": "The voice to use for synthesis. See VOICES.md in the Kokoro repo for options. Default is 'bm_fable' (British male)."
        },
    )
    kokoro_lang_code: str = field(
        default="b",
        metadata={
            "help": "Language code: 'a' for American English, 'b' for British English, 'j' for Japanese, etc. Default is 'b'."
        },
    )
    kokoro_speed: float = field(
        default=1.0,
        metadata={"help": "Speech speed multiplier. Values > 1.0 speed up, < 1.0 slow down. Default is 1.0."},
    )
    kokoro_temperature: float = field(
        default=0.667,
        metadata={
            "help": "Sampling temperature for Kokoro TTS. Higher values increase randomness/diversity, lower values make output more deterministic. Range: 0.1-2.0. Default is 0.667."
        },
    )
    kokoro_pitch: float = field(
        default=1.0,
        metadata={
            "help": "Pitch multiplier for Kokoro TTS. Values > 1.0 increase pitch, < 1.0 decrease pitch. Range: 0.5-2.0. Default is 1.0."
        },
    )
    kokoro_blocksize: int = field(
        default=512,
        metadata={"help": "The audio chunk size in samples for streaming output. Default is 512."},
    )
    # Kanade voice cloning options
    kokoro_kanade_enabled: bool = field(
        default=False,
        metadata={
            "help": "Enable Kanade voice conversion. When True, converts Kokoro TTS output to match the reference voice. Requires kanade_tokenizer package."
        },
    )
    kokoro_kanade_model: str = field(
        default="epheam/Kanade",
        metadata={
            "help": "Kanade model to use for voice conversion. Default is 'epheam/Kanade'."
        },
    )
    kokoro_reference_audio: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to reference audio file for Kanade voice cloning. If not set, looks for reference.wav in project directory. Required when kanade_enabled=True."
        },
    )
    kokoro_phoneme_mode: Literal["auto", "phoneme", "raw"] = field(
        default="auto",
        metadata={
            "help": "Phoneme mode for Kokoro TTS. 'auto' uses automatic phonemization, 'phoneme' accepts raw phoneme strings, 'raw' passes text as-is. Default is 'auto'."
        },
    )
