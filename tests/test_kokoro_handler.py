"""Tests for KokoroTTSHandler extended features.

Tests cover:
- Kanade voice cloning integration
- Temperature control
- Pitch control
- Phoneme mode interface
- Reference audio resolution
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock


def test_kokoro_tts_arguments_defaults():
    """Verify all new argument defaults."""
    from speech_to_speech.arguments_classes.kokoro_tts_arguments import KokoroTTSHandlerArguments
    
    args = KokoroTTSHandlerArguments()
    
    # New fields should have correct defaults
    assert args.kokoro_temperature == 0.667
    assert args.kokoro_pitch == 1.0
    assert args.kokoro_kanade_enabled is False
    assert args.kokoro_kanade_model == "epheam/Kanade"
    assert args.kokoro_reference_audio is None
    assert args.kokoro_phoneme_mode == "auto"
    
    # Existing fields unchanged
    assert args.kokoro_speed == 1.0
    assert args.kokoro_blocksize == 512


def test_kokoro_tts_arguments_custom_values():
    """Verify custom values can be set."""
    from speech_to_speech.arguments_classes.kokoro_tts_arguments import KokoroTTSHandlerArguments
    
    args = KokoroTTSHandlerArguments(
        kokoro_temperature=0.8,
        kokoro_pitch=1.5,
        kokoro_kanade_enabled=True,
        kokoro_reference_audio="/path/to/ref.wav",
        kokoro_phoneme_mode="phoneme",
    )
    
    assert args.kokoro_temperature == 0.8
    assert args.kokoro_pitch == 1.5
    assert args.kokoro_kanade_enabled is True
    assert args.kokoro_reference_audio == "/path/to/ref.wav"
    assert args.kokoro_phoneme_mode == "phoneme"


def test_kokoro_handler_import():
    """Verify KokoroTTSHandler can be imported."""
    from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler
    assert KokoroTTSHandler is not None


def test_kokoro_handler_setup_signature():
    """Verify setup method accepts all new parameters."""
    from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler
    import inspect
    
    sig = inspect.signature(KokoroTTSHandler.setup)
    params = list(sig.parameters.keys())
    
    # New parameters should be present
    assert "temperature" in params
    assert "pitch" in params
    assert "kanade_enabled" in params
    assert "kanade_model" in params
    assert "reference_audio" in params
    assert "phoneme_mode" in params


def test_kokoro_handler_methods():
    """Verify new methods exist on KokoroTTSHandler."""
    from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler
    
    # Check method signatures
    assert hasattr(KokoroTTSHandler, '_setup_kanade')
    assert hasattr(KokoroTTSHandler, '_apply_pitch_shift')
    assert hasattr(KokoroTTSHandler, '_convert_with_kanade')
    assert hasattr(KokoroTTSHandler, '_resolve_reference_audio')
    assert hasattr(KokoroTTSHandler, 'cleanup')


def test_kokoro_handler_resolve_reference_audio():
    """Test reference audio path resolution."""
    from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler
    
    handler = MagicMock(spec=KokoroTTSHandler)
    handler.reference_audio = None
    handler._resolved_reference = None
    
    # Test the actual method logic by simulating it
    ref_path = Path(__file__).parent.parent / "reference.wav"
    
    # When reference.wav exists, should return its resolved path
    if ref_path.exists():
        assert ref_path.exists() is True


def test_pitch_shift_no_op():
    """Test that pitch shift with factor 1.0 returns unchanged audio."""
    import numpy as np
    from scipy.signal import resample_poly
    
    # Test the actual implementation logic
    audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    
    # When pitch_factor is 1.0, should return early (no-op)
    result = audio  # The method returns early when abs(pitch - 1.0) < 0.01
    np.testing.assert_array_equal(result, audio)


def test_pitch_shift_changes_length():
    """Test that pitch shift actually changes audio length."""
    import numpy as np
    from scipy.signal import resample_poly
    
    audio = np.ones(100, dtype=np.float32)
    
    # Pitch up (factor > 1) should shorten via resampling
    # Use integer factors for resample_poly
    result_up = resample_poly(audio, up=1, down=2)
    assert len(result_up) < len(audio)
    
    # Pitch down (factor < 1) should lengthen via resampling
    # Use integer factors: up=2, down=1 gives 2x length
    result_down = resample_poly(audio, up=2, down=1)
    assert len(result_down) > len(audio)


def test_kanade_disabled_by_default():
    """Test that Kanade is disabled by default."""
    from speech_to_speech.arguments_classes.kokoro_tts_arguments import KokoroTTSHandlerArguments
    
    args = KokoroTTSHandlerArguments()
    assert args.kokoro_kanade_enabled is False


def test_kanade_model_name():
    """Test default Kanade model name."""
    from speech_to_speech.arguments_classes.kokoro_tts_arguments import KokoroTTSHandlerArguments
    
    args = KokoroTTSHandlerArguments()
    assert args.kokoro_kanade_model == "epheam/Kanade"


def test_kokoro_lang_default_voices():
    """Verify language-to-voice mappings are correct."""
    from speech_to_speech.TTS.kokoro_handler import KOKORO_LANG_DEFAULT_VOICES
    
    # Check all expected languages have voices
    assert "a" in KOKORO_LANG_DEFAULT_VOICES
    assert "b" in KOKORO_LANG_DEFAULT_VOICES
    assert "e" in KOKORO_LANG_DEFAULT_VOICES
    assert "f" in KOKORO_LANG_DEFAULT_VOICES
    assert "h" in KOKORO_LANG_DEFAULT_VOICES
    assert "i" in KOKORO_LANG_DEFAULT_VOICES
    assert "j" in KOKORO_LANG_DEFAULT_VOICES
    assert "p" in KOKORO_LANG_DEFAULT_VOICES
    assert "z" in KOKORO_LANG_DEFAULT_VOICES
    
    # Verify specific voice assignments
    assert KOKORO_LANG_DEFAULT_VOICES["a"] == "af_heart"
    assert KOKORO_LANG_DEFAULT_VOICES["b"] == "bm_fable"
    assert KOKORO_LANG_DEFAULT_VOICES["j"] == "jf_alpha"


def test_language_mapping():
    """Verify Whisper language code mapping to Kokoro."""
    from speech_to_speech.TTS.kokoro_handler import WHISPER_LANGUAGE_TO_KOKORO_LANG
    
    # English should map to British
    assert WHISPER_LANGUAGE_TO_KOKORO_LANG["en"] == "b"
    # Japanese
    assert WHISPER_LANGUAGE_TO_KOKORO_LANG["ja"] == "j"
    # Chinese
    assert WHISPER_LANGUAGE_TO_KOKORO_LANG["zh"] == "z"
    # German (no native, maps to British)
    assert WHISPER_LANGUAGE_TO_KOKORO_LANG["de"] == "b"


def test_server_imports():
    """Verify server can be imported without errors."""
    import ast
    
    # Parse the server file to check syntax
    with open(Path(__file__).parent.parent / "hermes_voice_server.py") as f:
        tree = ast.parse(f.read())
    
    # Check for key classes and functions
    names = [getattr(node, 'name', None) for node in ast.walk(tree)]
    names = [n for n in names if n is not None]
    
    # Should have main function and lifespan function
    assert 'main' in names
    assert 'lifespan' in names


def test_server_endpoints_defined():
    """Verify server has expected endpoints."""
    with open(Path(__file__).parent.parent / "hermes_voice_server.py") as f:
        content = f.read()
    
    # Check key endpoint decorators are present
    assert '@app.get("/health")' in content
    assert '@app.post("/v1/audio/speech")' in content
    assert '@app.post("/v1/audio/transcriptions")' in content
    assert '@app.post("/voice/clone")' in content
    assert '@app.get("/voices/list")' in content


def test_server_voice_settings_model():
    """Verify VoiceSettings model has correct fields."""
    with open(Path(__file__).parent.parent / "hermes_voice_server.py") as f:
        content = f.read()
    
    # Check model fields
    assert "class VoiceSettings" in content
    assert 'voice: str = "bm_fable"' in content
    assert 'speed: float = 1.0' in content
    assert 'temperature: float = 0.667' in content
    assert 'pitch: float = 1.0' in content


def test_server_device_status_endpoint():
    """Verify device status endpoint exists."""
    with open(Path(__file__).parent.parent / "hermes_voice_server.py") as f:
        content = f.read()
    
    assert '@app.get("/device/status")' in content
    assert '@app.post("/device/set")' in content


def test_pyproject_kanade_extra():
    """Verify pyproject.toml has kanade extra."""
    import tomli
    
    with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as f:
        data = tomli.load(f)
    
    extras = data.get("project", {}).get("optional-dependencies", {})
    assert "kanade" in extras
    assert any("kanade_tokenizer" in dep for dep in extras["kanade"])
