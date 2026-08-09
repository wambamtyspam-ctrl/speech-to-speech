"""
Hermes Voice Services Server

FastAPI server providing OpenAI-compatible endpoints for:
- TTS (Kokoro with Kanade voice cloning)
- STT (faster-whisper)
- Voice management

Usage:
    python hermes_voice_server.py --port 7860
    python hermes_voice_server.py --device cuda:0
    python hermes_voice_server.py --tts kokoro --stt faster-whisper
"""

import argparse
import asyncio
import io
import json
import mimetypes
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler
    from speech_to_speech.STT.faster_whisper_handler import FasterWhisperSTTHandler
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class VoiceSettings(BaseModel):
    """Voice settings for TTS."""
    voice: str = "bm_fable"
    speed: float = 1.0
    temperature: float = 0.667
    pitch: float = 1.0
    language: str = "b"


class VoiceCloneRequest(BaseModel):
    """Voice cloning request."""
    text: str
    voice: str = ""
    speed: float = 1.0
    reference_file: Optional[str] = None  # Path to reference audio


class DeviceStatus(BaseModel):
    """Device status response."""
    device: str
    tts_loaded: bool
    stt_loaded: bool
    kanade_loaded: bool


# Global state
tts_handler: Optional[KokoroTTSHandler] = None
stt_handler: Optional[FasterWhisperSTTHandler] = None
settings: VoiceSettings = VoiceSettings()
device: str = "auto"
kanade_enabled: bool = False
reference_audio: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global tts_handler, stt_handler
    
    # Load TTS
    if not tts_handler:
        tts_handler = KokoroTTSHandler(
            stop_event=None,  # Will be managed by server
            queue_in=None,
            queue_out=None,
            setup_kwargs={
                "should_listen": None,
                "device": device,
                "voice": settings.voice,
                "lang_code": settings.language,
                "speed": settings.speed,
                "temperature": settings.temperature,
                "pitch": settings.pitch,
                "kanade_enabled": kanade_enabled,
                "reference_audio": reference_audio,
            }
        )
        print(f"TTS loaded: {device}")
    
    # Load STT (lazy)
    print("Voice services ready")
    yield
    
    # Cleanup
    if tts_handler:
        try:
            tts_handler.cleanup()
        except Exception as e:
            print(f"Cleanup error: {e}")


app = FastAPI(
    title="Hermes Voice Services",
    description="OpenAI-compatible TTS and STT endpoints with voice cloning",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/status")
async def status():
    """Service status endpoint."""
    return {
        "tts": {
            "loaded": tts_handler is not None,
            "device": device,
            "voice": settings.voice,
            "speed": settings.speed,
            "temperature": settings.temperature,
            "pitch": settings.pitch,
            "kanade_enabled": kanade_enabled,
        },
        "stt": {
            "loaded": stt_handler is not None,
            "device": device,
        },
    }


@app.get("/device/status")
async def device_status():
    """Get device information."""
    import torch
    
    devices = {"cpu": True}
    if torch.cuda.is_available():
        devices["cuda"] = True
        for i in range(torch.cuda.device_count()):
            dev_id = f"cuda:{i}"
            devices[dev_id] = {
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
            }
    
    return {
        "available": devices,
        "current": device,
    }


@app.post("/device/set")
async def set_device(request: dict):
    """Change device for services."""
    global device, tts_handler
    
    new_device = request.get("device", device)
    services = request.get("services", ["tts"])
    
    # Validate device
    try:
        torch.device(new_device)
    except (RuntimeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid device: {new_device}")
    
    device = new_device
    
    result = {"changed": {}, "errors": {}}
    
    if "tts" in services:
        try:
            old_device = device
            # Reload TTS on new device
            tts_handler = None  # Will be reloaded on next request
            result["changed"]["tts"] = {"from": old_device, "to": new_device}
        except Exception as e:
            result["errors"]["tts"] = str(e)
    
    return result


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing."""
    return {
        "object": "list",
        "data": [
            {
                "id": "kokoro-tts-v1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "hermes",
            },
            {
                "id": "whisper-large-v3-turbo",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "hermes",
            },
        ],
    }


@app.post("/v1/audio/speech")
async def synthesize_speech(
    input: str = Form(..., description="Text to synthesize"),
    model: str = Form("kokoro-tts-v1", description="Model name"),
    voice: str = Form("", description="Voice name (empty = use default)"),
    speed: float = Form(0.0, ge=0.1, le=10.0, description="Speech speed (0 = use default)"),
    temperature: float = Form(0.0, ge=0.1, le=2.0, description="Temperature (0 = use default)"),
    pitch: float = Form(0.0, ge=0.5, le=2.0, description="Pitch (0 = use default)"),
    response_format: str = Form("mp3", description="Output format: mp3, wav, ogg, flac"),
    stream: bool = Form(False, description="Stream audio chunks"),
    clone_voice: bool = Form(False, description="Enable voice cloning"),
):
    """OpenAI-compatible TTS endpoint."""
    global settings
    
    if not input.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    
    # Resolve settings
    if voice:
        settings.voice = voice
    if speed > 0:
        settings.speed = speed
    if temperature > 0:
        settings.temperature = temperature
    if pitch > 0:
        settings.pitch = pitch
    
    # Ensure TTS is loaded
    if not tts_handler:
        tts_handler = KokoroTTSHandler(
            stop_event=None,
            queue_in=None,
            queue_out=None,
            setup_kwargs={
                "should_listen": None,
                "device": device,
                "voice": settings.voice,
                "lang_code": settings.language,
                "speed": settings.speed,
                "temperature": settings.temperature,
                "pitch": settings.pitch,
                "kanade_enabled": kanade_enabled and clone_voice,
                "reference_audio": reference_audio,
            }
        )
    
    try:
        # Generate audio
        audio_chunks = []
        for chunk in tts_handler._process_kokoro(input):
            audio_chunks.append(chunk)
        
        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        # Concatenate chunks
        full_audio = np.concatenate(audio_chunks)
        sample_rate = 16000  # Pipeline sample rate
        
        # Encode to requested format
        buf = io.BytesIO()
        sf.write(buf, full_audio, sample_rate, format=response_format.upper())
        audio_bytes = buf.getvalue()
        
        content_type = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
        }.get(response_format, "audio/mpeg")
        
        return Response(content=audio_bytes, media_type=content_type)
        
    except Exception as e:
        print(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form("whisper-large-v3-turbo", description="Model name"),
    language: Optional[str] = Form(None, description="Language code"),
    response_format: str = Form("json", description="Response format: json, text, srt, verbose_json"),
    temperature: float = Form(0.0, description="Temperature (ignored by faster-whisper)"),
):
    """OpenAI-compatible STT endpoint."""
    global stt_handler
    
    # Load STT if needed
    if not stt_handler:
        stt_handler = FasterWhisperSTTHandler(
            stop_event=None,
            queue_in=None,
            queue_out=None,
            setup_kwargs={
                "should_listen": None,
                "device": device,
            }
        )
    
    try:
        # Read audio
        audio_bytes = await file.read()
        buf = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(buf, dtype="float32")
        
        # Transcribe
        result = stt_handler.transcribe(
            audio=audio_data,
            sample_rate=sample_rate,
            language=language,
        )
        
        if response_format == "text":
            return Response(content=result["text"], media_type="text/plain")
        elif response_format == "srt":
            srt_lines = []
            for i, seg in enumerate(result.get("segments", []), 1):
                start = format_srt_time(seg["start"])
                end = format_srt_time(seg["end"])
                srt_lines.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")
            return Response(content="\n".join(srt_lines), media_type="text/plain")
        else:
            return JSONResponse(content=result)
            
    except Exception as e:
        print(f"STT error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/translations")
async def translate_audio(
    file: UploadFile = File(..., description="Audio file to translate"),
    model: str = Form("whisper-large-v3-turbo", description="Model name"),
    response_format: str = Form("json", description="Response format"),
):
    """Translate audio to English."""
    global stt_handler
    
    if not stt_handler:
        stt_handler = FasterWhisperSTTHandler(
            stop_event=None,
            queue_in=None,
            queue_out=None,
            setup_kwargs={
                "should_listen": None,
                "device": device,
            }
        )
    
    try:
        audio_bytes = await file.read()
        buf = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(buf, dtype="float32")
        
        result = stt_handler.transcribe(
            audio=audio_data,
            sample_rate=sample_rate,
            task="translate",
        )
        
        if response_format == "text":
            return Response(content=result["text"], media_type="text/plain")
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voices/list")
async def list_voices():
    """List available Kokoro voices."""
    if not tts_handler:
        raise HTTPException(status_code=503, detail="TTS not loaded")
    
    # Return known voices from the handler
    voices = []
    for lang_code, voice in KOKORO_LANG_DEFAULT_VOICES.items():
        voices.append({
            "id": voice,
            "name": f"{lang_code} - {voice}",
            "language": lang_code,
            "gender": "female" if voice[1] == "f" else "male",
        })
    
    # Add more voices from the handler's known list
    all_voices = [
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        "ef_dora", "em_alex", "em_santa",
        "ff_siwis",
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        "if_sara", "im_nicola",
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
        "pf_dora", "pm_alex", "pm_santa",
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ]
    
    for voice in all_voices:
        if not any(v["id"] == voice for v in voices):
            lang = "a" if voice[0] == "a" else "b" if voice[0] == "b" else "a"
            voices.append({
                "id": voice,
                "name": voice,
                "language": lang,
                "gender": "female" if voice[1] == "f" else "male",
            })
    
    return {"voices": sorted(voices, key=lambda x: x["id"])}


@app.post("/voice/clone")
async def clone_voice(
    text: str = Form(..., description="Text to synthesize"),
    voice: str = Form("", description="Base Kokoro voice"),
    speed: float = Form(1.0, ge=0.1, le=10.0),
    reference_file: Optional[UploadFile] = File(None, description="Reference audio file"),
    response_format: str = Form("mp3"),
):
    """Voice cloning endpoint - generate speech and convert to reference voice."""
    global settings, kanade_enabled, reference_audio
    
    if not tts_handler:
        raise HTTPException(status_code=503, detail="TTS not loaded")
    
    # Update reference if provided
    if reference_file:
        ref_bytes = await reference_file.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_bytes)
            reference_audio = tmp.name
        kanade_enabled = True
    
    # Update voice settings
    if voice:
        settings.voice = voice
    settings.speed = speed
    
    try:
        # Generate base audio
        audio_chunks = []
        for chunk in tts_handler._process_kokoro(text):
            audio_chunks.append(chunk)
        
        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        full_audio = np.concatenate(audio_chunks)
        sample_rate = 16000
        
        # Apply voice cloning if enabled
        if kanade_enabled and tts_handler.kanade_load_success and tts_handler._resolved_reference:
            # Resample to Kanade's sample rate
            from scipy.signal import resample_poly
            full_audio = resample_poly(full_audio, up=tts_handler._kanade_sample_rate, down=sample_rate)
            
            # Convert with Kanade
            full_audio = tts_handler._convert_with_kanade(full_audio, source_sr=tts_handler._kanade_sample_rate)
            sample_rate = tts_handler._kanade_sample_rate
        
        # Encode to requested format
        buf = io.BytesIO()
        sf.write(buf, full_audio, sample_rate, format=response_format.upper())
        audio_bytes = buf.getvalue()
        
        content_type = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
        }.get(response_format, "audio/mpeg")
        
        return Response(content=audio_bytes, media_type=content_type)
        
    except Exception as e:
        print(f"Voice clone error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hermes Voice Services Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--device", default="auto", help="Device to use (cuda:0, cpu, auto)")
    parser.add_argument("--voice", default="bm_fable", help="Default voice")
    parser.add_argument("--speed", type=float, default=1.0, help="Default speed")
    parser.add_argument("--temperature", type=float, default=0.667, help="Default temperature")
    parser.add_argument("--pitch", type=float, default=1.0, help="Default pitch")
    parser.add_argument("--language", default="b", help="Default language code")
    parser.add_argument("--kanade", action="store_true", help="Enable Kanade voice cloning")
    parser.add_argument("--reference", default=None, help="Path to reference audio")
    parser.add_argument("--no-stt", action="store_true", help="Disable STT")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    global device, settings, kanade_enabled, reference_audio
    
    device = args.device
    settings = VoiceSettings(
        voice=args.voice,
        speed=args.speed,
        temperature=args.temperature,
        pitch=args.pitch,
        language=args.language,
    )
    kanade_enabled = args.kanade
    reference_audio = args.reference
    
    print(f"Starting Hermes Voice Services on {args.host}:{args.port}")
    print(f"Device: {device}")
    print(f"TTS: Kokoro (voice={args.voice}, speed={args.speed}, temp={args.temperature}, pitch={args.pitch})")
    if args.kanade:
        print(f"Voice cloning: ENABLED (reference={args.reference})")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
