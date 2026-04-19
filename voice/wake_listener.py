"""
voice/wake_listener.py
──────────────────────
Long-running subprocess: continuously detects wake words and records commands.
Loads WhisperModel ONCE directly (no sub-subprocess transcription_server).

Architecture:
  Primary:  OpenWakeWord (fast, ~1ms/frame, no hallucinations)
  Fallback: Whisper STT with energy gate + hallucination filter

stdout:  {"transcript": "...", "lang": "en"}
stderr:  progress / debug
"""
import argparse, json, os, sys, tempfile
import numpy as np

SAMPLE_RATE      = 16_000
CHUNK_SAMPLES    = 1_280
WAKE_CHUNK_SECS  = 2.5
SILENCE_TIMEOUT  = float(os.getenv("AIDA_SILENCE_TIMEOUT",  "1.0"))
ENERGY_THRESHOLD = float(os.getenv("AIDA_ENERGY_THRESHOLD", "0.015"))
MAX_CMD_SECS     = 30.0
WHISPER_MODEL    = os.getenv("WHISPER_MODEL", "small")
OWW_THRESHOLD    = float(os.getenv("AIDA_OWW_THRESHOLD",    "0.5"))

WAKE_KEYWORDS = [
    "aida","аида","aide","айда",
    "jarvis","джарвис",
    "assistant","ассистент",
    "helper","помощник",
]
HALLUCINATIONS = frozenset({
    "","."," ","...","you","i","a","the",
    "thank you","thank you.","thanks","thanks.",
    "bye","bye.","goodbye","goodbye.","okay","ok",
    "so","uh","um","hmm","oh","ah",
    "спасибо","спасибо.","пока","пока.",
    "да","нет","ладно","хорошо",
})


def _gpu():
    try:
        import torch; return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError: return "cpu"

def _rms(a): return float(np.sqrt(np.mean(a.astype(np.float32)**2)))

def _is_valid_wake(text):
    t = text.strip().lower().rstrip(" .")
    return t not in HALLUCINATIONS and len(t)>2 and any(kw in t for kw in WAKE_KEYWORDS)

def _record_chunks(n, device=None):
    import sounddevice as sd
    f=[]
    with sd.InputStream(samplerate=SAMPLE_RATE,channels=1,dtype="float32",device=device) as s:
        for _ in range(n): d,_=s.read(CHUNK_SAMPLES); f.append(d.flatten())
    return np.concatenate(f)

def _record_vad(device=None):
    import sounddevice as sd
    chunk=int(SAMPLE_RATE*0.05); max_c=int(MAX_CMD_SECS/0.05)
    sil_need=int(SILENCE_TIMEOUT/0.05)
    f=[]; silent=0; speech=False
    with sd.InputStream(samplerate=SAMPLE_RATE,channels=1,dtype="float32",device=device) as s:
        for _ in range(max_c):
            d,_=s.read(chunk); flat=d.flatten(); f.append(flat)
            if _rms(flat)>=ENERGY_THRESHOLD: speech=True; silent=0
            elif speech:
                silent+=1
                if silent>=sil_need: break
    return np.concatenate(f) if f else np.zeros(1,dtype=np.float32)

def _save_wav(audio):
    import soundfile as sf
    fd,p=tempfile.mkstemp(suffix=".wav"); os.close(fd)
    sf.write(p,audio,SAMPLE_RATE); return p

# ── Load Whisper ONCE at startup ──────────────────────────────────────────────
def _load_whisper():
    from faster_whisper import WhisperModel
    dev=_gpu()
    print(f"[wake_listener] Loading Whisper {WHISPER_MODEL} on {dev}...",
          file=sys.stderr,flush=True)
    for ct in (["float16","int8"] if dev=="cuda" else ["int8","float32"]):
        try:
            m=WhisperModel(WHISPER_MODEL,device=dev,compute_type=ct)
            print(f"[wake_listener] Whisper ready ({dev}/{ct})",file=sys.stderr,flush=True)
            return m
        except Exception as e: print(f"[wake_listener] {ct} fail: {e}",file=sys.stderr)
    raise RuntimeError("Cannot load WhisperModel")

def _transcribe(model,path,language=None):
    segs,info=model.transcribe(path,beam_size=5,language=language)
    return " ".join(s.text for s in segs).strip(), info.language

def _try_oww():
    try:
        from openwakeword.model import Model
        oww=Model(wakeword_models=["hey_jarvis"],enable_speex_noise_suppression=False)
        print("[wake_listener] OpenWakeWord ready",file=sys.stderr,flush=True)
        return oww
    except ImportError:
        print("[wake_listener] openwakeword not installed → Whisper fallback",file=sys.stderr,flush=True)
    except Exception as e:
        print(f"[wake_listener] OWW fail: {e}",file=sys.stderr,flush=True)
    return None

def _emit(text,lang):
    print(json.dumps({"transcript":text,"lang":lang}),flush=True)

def _handle_command(whisper_model,device,lang):
    audio=_record_vad(device=device)
    if _rms(audio)<ENERGY_THRESHOLD*0.5:
        print("[wake_listener] command: silence",file=sys.stderr,flush=True); return
    p=_save_wav(audio)
    try:
        text,detected_lang=_transcribe(whisper_model,p,language=lang)
        print(f"[wake_listener] cmd: {text!r} lang={detected_lang}",file=sys.stderr,flush=True)
        if text.strip(): _emit(text,detected_lang)
    finally:
        try: os.unlink(p)
        except: pass

def _oww_loop(oww,whisper_model,device,lang):
    import sounddevice as sd
    print("[wake_listener] OWW streaming...",file=sys.stderr,flush=True)
    with sd.InputStream(samplerate=SAMPLE_RATE,channels=1,dtype="float32",device=device) as stream:
        while True:
            frame,_=stream.read(CHUNK_SAMPLES)
            pcm=(frame.flatten()*32_768).astype(np.int16)
            pred=oww.predict(pcm)
            score=max(pred.values()) if pred else 0.0
            if score>=OWW_THRESHOLD:
                word=max(pred,key=pred.get)
                print(f"[wake_listener] OWW {word} score={score:.2f}",file=sys.stderr,flush=True)
                oww.reset()
                _handle_command(whisper_model,device,lang)

def _whisper_loop(whisper_model,device,lang):
    n=int(WAKE_CHUNK_SECS*SAMPLE_RATE/CHUNK_SAMPLES)
    print("[wake_listener] Whisper-fallback streaming...",file=sys.stderr,flush=True)
    while True:
        audio=_record_chunks(n,device=device)
        if _rms(audio)<ENERGY_THRESHOLD: continue
        p=_save_wav(audio)
        try: text,_=_transcribe(whisper_model,p,language="en")
        finally:
            try: os.unlink(p)
            except: pass
        print(f"[wake_listener] heard: {text!r}",file=sys.stderr,flush=True)
        if _is_valid_wake(text):
            print(f"[wake_listener] Wake: {text!r}",file=sys.stderr,flush=True)
            _handle_command(whisper_model,device,lang)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",type=int,default=None)
    p.add_argument("--lang",  type=str,default=None)
    args=p.parse_args()
    lang=None if args.lang in (None,"auto","") else args.lang
    try:
        whisper=_load_whisper()
    except Exception as e:
        print(f"[wake_listener] FATAL: {e}",file=sys.stderr,flush=True); sys.exit(1)
    oww=_try_oww()
    try:
        if oww: _oww_loop(oww,whisper,args.device,lang)
        else:   _whisper_loop(whisper,args.device,lang)
    except KeyboardInterrupt:
        print("[wake_listener] Stopped.",file=sys.stderr)
    except Exception as e:
        print(f"[wake_listener] Fatal: {e}",file=sys.stderr,flush=True); sys.exit(1)

if __name__=="__main__": main()
