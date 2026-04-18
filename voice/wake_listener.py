"""
voice/wake_listener.py
──────────────────────
Long-running subprocess: continuously detects wake words and records commands.

Architecture:
  Primary:  OpenWakeWord (fast, ~1ms/frame, no hallucinations)
  Fallback: Whisper STT with energy gate + hallucination filter

stdout:  {"transcript": "...", "lang": "en"}
stderr:  progress / debug
"""

import argparse, json, os, sys, tempfile
import numpy as np

SAMPLE_RATE      = 16_000
CHUNK_SAMPLES    = 1_280           # 80 ms — OWW standard frame
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

# ── Transcription via sub-subprocess ─────────────────────────────────────────
# wake_listener spawns its own transcription_server so the Whisper model
# stays loaded for the entire wake word session (not reloaded each command).

_TRANS_PROC = None
_TRANS_LOCK = __import__("threading").Lock()


def _get_trans_server():
    global _TRANS_PROC
    if _TRANS_PROC and _TRANS_PROC.poll() is None:
        return _TRANS_PROC
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "transcription_server.py")
    if not os.path.exists(script):
        raise RuntimeError("transcription_server.py not found next to wake_listener.py")
    print("[wake_listener] Starting transcription_server sub-process ...",
          file=sys.stderr, flush=True)
    _TRANS_PROC = __import__("subprocess").Popen(
        [__import__("sys").executable, script],
        stdin=__import__("subprocess").PIPE,
        stdout=__import__("subprocess").PIPE,
        stderr=__import__("subprocess").PIPE,
        bufsize=0,
    )
    import threading, queue as _queue
    rq = _queue.Queue()
    def _r(): rq.put(_TRANS_PROC.stdout.readline())
    threading.Thread(target=_r, daemon=True).start()
    # drain stderr
    def _e():
        for l in _TRANS_PROC.stderr:
            print(f"[trans-srv] {l.decode(errors='replace').rstrip()}", file=sys.stderr)
    threading.Thread(target=_e, daemon=True).start()
    line = rq.get(timeout=120)
    if b"READY" not in line:
        raise RuntimeError(f"transcription_server failed: {line!r}")
    print("[wake_listener] Transcription server READY.", file=sys.stderr, flush=True)
    return _TRANS_PROC


def _transcribe_file(path, language=None):
    import json as _json
    with _TRANS_LOCK:
        proc = _get_trans_server()
        req  = _json.dumps({"path": path, "lang": language or "auto"}) + "\n"
        proc.stdin.write(req.encode()); proc.stdin.flush()
        resp = _json.loads(proc.stdout.readline().decode().strip())
        return resp.get("text",""), resp.get("lang","en")


# Keep _load_whisper / _transcribe as aliases for OWW path that needs them
def _load_whisper():
    # Model is loaded inside transcription_server sub-process now.
    # This function just ensures the server is alive.
    _get_trans_server()
    return None   # no local model object needed

def _transcribe(model, path, language=None):
    return _transcribe_file(path, language)

def _try_oww():
    try:
        from openwakeword.model import Model
        oww=Model(wakeword_models=["hey_jarvis"],enable_speex_noise_suppression=False)
        print("[wake_listener] OpenWakeWord ready (hey_jarvis)",file=sys.stderr,flush=True)
        return oww
    except ImportError:
        print("[wake_listener] openwakeword not installed → Whisper fallback",file=sys.stderr,flush=True)
    except Exception as e:
        print(f"[wake_listener] OWW fail: {e} → Whisper fallback",file=sys.stderr,flush=True)
    return None

def _emit(text,lang):
    print(json.dumps({"transcript":text,"lang":lang}),flush=True)

def _handle_command(device,lang):
    """Record command after wake word and transcribe it."""
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

def _oww_loop(oww,_model_unused,device,lang):
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
                print(f"[wake_listener] 🔔 OWW {word} score={score:.2f}",file=sys.stderr,flush=True)
                oww.reset()
                _handle_command(device,lang)

def _whisper_loop(_model_unused,device,lang):
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
            print(f"[wake_listener] 🔔 Wake: {text!r}",file=sys.stderr,flush=True)
            _handle_command(device,lang)

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
        if oww: _oww_loop(oww,None,args.device,lang)
        else:   _whisper_loop(None,args.device,lang)
    except KeyboardInterrupt:
        print("[wake_listener] Stopped.",file=sys.stderr)
    except Exception as e:
        print(f"[wake_listener] Fatal: {e}",file=sys.stderr,flush=True); sys.exit(1)

if __name__=="__main__": main()
