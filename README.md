# Bmtts-f-l--a
a ka fisa. 
Bmtss fɔlɔ -a

pip install fastapi uvicorn google-genai pydub python-multipart
# ffmpeg مطلوب للدمج النظيف
python main.py

# main.py — Bamanankan TTS (ملف واحد)
# pip install fastapi uvicorn google-genai pydub python-multipart
# ثم: python main.py

import os, re, io, uuid, base64, json, asyncio
from pathlib import Path
from typing import Optional, List, Tuple

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
KEY_FILE = ROOT / ".gemini_key"
OUT.mkdir(exist_ok=True)

# ——— أرقام بامبارا ———
U = {0:"wolofila",1:"kelen",2:"fila",3:"saba",4:"naani",5:"duuru",6:"wɔɔrɔ",7:"wolonwula",8:"seyin",9:"kɔnɔntɔn"}
T10 = {10:"tan",11:"tan ni kelen",12:"tan ni fila",13:"tan ni saba",14:"tan ni naani",
       15:"tan ni duuru",16:"tan ni wɔɔrɔ",17:"tan ni wolonwula",18:"tan ni seyin",19:"tan ni kɔnɔntɔn"}
T20 = {20:"mugan",30:"bi saba",40:"bi naani",50:"bi duuru",60:"bi wɔɔrɔ",70:"bi wolonwula",80:"bi seyin",90:"bi kɔnɔntɔn"}

def n2bm(n: int) -> str:
    n = int(n)
    if n < 0: return "tɛmɛnen " + n2bm(-n)
    if n < 10: return U[n]
    if n < 20: return T10[n]
    if n < 100:
        t, r = (n//10)*10, n%10
        b = T20.get(t, f"bi {U[t//10]}")
        return b if r == 0 else f"{b} ni {U[r]}"
    if n < 1000:
        h, r = n//100, n%100
        b = "kɛmɛ" if h == 1 else f"kɛmɛ {U[h]}"
        return b if r == 0 else f"{b} ni {n2bm(r)}"
    if n < 10**6:
        th, r = n//1000, n%1000
        b = "waga kelen" if th == 1 else f"waga {n2bm(th)}"
        return b if r == 0 else f"{b} ni {n2bm(r)}"
    m, r = n//10**6, n%10**6
    b = f"miliyɔn {n2bm(m)}"
    return b if r == 0 else f"{b} ni {n2bm(r)}"

def nums_to_bm(t: str) -> str:
    return re.sub(r"\b\d+\b", lambda m: n2bm(int(m.group())), t)

VOICES = ["Kore","Orus","Alnilam","Schedar","Gacrux","Sulafat","Charon","Iapetus",
          "Zephyr","Puck","Aoede","Leda","Fenrir","Achird","Algieba","Despina"]
STYLE = ("Calm, dignified steady Bambara narration. Warm authoritative voice. "
         "Clear precise pronunciation, even pacing. Clean studio, no noise no crackle.")
MODELS = ["gemini-2.5-flash-preview-tts","gemini-2.5-pro-preview-tts","gemini-2.5-flash-tts"]
CHUNK = 1000
jobs = {}

def load_key() -> str:
    k = os.getenv("GEMINI_API_KEY", "").strip()
    if k: return k
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    return ""

def save_key(k: str):
    KEY_FILE.write_text(k.strip(), encoding="utf-8")
    os.environ["GEMINI_API_KEY"] = k.strip()

def client():
    k = load_key()
    if not k: raise RuntimeError("لا يوجد مفتاح — أدخله من الواجهة")
    if not genai: raise RuntimeError("ثبّت: pip install google-genai")
    return genai.Client(api_key=k)

def split_text(t: str) -> List[str]:
    parts = re.split(r"(?<=[.!?؟。\n;:])\s+", t.strip())
    out, cur = [], ""
    for s in parts:
        s = s.strip()
        if not s: continue
        if len(cur)+len(s)+1 <= CHUNK:
            cur = (cur+" "+s).strip()
        else:
            if cur: out.append(cur)
            if len(s) > CHUNK:
                for i in range(0, len(s), CHUNK):
                    out.append(s[i:i+CHUNK])
                cur = ""
            else:
                cur = s
    if cur: out.append(cur)
    return out or [t[:CHUNK]]

def extract_audio(resp) -> bytes:
    for p in resp.candidates[0].content.parts:
        d = getattr(getattr(p, "inline_data", None), "data", None)
        if d:
            return base64.b64decode(d) if isinstance(d, str) else d
    raise RuntimeError("لا بيانات صوت")

def to_seg(raw: bytes):
    if not AudioSegment:
        raise RuntimeError("ثبّت: pip install pydub + ffmpeg")
    bio = io.BytesIO(raw)
    try:
        seg = AudioSegment.from_file(bio, format="wav")
    except Exception:
        bio.seek(0)
        try:
            seg = AudioSegment.from_file(bio)
        except Exception:
            seg = AudioSegment.from_raw(io.BytesIO(raw), sample_width=2, frame_rate=24000, channels=1)
    try:
        seg = seg.strip_silence(silence_len=80, silence_thresh=-42, padding=40)
    except Exception:
        pass
    return seg.set_frame_rate(24000).set_channels(1).set_sample_width(2)

def synth_one(text: str, voice: str, style: str) -> bytes:
    c = client()
    prompt = f"{style or STYLE}\n\nRead this Bambara clearly and steadily. No extra words.\n\n{text}"
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice if voice in VOICES else "Kore")
            )
        ),
    )
    err = None
    for m in MODELS:
        try:
            r = c.models.generate_content(model=m, contents=prompt, config=cfg)
            return extract_audio(r)
        except Exception as e:
            err = e
    raise RuntimeError(str(err))

def synthesize(text: str, voice="Kore", style=STYLE, gap_ms=220, read_numbers=True, progress=None) -> Tuple[bytes, list, int]:
    logs = []
    text = text.strip()
    if not text: raise ValueError("نص فارغ")
    if read_numbers:
        text = nums_to_bm(text)
        logs.append("✔ أرقام → بامبارا")
    chunks = split_text(text)
    logs.append(f"مقاطع: {len(chunks)}")
    segs = []
    for i, ch in enumerate(chunks):
        logs.append(f"▶ {i+1}/{len(chunks)}")
        if progress: progress(int(i/max(len(chunks),1)*90))
        segs.append(to_seg(synth_one(ch, voice, style)))
    gap = AudioSegment.silent(duration=max(0, gap_ms), frame_rate=24000)
    comb = segs[0]
    for s in segs[1:]:
        comb = comb + gap + s
    buf = io.BytesIO()
    comb.export(buf, format="wav", parameters=["-ac","1","-ar","24000"])
    if progress: progress(100)
    logs.append(f"✅ {len(comb)/1000:.1f}ث")
    return buf.getvalue(), logs, len(segs)

# ——— API ———
app = FastAPI()

class TTSIn(BaseModel):
    text: str
    voice: str = "Kore"
    style: str = STYLE
    read_numbers: bool = True
    gap_ms: int = 220
    voice_ref: Optional[str] = None

class KeyIn(BaseModel):
    key: str

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.get("/api/health")
def health():
    k = bool(load_key())
    return {"api_key_present": k, "engine_ready": k and genai is not None}

@app.post("/api/key")
def set_key(body: KeyIn):
    if len(body.key.strip()) < 10:
        return {"ok": False, "error": "مفتاح غير صالح"}
    save_key(body.key)
    return {"ok": True}

@app.get("/api/voices")
def voices():
    rec = ["Kore","Orus","Alnilam","Schedar","Gacrux","Sulafat"]
    return {"voices": [{"name":v,"style":"","recommended":v in rec} for v in VOICES], "recommended": rec}

@app.get("/api/number")
def number(n: int):
    return {"n": n, "bambara": n2bm(n)}

@app.post("/api/tts")
def tts(body: TTSIn):
    try:
        data, logs, parts = synthesize(body.text, body.voice, body.style or STYLE, body.gap_ms, body.read_numbers)
        fn = f"t_{uuid.uuid4().hex[:10]}.wav"
        (OUT/fn).write_bytes(data)
        return {"ok": True, "file": f"/out/{fn}", "bytes": len(data), "parts": parts, "voice": body.voice, "log": logs}
    except Exception as e:
        return {"ok": False, "error": str(e), "log": [str(e)]}

@app.post("/api/tts-long")
async def tts_long(body: TTSIn):
    jid = uuid.uuid4().hex[:10]
    jobs[jid] = {"status":"running","percent":0,"log":[],"file":None,"bytes":0,"error":None,"total":0}

    def run():
        try:
            def prog(p): jobs[jid]["percent"] = p
            data, logs, parts = synthesize(body.text, body.voice, body.style or STYLE, body.gap_ms, body.read_numbers, prog)
            fn = f"L_{jid}.wav"
            (OUT/fn).write_bytes(data)
            jobs[jid].update(status="done", percent=100, log=logs, file=f"/out/{fn}", bytes=len(data), total=parts)
        except Exception as e:
            jobs[jid].update(status="failed", error=str(e), log=[str(e)])

    asyncio.get_event_loop().run_in_executor(None, run)
    return {"ok": True, "job": jid}

@app.get("/api/job/{jid}")
def job(jid: str):
    j = jobs.get(jid)
    if not j: return {"ok": False}
    return {"ok": True, "job": j, "percent": j["percent"], "log": j.get("log", [])}

@app.post("/api/clone")
async def clone(source_audio: UploadFile = File(...), consent_audio: UploadFile = File(...),
                display_name: str = Form("Bambara"), store: str = Form("true")):
    # حفظ العينات محلياً — التوليد يبقى بالصوت الجاهز + أسلوب وقور
    sid = uuid.uuid4().hex[:8]
    (OUT/f"src_{sid}.bin").write_bytes(await source_audio.read())
    (OUT/f"con_{sid}.bin").write_bytes(await consent_audio.read())
    ref = f"voice_{sid}"
    return {"ok": True, "voice_ref": ref, "log": ["✔ حُفظت البصمة. استخدم صوتاً جاهزاً قوياً (Kore/Orus) حتى يتوفر Instant Voice في حسابك."]}

@app.get("/api/myvoices")
def myvoices():
    return {"ok": True, "voices": []}

@app.post("/api/design")
def design(body: dict):
    ref = f"voice_{uuid.uuid4().hex[:8]}"
    return {"ok": True, "voice_ref": ref, "preview": None, "log": ["✔ وُصف الصوت. طبّق الأسلوب في خانة الأسلوب."]}

@app.get("/out/{name}")
def out_file(name: str):
    p = OUT / name
    if not p.exists(): return HTMLResponse("404", 404)
    return FileResponse(p, media_type="audio/wav")

# ——— واجهة مصغّرة ———
HTML = r"""<!DOCTYPE html>
<html lang="ar" dir="rtl"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bamanankan TTS</title>
<style>
:root{--bg:#0b1020;--card:#161f3d;--line:#26325c;--t:#e8edff;--m:#8fa0cc;--a:#4f8cff;--ok:#22c55e;--err:#ef4444}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:radial-gradient(900px 400px at 50% -10%,#1b2a55,var(--bg));color:var(--t);min-height:100vh;padding:16px;line-height:1.6}
.w{max-width:720px;margin:auto}
h1{text-align:center;font-size:1.5rem;margin-bottom:4px}h1 span{color:var(--a)}
.sub{text-align:center;color:var(--m);font-size:.85rem;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin-bottom:12px}
.card h2{font-size:.95rem;margin-bottom:10px;border-right:3px solid var(--a);padding-right:8px}
textarea,input,select{width:100%;background:#121a33;border:1px solid var(--line);border-radius:10px;color:var(--t);padding:10px;font:inherit;margin-top:4px}
textarea{min-height:120px;resize:vertical}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media(max-width:560px){.row{grid-template-columns:1fr}}
label{font-size:.78rem;color:var(--m)}
.btns{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
button{background:var(--a);color:#fff;border:0;border-radius:10px;padding:10px 16px;font-weight:700;cursor:pointer;font:inherit}
button.g{background:transparent;border:1px solid var(--line);color:var(--t)}
button:disabled{opacity:.5}
.pill{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.72rem;border:1px solid var(--line);color:var(--m);margin:0 4px}
.pill.on{border-color:var(--ok);color:var(--ok)}.pill.off{border-color:var(--err);color:var(--err)}
#log{background:#080d1c;border:1px solid var(--line);border-radius:8px;padding:10px;font:12px monospace;color:#9fb3e0;max-height:160px;overflow:auto;direction:ltr;text-align:left;white-space:pre-wrap}
.bar{height:6px;background:#121a33;border-radius:4px;margin-top:8px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--a),var(--ok));transition:.3s}
audio{width:100%;margin-top:10px}
.stats{font-size:.8rem;color:var(--m);margin-top:6px}.stats b{color:var(--t)}
.chk{display:flex;align-items:center;gap:8px;margin-top:8px;font-size:.85rem}
.chk input{width:auto}
</style></head><body><div class="w">
<header>
<h1>🎙️ Bamanankan <span>TTS</span></h1>
<div class="sub">نص + أرقام بالبامبارا · صوت وقور نقي · نصوص طويلة</div>
<div style="text-align:center">
<span class="pill" id="kp">المفتاح…</span>
<span class="pill" id="ep">المحرّك…</span>
</div>
</header>

<div class="card">
<h2>مفتاح Gemini (مرة واحدة — يُحفظ محلياً)</h2>
<div class="row">
<input type="password" id="key" placeholder="الصق مفتاح Google AI Studio هنا">
<button id="btnKey">💾 حفظ المفتاح</button>
</div>
</div>

<div class="card">
<h2>١. النص</h2>
<textarea id="text">I ni ce! N tɔgɔ ye Awa ye.
N yɛrɛ bɛ san 2026. A ye 1250 fɔ.
Hɛrɛ sira? Hɛrɛ dɔrɔn.</textarea>
<div class="stats">حروف: <b id="cc">0</b> · كلمات: <b id="wc">0</b> · مقاطع: <b id="ch">0</b></div>
<div class="chk"><input type="checkbox" id="rn" checked><label for="rn">أرقام بامبارا (7→wolonwula)</label></div>
</div>

<div class="card">
<h2>٢. الصوت</h2>
<div class="row">
<div><label>صوت</label><select id="voice"></select></div>
<div><label>وقفة ms</label><input type="number" id="gap" value="220" min="0" max="1000" step="20"></div>
</div>
<label style="margin-top:8px;display:block">أسلوب</label>
<textarea id="style" style="min-height:70px"></textarea>
</div>

<div class="card">
<h2>٣. توليد</h2>
<div class="btns">
<button id="go">▶️ ولّد</button>
<button class="g" id="gol">🧵 نص طويل</button>
<button class="g" id="stop">■ إيقاف</button>
</div>
<div class="bar"><i id="bar"></i></div>
<audio id="player" controls style="display:none"></audio>
<div id="log" style="margin-top:8px"></div>
</div>
</div>
<script>
const $=id=>document.getElementById(id);
const log=s=>{$("log").textContent+=s+"\n";$("log").scrollTop=99999};
const bar=p=>$("bar").style.width=Math.min(100,p)+"%";
let JOB=null,POLL=null;

async function health(){
  try{
    const d=await(await fetch("/api/health")).json();
    $("kp").textContent=d.api_key_present?"✔ المفتاح محفوظ":"✖ أدخل المفتاح";
    $("kp").className="pill "+(d.api_key_present?"on":"off");
    $("ep").textContent=d.engine_ready?"✔ جاهز":"⏳";
    $("ep").className="pill "+(d.engine_ready?"on":"");
  }catch(e){log("⚠️ "+e)}
}
async function loadVoices(){
  const d=await(await fetch("/api/voices")).json();
  $("voice").innerHTML=d.voices.map(v=>`<option value="${v.name}">${v.recommended?"⭐ ":""}${v.name}</option>`).join("");
  $("voice").value="Kore";
}
function count(){
  const t=$("text").value||"";
  $("cc").textContent=t.length;
  $("wc").textContent=t.trim()?t.trim().split(/\s+/).length:0;
  $("ch").textContent=t.length<=1000?(t.trim()?1:0):Math.ceil(t.length/1000);
}
$("text").oninput=count;

$("btnKey").onclick=async()=>{
  const k=$("key").value.trim();
  if(!k)return log("❌ الصق المفتاح");
  const d=await(await fetch("/api/key",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({key:k})})).json();
  log(d.ok?"✔ المفتاح حُفظ محلياً":("❌ "+d.error));
  $("key").value="";
  health();
};

async function synth(long){
  const text=$("text").value.trim();
  if(!text)return log("❌ اكتب نصاً");
  const body={text,voice:$("voice").value,style:$("style").value,read_numbers:$("rn").checked,gap_ms:+$("gap").value||220};
  $("go").disabled=$("gol").disabled=true;bar(10);log(long?"🧵 مهمة طويلة…":"▶️ توليد…");
  try{
    if(!long){
      const d=await(await fetch("/api/tts",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})).json();
      (d.log||[]).forEach(log);
      if(!d.ok){log("❌ "+d.error);bar(0);return}
      bar(100);play(d.file);log("✅ تم");
    }else{
      const d=await(await fetch("/api/tts-long",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})).json();
      if(!d.ok){log("❌");return}
      JOB=d.job;log("🆔 "+JOB);
      if(POLL)clearInterval(POLL);
      POLL=setInterval(async()=>{
        const j=await(await fetch("/api/job/"+JOB)).json();
        if(!j.ok)return;
        bar(j.percent);
        (j.log||[]).slice(-2).forEach(s=>{if(!$("log").textContent.includes(s))log(s)});
        if(j.job.status==="done"){clearInterval(POLL);play(j.job.file);log("✅ طويل جاهز");bar(100)}
        if(j.job.status==="failed"){clearInterval(POLL);log("❌ "+j.job.error);bar(0)}
      },2000);
    }
  }catch(e){log("❌ "+e)}
  finally{$("go").disabled=$("gol").disabled=false;setTimeout(()=>bar(0),1500)}
}
function play(u){const p=$("player");p.src=u+"?t="+Date.now();p.style.display="block";p.play().catch(()=>{})}
$("go").onclick=()=>synth(false);
$("gol").onclick=()=>synth(true);
$("stop").onclick=()=>$("player").pause();

$("style").value="Calm, dignified and steady Bambara narration. Warm authoritative voice. Clear precise pronunciation, even pacing. Clean studio quality, no crackle.";
count();loadVoices();health();
log("ℹ️ أدخل المفتاح مرة واحدة ثم ولّد الصوت.");
</script></body></html>"""

if __name__ == "__main__":
    print("→ http://127.0.0.1:8000")
    print("  المفتاح: من الواجهة (يُحفظ في .gemini_key) أو GEMINI_API_KEY")
    uvicorn.run(app, host="0.0.0.0", port=8000)
