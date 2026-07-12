// Realtime duplex voice chat for MiniCPM-o 4.5.
// mic (AudioWorklet -> 16k mono pcm16) --> WS /v1/realtime?duplex=1 --> 24k playback (AudioWorklet).
//
// Interaction modes (client-side behaviors over the same WS):
//   full : continuous mic; model-driven turn-taking (or server_vad). Current default behavior.
//   half : push-to-talk; mic frames are only sent while "Hold to talk" is held.
//   turn : record one utterance (hold), then on release send commit{final}+response.create.
//
// Voice: chosen BEFORE the call in the initial session.update.
//   named voice -> session.voice (e.g. "default")
//   reference-audio clone -> session.extra_body.ref_audio = data:audio/wav;base64,...
//   (neither can change after audio output starts -> server rejects voice_update_after_audio).
//
// Barge-in: only an explicit cancelled/truncated response flushes playback.

(() => {
  'use strict';

  // ---- DOM ----
  const callBtn = document.getElementById('callBtn');
  const pttBtn = document.getElementById('pttBtn');
  const statusEl = document.getElementById('status');
  const stateEl = document.getElementById('modelState');
  const timerEl = document.getElementById('timer');
  const convEl = document.getElementById('conversation');
  const logEl = document.getElementById('log');
  const vuFill = document.getElementById('vuFill');
  const micDot = document.getElementById('micDot');
  const voiceSel = document.getElementById('voiceSel');
  const refField = document.getElementById('refField');
  const refFile = document.getElementById('refFile');
  const modeSel = document.getElementById('modeSel');
  const turnDetSel = document.getElementById('turnDetSel');
  const lockNote = document.getElementById('lockNote');

  // ---- Config ----
  function sameOriginWsBase() {
    const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
    let basePath = location.pathname || '';
    if (basePath.endsWith('/')) basePath = basePath.slice(0, -1);
    else basePath = basePath.slice(0, basePath.lastIndexOf('/'));
    return scheme + location.host + basePath;
  }
  const WS_BASE =
    (window.DUPLEX_WS_BASE && window.DUPLEX_WS_BASE.trim()) ||
    sameOriginWsBase();
  const MODEL = 'openbmb/MiniCPM-o-4_5';
  // ?runtime=fsm selects the experimental Track A continuous full-duplex runtime.
  const RUNTIME = (new URLSearchParams(location.search).get('runtime') || '').toLowerCase();
  const WS_URL = `${WS_BASE}/v1/realtime?duplex=1&model=${encodeURIComponent(MODEL)}` +
    (RUNTIME ? `&runtime=${encodeURIComponent(RUNTIME)}` : '');

  const TARGET_SR = 16000;     // mic upload rate
  const PLAYBACK_SR = 24000;   // model audio rate
  const SEND_INTERVAL_MS = 200;
  const MAX_RECONNECTS = 4;
  const INITIAL_WS_ATTEMPTS = 4;
  const QUERY = new URLSearchParams(location.search);
  const PLAYBACK_MODE = (QUERY.get('playback') || '').toLowerCase();
  const BUFFER_OUTPUT_AUDIO = PLAYBACK_MODE === 'buffered' || QUERY.get('buffered') === '1';
  const ALLOW_BARGE_IN = QUERY.get('bargein') === '1';
  const ECHO_GUARD_MS = 350;
  const ASSISTANT_AUDIO_IDLE_MS = 3500;
  const ASSISTANT_BOUNDARY_FALLBACK_MS = 30000;

  // ---- State ----
  let ws = null;
  let micStream = null;
  let micCtx = null;
  let micNode = null;
  let playCtx = null;
  let ttsNode = null;
  let running = false;          // call is up (WS + audio graph)
  let isPlaying = false;
  let pendingPCM = [];
  let sendTimer = null;
  let timerTimer = null;
  let captureSR = TARGET_SR;
  let playbackSR = PLAYBACK_SR;
  let callStart = 0;
  let mode = 'full';            // full | half | turn
  let micGateOpen = true;       // in half/turn, only true while button held
  let recording = false;        // turn mode: currently capturing an utterance
  let reconnects = 0;
  let manualStop = false;
  let sessionConfig = null;     // cached so reconnect re-applies voice/mode
  let playbackUnderruns = 0;
  let bufferedPlayback = new Map();
  let bufferedPlaybackTimers = new Map();
  let bufferedSources = new Set();
  let bufferedNextStartTime = 0;
  let assistantOutputActive = false;
  let assistantSawAudio = false;
  let assistantServerBoundarySeen = false;
  let assistantPlaybackDrainPending = false;
  let echoGuardTimer = null;
  let assistantAudioIdleTimer = null;
  let assistantBoundaryFallbackTimer = null;

  // transcript turn tracking
  let liveAsst = null;          // {el, text} for the assistant turn in progress
  let liveUser = null;

  function log(msg, cls) {
    const t = new Date().toLocaleTimeString();
    const line = `[${t}] ${msg}`;
    const span = document.createElement('div');
    if (cls) span.className = cls;
    span.textContent = line;
    logEl.insertBefore(span, logEl.firstChild);
  }
  function setStatus(s) { statusEl.textContent = s; }
  function setModelState(s) {
    stateEl.textContent = s;
    stateEl.className = 'badge ' + (s === 'speaking' ? 'speaking' : s === 'listening' ? 'listening' : 'idle');
  }
  function setMicLive(on) {
    micDot.classList.toggle('live', !!on);
    if (!on) vuFill.style.width = '0%';
  }
  function setFullModeMicGate(open) {
    if (mode !== 'full') return;
    micGateOpen = !!open;
    setMicLive(micGateOpen);
    if (!micGateOpen) pendingPCM = [];
  }
  function clearAssistantBoundaryFallback() {
    if (assistantBoundaryFallbackTimer) {
      clearTimeout(assistantBoundaryFallbackTimer);
      assistantBoundaryFallbackTimer = null;
    }
  }
  function assistantShouldWaitForServerBoundary() {
    return (
      mode === 'full' &&
      !ALLOW_BARGE_IN &&
      assistantSawAudio &&
      !assistantServerBoundarySeen &&
      !assistantPlaybackDrainPending
    );
  }
  function scheduleAssistantBoundaryFallback() {
    if (assistantBoundaryFallbackTimer || !assistantShouldWaitForServerBoundary()) return;
    assistantBoundaryFallbackTimer = setTimeout(() => {
      assistantBoundaryFallbackTimer = null;
      if (assistantShouldWaitForServerBoundary()) {
        log('assistant boundary timeout — reopening mic after audio idle');
        endAssistantOutput(ECHO_GUARD_MS);
      }
    }, ASSISTANT_BOUNDARY_FALLBACK_MS);
  }
  function beginAssistantOutput(hasAudio) {
    if (!assistantOutputActive) {
      assistantServerBoundarySeen = false;
      assistantPlaybackDrainPending = false;
      clearAssistantBoundaryFallback();
    }
    assistantOutputActive = true;
    if (hasAudio) {
      assistantSawAudio = true;
      clearAssistantBoundaryFallback();
    }
    if (echoGuardTimer) {
      clearTimeout(echoGuardTimer);
      echoGuardTimer = null;
    }
    if (!ALLOW_BARGE_IN) setFullModeMicGate(false);
    setModelState('speaking');
  }
  function scheduleAssistantAudioIdle(extraMs) {
    if (assistantAudioIdleTimer) clearTimeout(assistantAudioIdleTimer);
    const delay = Math.max(
      ASSISTANT_AUDIO_IDLE_MS,
      (Number(extraMs) > 0 ? Number(extraMs) : 0) + ECHO_GUARD_MS,
    );
    assistantAudioIdleTimer = setTimeout(() => {
      assistantAudioIdleTimer = null;
      if (assistantShouldWaitForServerBoundary()) {
        scheduleAssistantBoundaryFallback();
        return;
      }
      endAssistantOutput(0);
    }, delay);
  }
  function isFinalAssistantBoundary(e) {
    if (!e) return false;
    if (e.end_of_turn === true) return true;
    const resp = e.response || {};
    return resp.end_of_turn === true;
  }
  function markAssistantServerBoundary(e) {
    // In full-duplex auto-response mode the backend can emit response.listen /
    // response.done at a native segment boundary. That is a scheduler control
    // point, not proof that all assistant audio has played. Reopening the mic
    // here feeds the assistant's own speaker output back into Stage0.
    if (
      mode === 'full' &&
      !ALLOW_BARGE_IN &&
      assistantSawAudio &&
      !isFinalAssistantBoundary(e)
    ) {
      if (ttsNode) {
        assistantPlaybackDrainPending = true;
        ttsNode.port.postMessage({ type: 'drain' });
      }
      scheduleAssistantAudioIdle(ECHO_GUARD_MS);
      return;
    }
    assistantServerBoundarySeen = true;
    clearAssistantBoundaryFallback();
    if (assistantSawAudio) {
      if (ttsNode) ttsNode.port.postMessage({ type: 'drain' });
      scheduleAssistantAudioIdle(ECHO_GUARD_MS);
    } else {
      endAssistantOutput(0);
    }
  }
  function endAssistantOutput(delayMs) {
    assistantOutputActive = false;
    assistantSawAudio = false;
    assistantServerBoundarySeen = false;
    assistantPlaybackDrainPending = false;
    clearAssistantBoundaryFallback();
    if (assistantAudioIdleTimer) {
      clearTimeout(assistantAudioIdleTimer);
      assistantAudioIdleTimer = null;
    }
    if (echoGuardTimer) clearTimeout(echoGuardTimer);
    const finish = () => {
      echoGuardTimer = null;
      if (!running || assistantOutputActive) return;
      if (!ALLOW_BARGE_IN) setFullModeMicGate(mode === 'full');
      setModelState('listening');
    };
    const delay = Number(delayMs) >= 0 ? Number(delayMs) : ECHO_GUARD_MS;
    if (delay > 0) echoGuardTimer = setTimeout(finish, delay);
    else finish();
  }

  // ---- transcript turns ----
  function clearConvPlaceholder() {
    const e = convEl.querySelector('.empty');
    if (e) e.remove();
  }
  function appendTurn(who) {
    clearConvPlaceholder();
    const turn = document.createElement('div');
    turn.className = 'turn live ' + who;
    const w = document.createElement('div'); w.className = 'who'; w.textContent = who;
    const txt = document.createElement('div'); txt.className = 'text';
    turn.appendChild(w); turn.appendChild(txt);
    convEl.appendChild(turn);
    convEl.scrollTop = convEl.scrollHeight;
    return { el: turn, txt, text: '' };
  }
  function finalizeTurn(t) {
    if (t && t.el) t.el.classList.remove('live');
  }
  function asstDelta(d) {
    if (!liveAsst) liveAsst = appendTurn('assistant');
    liveAsst.text += d; liveAsst.txt.textContent = liveAsst.text;
    convEl.scrollTop = convEl.scrollHeight;
  }
  function asstDone() { finalizeTurn(liveAsst); liveAsst = null; }
  function userDelta(d) {
    if (!liveUser) liveUser = appendTurn('user');
    liveUser.text += d; liveUser.txt.textContent = liveUser.text;
    convEl.scrollTop = convEl.scrollHeight;
  }
  function userDone(full) {
    if (full && !liveUser) liveUser = appendTurn('user');
    if (full && liveUser) { liveUser.text = full; liveUser.txt.textContent = full; }
    finalizeTurn(liveUser); liveUser = null;
  }

  // ---- base64 helpers ----
  function int16ToBase64(int16) {
    const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
    let bin = '';
    const CH = 0x8000;
    for (let i = 0; i < bytes.length; i += CH) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
    return btoa(bin);
  }
  function base64ToBytes(b64) {
    const bin = atob(b64);
    const len = bin.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }
  function base64ToInt16(b64) {
    const bytes = base64ToBytes(b64);
    return new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
  }
  function float32ToInt16(float32) {
    const out = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const v = Math.max(-1, Math.min(1, Number(float32[i]) || 0));
      out[i] = v < 0 ? Math.round(v * 32768) : Math.round(v * 32767);
    }
    return out;
  }
  async function decodeOutputAudioDelta(e) {
    const d = e.delta || (e.response && e.response.audio);
    if (!d) return null;
    const rawFmt =
      (typeof e.format === 'string' ? e.format : '') ||
      (e.format && typeof e.format.type === 'string' ? e.format.type : '');
    const fmt = rawFmt.toLowerCase();
    const fallbackSR = e.sample_rate_hz || (e.format && e.format.rate) || PLAYBACK_SR;
    if (!fmt || fmt === 'pcm16' || fmt === 'pcm' || fmt === 'pcm_s16le' || fmt === 's16le') {
      return { pcm: base64ToInt16(d), sr: fallbackSR };
    }
    const bytes = base64ToBytes(d);
    if (fmt === 'pcm_f32le' || fmt === 'f32le' || fmt === 'audio/pcm_f32le') {
      return {
        pcm: float32ToInt16(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2)),
        sr: fallbackSR,
      };
    }
    if (fmt === 'wav' || fmt === 'audio/wav') {
      if (!playCtx) return null;
      const decoded = await playCtx.decodeAudioData(
        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
      );
      return {
        pcm: float32ToInt16(decoded.getChannelData(0)),
        sr: decoded.sampleRate || fallbackSR,
      };
    }
    return { pcm: base64ToInt16(d), sr: fallbackSR };
  }
  function fileToDataURI(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = () => reject(new Error('could not read reference audio'));
      r.readAsDataURL(file);
    });
  }

  // ---- resampler (capture/playback sample-rate conversion) ----
  function resampleInt16(int16In, srIn, srOut) {
    if (srIn === srOut) return int16In;
    if (srIn > srOut) {
      // anti-alias downsample: box-average over each input window
      const step = srIn / srOut;
      const outLen = Math.floor(int16In.length / step);
      const out = new Int16Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const start = Math.floor(i * step);
        const end = Math.min(Math.floor((i + 1) * step), int16In.length);
        let sum = 0, n = 0;
        for (let j = start; j < end; j++) { sum += int16In[j]; n++; }
        out[i] = n ? (sum / n) | 0 : 0;
      }
      return out;
    }
    const ratio = srOut / srIn;
    const outLen = Math.floor(int16In.length * ratio);
    const out = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const pos = i / ratio;
      const i0 = Math.floor(pos);
      const i1 = Math.min(i0 + 1, int16In.length - 1);
      const frac = pos - i0;
      out[i] = (int16In[i0] * (1 - frac) + int16In[i1] * frac) | 0;
    }
    return out;
  }

  // ---- VU meter from captured Int16 ----
  function updateVU(int16) {
    let peak = 0;
    for (let i = 0; i < int16.length; i += 8) { const a = Math.abs(int16[i]); if (a > peak) peak = a; }
    const pct = Math.min(100, (peak / 32768) * 140);
    vuFill.style.width = pct.toFixed(0) + '%';
  }

  function sendMicAppend(int16) {
    if (!int16 || int16.length === 0) return;
    ws.send(JSON.stringify({
      type: 'input_audio_buffer.append',
      audio: int16ToBase64(int16),
      format: 'pcm16',
      sample_rate_hz: TARGET_SR,
    }));
  }

  function flushMic(force) {
    if (!running || !ws || ws.readyState !== WebSocket.OPEN || pendingPCM.length === 0) return;
    // Mic gate: in half/turn mode only stream while the button is held. A forced
    // flush is used on PTT release so the last partial frame is not dropped.
    if (!force && !micGateOpen) {
      pendingPCM = [];
      return;
    }
    let total = 0;
    for (const c of pendingPCM) total += c.length;
    const cat = new Int16Array(total);
    let off = 0;
    for (const c of pendingPCM) { cat.set(c, off); off += c.length; }
    pendingPCM = [];
    const res = resampleInt16(cat, captureSR, TARGET_SR);
    sendMicAppend(res);
  }

  // ---- playback / barge-in ----
  function feedPlayback(int16, sourceSR) {
    if (!ttsNode || !int16 || int16.length === 0) return;
    const sr = Number(sourceSR) > 0 ? Number(sourceSR) : PLAYBACK_SR;
    const pcm = resampleInt16(int16, sr, playbackSR);
    ttsNode.port.postMessage({ type: 'audio', pcm }, [pcm.buffer]);
  }
  function sendPlaybackAck(playedMs) {
    if (!running || !ws || ws.readyState !== WebSocket.OPEN) return;
    const ms = Math.max(0, Math.round(Number(playedMs) || 0));
    if (ms <= 0) return;
    try {
      ws.send(JSON.stringify({
        type: 'playback.ack',
        played_ms: ms,
        committed_ms: ms,
      }));
    } catch (_) {}
  }
  function responseKey(e) {
    return (e && (e.response_id || (e.response && e.response.id))) || '__default__';
  }
  function bufferPlayback(e, int16, sourceSR) {
    if (!int16 || int16.length === 0) return;
    const sr = Number(sourceSR) > 0 ? Number(sourceSR) : PLAYBACK_SR;
    const pcm = resampleInt16(int16, sr, playbackSR);
    const key = responseKey(e);
    const chunks = bufferedPlayback.get(key) || [];
    chunks.push(pcm);
    bufferedPlayback.set(key, chunks);
    scheduleBufferedPlaybackFallback(e);
    scheduleAssistantAudioIdle((pcm.length * 1000) / playbackSR);
  }
  function clearBufferedPlaybackTimer(key) {
    const timer = bufferedPlaybackTimers.get(key);
    if (timer) clearTimeout(timer);
    bufferedPlaybackTimers.delete(key);
  }
  function scheduleBufferedPlaybackFallback(e) {
    const key = responseKey(e);
    clearBufferedPlaybackTimer(key);
    bufferedPlaybackTimers.set(key, setTimeout(() => {
      bufferedPlaybackTimers.delete(key);
      if (playBufferedResponse({ response_id: key })) {
        log('playback: flushed buffered audio after idle timeout');
      }
    }, 1800));
  }
  function playBufferedResponse(e) {
    const key = responseKey(e);
    clearBufferedPlaybackTimer(key);
    const chunks = bufferedPlayback.get(key);
    if (!chunks || chunks.length === 0) return false;
    bufferedPlayback.delete(key);
    let total = 0;
    for (const c of chunks) total += c.length;
    const cat = new Int16Array(total);
    let off = 0;
    for (const c of chunks) { cat.set(c, off); off += c.length; }
    playBufferedPcm(cat);
    return true;
  }
  function playBufferedPcm(pcm) {
    if (!playCtx || !pcm || pcm.length === 0) return;
    const audioBuffer = playCtx.createBuffer(1, pcm.length, playbackSR);
    const channel = audioBuffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) {
      channel[i] = Math.max(-1, Math.min(1, pcm[i] / 32768));
    }
    const source = playCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(playCtx.destination);
    bufferedSources.add(source);
    const startAt = Math.max(playCtx.currentTime + 0.02, bufferedNextStartTime);
    bufferedNextStartTime = startAt + audioBuffer.duration;
    source.onended = () => {
      bufferedSources.delete(source);
      if (bufferedSources.size === 0) {
        bufferedNextStartTime = 0;
        isPlaying = false;
        endAssistantOutput();
      }
      try { source.disconnect(); } catch (_) {}
    };
    isPlaying = true;
    beginAssistantOutput(true);
    source.start(startAt);
  }
  function stopBufferedSources() {
    for (const source of bufferedSources) {
      try { source.onended = null; source.stop(); } catch (_) {}
      try { source.disconnect(); } catch (_) {}
    }
    bufferedSources.clear();
    bufferedNextStartTime = 0;
  }
  function flushPlayback(reason) {
    if (ttsNode) ttsNode.port.postMessage({ type: 'clear' });
    stopBufferedSources();
    for (const key of bufferedPlaybackTimers.keys()) clearBufferedPlaybackTimer(key);
    bufferedPlayback.clear();
    if (isPlaying) log('barge-in: flush playback (' + reason + ')');
    isPlaying = false;
    endAssistantOutput(0);
  }

  // ---- WS event handling ----
  function handleEvent(e) {
    switch (e.type) {
      case 'session.created': log('session.created'); reconnects = 0; break;
      case 'session.updated': break;
      case 'response.listen':
        markAssistantServerBoundary(e);
        // Turn-based: a listen decision is a complete (empty) reply. Without
        // this the status would hang on "waiting for reply" since the model
        // emits no audio and may not send a prompt response.done.
        if (mode === 'turn') setStatus('model listened — no reply (hold to record; try a fuller sentence)');
        break;
      case 'response.speak':
      case 'response.created':
        beginAssistantOutput(false);
        break;
      case 'response.audio.delta': {
        beginAssistantOutput(true);
        decodeOutputAudioDelta(e).then((decoded) => {
          if (!decoded || !decoded.pcm || decoded.pcm.length === 0) return;
          const pcm = decoded.pcm;
          const sr = Number(decoded.sr) > 0 ? Number(decoded.sr) : PLAYBACK_SR;
          if (BUFFER_OUTPUT_AUDIO) bufferPlayback(e, pcm, sr);
          else {
            isPlaying = true;
            feedPlayback(pcm, sr);
            scheduleAssistantAudioIdle((pcm.length * 1000) / sr);
          }
        }).catch((err) => {
          log('playback decode failed: ' + ((err && err.message) || err), 'err');
        });
        break;
      }
      case 'response.audio.done':
        if (BUFFER_OUTPUT_AUDIO) playBufferedResponse(e);
        markAssistantServerBoundary(e);
        break;
      case 'response.audio_transcript.delta':
        if (e.delta) asstDelta(e.delta);
        break;
      case 'response.audio_transcript.done':
        asstDone();
        break;
      case 'conversation.item.input_audio_transcription.delta':
        if (e.delta) userDelta(e.delta);
        break;
      case 'conversation.item.input_audio_transcription.completed':
        userDone(e.transcript || null);
        break;
      case 'conversation.item.truncated':
        flushPlayback('truncated'); asstDone();
        break;
      case 'response.done': {
        const st = (e.response && e.response.status) || e.status;
        if (st === 'cancelled') flushPlayback('cancelled');
        else {
          if (BUFFER_OUTPUT_AUDIO) playBufferedResponse(e);
          markAssistantServerBoundary(e);
        }
        asstDone();
        if (mode === 'turn') setStatus('idle (hold to record)');
        break;
      }
      case 'error':
        log('server error: ' + JSON.stringify(e.error || e.code || e), 'err');
        break;
      default: break;
    }
  }

  // ---- build the initial session.update payload (voice + turn detection) ----
  async function buildSessionConfig() {
    const session = { modalities: ['audio', 'text'] };
    if (turnDetSel.value === 'server_vad') session.turn_detection = { type: 'server_vad' };

    if (voiceSel.value === '__ref__') {
      const f = refFile.files && refFile.files[0];
      if (!f) throw new Error('select a reference audio file (or pick Default voice)');
      const dataURI = await fileToDataURI(f);     // data:audio/...;base64,...
      session.extra_body = { ref_audio: dataURI }; // adapter resolves before runtime open
      log('voice: cloning from reference audio "' + f.name + '"');
    } else {
      session.voice = voiceSel.value;              // named voice, e.g. "default"
      log('voice: ' + voiceSel.value);
    }
    // Full-duplex: ask the server to run continuous per-chunk generation (model
    // decides speak/listen) instead of waiting for an explicit response.create.
    if (mode === 'full') session.extra_body = Object.assign({}, session.extra_body, { auto_response: true });
    return session;
  }

  // ---- push-to-talk wiring ----
  function pttDown() {
    if (!running || mode === 'full') return;
    pendingPCM = [];
    micGateOpen = true;
    recording = true;
    pttBtn.classList.add('held');
    setMicLive(true);
    setStatus(mode === 'turn' ? 'recording…' : 'talking…');
    // clear any input buffered from before this press
    try { ws.send(JSON.stringify({ type: 'input_audio_buffer.clear' })); } catch (_) {}
  }
  function pttUp() {
    if (!running || mode === 'full' || !recording) return;
    recording = false;
    pttBtn.classList.remove('held');
    flushMic(true); // send whatever is still pending while gate was open
    micGateOpen = false;
    setMicLive(false);
    if (mode === 'turn') {
      // turn-based: finalize the utterance and ask for a reply
      try {
        ws.send(JSON.stringify({ type: 'input_audio_buffer.commit', final: true }));
        ws.send(JSON.stringify({ type: 'response.create' }));
        setStatus('sent — waiting for reply');
        log('turn committed -> response.create');
      } catch (_) {}
    } else {
      // half-duplex: stop streaming; model decides when to reply
      setStatus('idle (hold to talk)');
    }
  }
  pttBtn.addEventListener('mousedown', pttDown);
  pttBtn.addEventListener('mouseup', pttUp);
  pttBtn.addEventListener('mouseleave', () => { if (recording) pttUp(); });
  pttBtn.addEventListener('touchstart', (ev) => { ev.preventDefault(); pttDown(); }, { passive: false });
  pttBtn.addEventListener('touchend', (ev) => { ev.preventDefault(); pttUp(); }, { passive: false });

  // ---- timer ----
  function fmt(s) { const m = (s / 60) | 0, ss = s % 60; return String(m).padStart(2, '0') + ':' + String(ss).padStart(2, '0'); }
  function startTimer() { callStart = Date.now(); timerEl.textContent = '00:00';
    timerTimer = setInterval(() => { timerEl.textContent = fmt(((Date.now() - callStart) / 1000) | 0); }, 1000); }
  function stopTimer() { if (timerTimer) { clearInterval(timerTimer); timerTimer = null; } }

  // ---- config lock during call ----
  function lockConfig(lock) {
    [voiceSel, refFile, modeSel, turnDetSel].forEach((el) => { if (el) el.disabled = lock; });
    lockNote.style.display = lock ? 'block' : 'none';
  }

  // ---- start / stop ----
  async function startCall() {
    if (running) return;
    callBtn.disabled = true;
    manualStop = false;
    playbackUnderruns = 0;
    for (const key of bufferedPlaybackTimers.keys()) clearBufferedPlaybackTimer(key);
    bufferedPlayback.clear();
    stopBufferedSources();
    mode = modeSel.value;
    setStatus('connecting…');
    try {
      sessionConfig = await buildSessionConfig();

      playCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: PLAYBACK_SR });
      playbackSR = playCtx.sampleRate || PLAYBACK_SR;
      if (!BUFFER_OUTPUT_AUDIO) {
        await playCtx.audioWorklet.addModule('static/ttsPlaybackProcessor.js');
        ttsNode = new AudioWorkletNode(playCtx, 'tts-playback-processor');
        ttsNode.port.onmessage = (ev) => {
          const data = ev.data || {};
          if (data.type === 'ttsPlaybackStarted') {
            isPlaying = true;
            beginAssistantOutput(true);
            return;
          }
          if (data.type === 'ttsPlaybackStopped') {
            isPlaying = false;
            sendPlaybackAck(data.playedMs);
            if (assistantServerBoundarySeen || assistantPlaybackDrainPending) endAssistantOutput(ECHO_GUARD_MS);
            return;
          }
          if (data.type === 'ttsPlaybackUnderrun') {
            playbackUnderruns = data.count || (playbackUnderruns + 1);
            if (playbackUnderruns === 1 || playbackUnderruns % 5 === 0) {
              log('playback underrun #' + playbackUnderruns +
                  ' — rebuffering TTS audio (target=' + (data.minStartMs || '?') + 'ms)');
            }
          }
        };
        ttsNode.port.postMessage({
          type: 'config',
          minStartMs: 700,
          maxStartMs: 1800,
          drainGraceMs: 1200,
          gapFillMs: 1000,
        });
        ttsNode.connect(playCtx.destination);
      }
      await playCtx.resume();

      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true,
                 autoGainControl: true, sampleRate: { ideal: TARGET_SR } },
      });
      // Capture at 16k directly so the browser does anti-aliased resampling
      // (naive JS downsample 48k->16k aliases -> garbled audio the model mishears).
      try { micCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SR }); }
      catch (_) { micCtx = new (window.AudioContext || window.webkitAudioContext)(); }
      captureSR = micCtx.sampleRate;
      await micCtx.audioWorklet.addModule('static/pcmWorkletProcessor.js');
      const src = micCtx.createMediaStreamSource(micStream);
      micNode = new AudioWorkletNode(micCtx, 'pcm-worklet-processor');
      micNode.port.onmessage = (ev) => {
        if (!running) return;
        const i16 = new Int16Array(ev.data);
        if (!micGateOpen) return;
        updateVU(i16);
        pendingPCM.push(i16);
      };
      src.connect(micNode);
      const sink = micCtx.createGain(); sink.gain.value = 0;
      micNode.connect(sink).connect(micCtx.destination);

      await openWSWithRetry();

      running = true;
      // mode-specific gating: full streams continuously; half/turn gate on the PTT button.
      micGateOpen = (mode === 'full');
      pttBtn.style.display = (mode === 'full') ? 'none' : 'inline-block';
      pttBtn.textContent = (mode === 'turn') ? 'Hold to record' : 'Hold to talk';
      setMicLive(mode === 'full');

      sendTimer = setInterval(flushMic, SEND_INTERVAL_MS);
      callBtn.textContent = 'Hang up';
      callBtn.classList.add('active');
      callBtn.disabled = false;
      lockConfig(true);
      startTimer();
      setStatus('in call (' + mode + ', captureSR=' + captureSR + '->16k, playbackSR=' + playbackSR +
                ', playback=' + (BUFFER_OUTPUT_AUDIO ? 'buffered' : 'streaming-default') + ')');
      setModelState('listening');
      log('call started — mode=' + mode);
    } catch (err) {
      const m = err && err.message ? err.message : err;
      log('start failed: ' + m, 'err');
      setStatus('error: ' + m);
      await stopCall();
      callBtn.disabled = false;
    }
  }

  function openWS() {
    return new Promise((resolve, reject) => {
      ws = new WebSocket(WS_URL);
      ws.binaryType = 'arraybuffer';
      let opened = false;
      let settled = false;
      ws.onopen = () => {
        opened = true;
        settled = true;
        ws.send(JSON.stringify({ type: 'session.update', session: sessionConfig }));
        log('ws open -> session.update (turn_detection=' +
            (turnDetSel.value === 'server_vad' ? 'server_vad' : 'model-driven') + ')');
        resolve();
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data !== 'string') return;
        let e; try { e = JSON.parse(ev.data); } catch (_) { return; }
        handleEvent(e);
      };
      ws.onerror = () => {
        if (!opened && !settled) {
          settled = true;
          reject(new Error('ws connect failed: ' + WS_URL));
        }
      };
      ws.onclose = (ev) => {
        const detail = 'code=' + ev.code + ' clean=' + ev.wasClean + (ev.reason ? ' reason=' + ev.reason : '');
        if (!opened && !settled) {
          settled = true;
          reject(new Error('ws closed before open: ' + detail + ' ' + WS_URL));
          return;
        }
        if (!running || manualStop) { log('ws closed'); return; }
        log('ws closed unexpectedly (' + detail + ')', 'err');
        tryReconnect();
      };
    });
  }

  async function openWSWithRetry() {
    let lastErr = null;
    for (let attempt = 1; attempt <= INITIAL_WS_ATTEMPTS; attempt++) {
      try {
        if (attempt > 1) setStatus('connecting… websocket retry ' + attempt + '/' + INITIAL_WS_ATTEMPTS);
        await openWS();
        return;
      } catch (err) {
        lastErr = err;
        log('ws connect attempt ' + attempt + '/' + INITIAL_WS_ATTEMPTS + ' failed: ' +
            ((err && err.message) || err), 'err');
        try { if (ws) ws.close(); } catch (_) {}
        ws = null;
        if (attempt < INITIAL_WS_ATTEMPTS) {
          await new Promise((resolve) => setTimeout(resolve, 350 * attempt));
        }
      }
    }
    throw lastErr || new Error('ws connect failed: ' + WS_URL);
  }

  function tryReconnect() {
    if (manualStop || !running) return;
    if (reconnects >= MAX_RECONNECTS) { log('giving up after ' + reconnects + ' reconnects', 'err'); stopCall(); return; }
    reconnects += 1;
    const delay = Math.min(4000, 500 * Math.pow(2, reconnects - 1));
    setStatus('reconnecting (' + reconnects + ')…');
    flushPlayback('reconnect');
    setTimeout(() => {
      if (manualStop || !running) return;
      openWS().then(() => {
        log('reconnected');
        setStatus('in call (' + mode + ')');
      }).catch((e) => { log('reconnect failed: ' + e.message, 'err'); tryReconnect(); });
    }, delay);
  }

  async function stopCall() {
    manualStop = true;
    running = false;
    recording = false;
    pttBtn.classList.remove('held');
    pttBtn.style.display = 'none';
    callBtn.classList.remove('active');
    callBtn.textContent = 'Start call';
    if (sendTimer) { clearInterval(sendTimer); sendTimer = null; }
    stopTimer();
    pendingPCM = [];
    flushPlayback('hangup');
    asstDone(); userDone(null);
    try { if (ws && ws.readyState === WebSocket.OPEN) ws.close(); } catch (_) {}
    ws = null;
    try { if (micNode) micNode.disconnect(); } catch (_) {}
    try { if (micStream) micStream.getTracks().forEach((tr) => tr.stop()); } catch (_) {}
    try { if (micCtx) await micCtx.close(); } catch (_) {}
    try { if (playCtx) await playCtx.close(); } catch (_) {}
    micNode = null; micStream = null; micCtx = null; ttsNode = null; playCtx = null;
    setMicLive(false);
    setStatus('idle');
    setModelState('idle');
    lockConfig(false);
    log('call ended');
  }

  // ---- UI wiring ----
  voiceSel.addEventListener('change', () => {
    refField.style.display = (voiceSel.value === '__ref__') ? 'flex' : 'none';
  });
  callBtn.addEventListener('click', () => { if (running) stopCall(); else startCall(); });

  setModelState('idle');
  setStatus('idle — WS ' + WS_URL);
  log('ready. WS target: ' + WS_URL);
})();
