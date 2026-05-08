(() => {
  const feed = document.getElementById("feed");
  const btnRec = document.getElementById("btnModeRec");
  const btnReg = document.getElementById("btnModeReg");
  const modeHint = document.getElementById("modeHint");
  const statusPill = document.getElementById("statusPill");
  const settingsForm = document.getElementById("settingsForm");
  const settingsMsg = document.getElementById("settingsMsg");
  const registerForm = document.getElementById("registerForm");
  const btnRegister = document.getElementById("btnRegister");
  const regProgress = document.getElementById("regProgress");
  const regBar = document.getElementById("regBar");
  const regMsg = document.getElementById("regMsg");
  const identityList = document.getElementById("identityList");
  const btnRefreshIds = document.getElementById("btnRefreshIds");
  const alignedFace = document.getElementById("alignedFace");

  let mode = "recognition";
  const regLog = (...args) => console.debug("[register]", ...args);

  function setMode(next) {
    mode = next === "register" ? "register" : "recognition";
    btnRec.classList.toggle("active", mode === "recognition");
    btnReg.classList.toggle("active", mode === "register");
    feed.src = `/video_feed?mode=${mode}&t=${Date.now()}`;
    modeHint.textContent =
      mode === "recognition"
        ? "Live identification on the largest face in view."
        : "Preview for enrollment — largest face is tagged; use the form to capture embeddings.";
  }

  btnRec.addEventListener("click", () => setMode("recognition"));
  btnReg.addEventListener("click", () => setMode("register"));

  async function pollStatus() {
    try {
      const r = await fetch("/status");
      const j = await r.json();
      const busy = j.registration_busy ? " · registering" : "";
      statusPill.textContent = `${j.enrolled_count} enrolled · ${j.camera_source}${busy} · ${j.memory_rss_mb} MB RSS`;
    } catch {
      statusPill.textContent = "status unavailable";
    }
  }

  function setStage(id, ok, detail) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle("ok", Boolean(ok));
    el.classList.toggle("warn", !ok);
    const span = el.querySelector("span");
    if (span) span.textContent = detail || (ok ? "ok" : "waiting");
  }

  async function pollPipeline() {
    try {
      const r = await fetch(`/pipeline/status?t=${Date.now()}`);
      const j = await r.json();
      const dets = j.detections || [];
      const primary = dets.find((d) => d.primary) || dets[0];
      const timings = j.pipeline_timings_ms || {};
      const rec = j.recognition;

      setStage("stageCapture", Boolean(j.camera_source), j.camera_source || "waiting");
      setStage("stageDetect", dets.length > 0, `${j.num_faces || 0} face(s) · ${j.detection_ms || 0}ms`);
      setStage("stageAlign", Boolean(timings.align), timings.align ? `${timings.align}ms` : "waiting");
      setStage("stageEmbed", Boolean(timings.embed), timings.embed ? `${timings.embed}ms` : "waiting");
      setStage(
        "stageRecognize",
        Boolean(rec && rec.name),
        rec && rec.name ? `${rec.name} ${Number(rec.confidence || 0).toFixed(2)}` : "waiting",
      );

      document.getElementById("pipeCamera").textContent = j.camera_source || "-";
      document.getElementById("pipeFaces").textContent = String(j.num_faces ?? "-");
      document.getElementById("pipeBbox").textContent = primary ? primary.bbox_xywh.join(", ") : "-";
      document.getElementById("pipeConf").textContent = primary ? Number(primary.confidence).toFixed(3) : "-";
      document.getElementById("pipeRec").textContent = rec
        ? `${rec.accepted ? "accepted" : "rejected"} · ${rec.name} · ${Number(rec.confidence).toFixed(3)}`
        : mode === "register"
          ? "enrollment preview"
          : "-";
      document.getElementById("pipeTimings").textContent =
        `detect ${j.detection_ms || 0}ms · align ${timings.align || 0}ms · embed ${timings.embed || 0}ms`;
      document.getElementById("pipeError").textContent = j.error || "";
      if (alignedFace && dets.length > 0) {
        alignedFace.src = `/pipeline/aligned_face.jpg?t=${Date.now()}`;
      }
    } catch (e) {
      setStage("stageCapture", false, "offline");
    }
  }

  async function loadSettings() {
    settingsMsg.textContent = "";
    settingsMsg.classList.remove("err");
    try {
      const r = await fetch("/settings");
      const j = await r.json();
      document.getElementById("fieldRegistration").value = j.registration;
      document.getElementById("fieldExemplarSel").value = j.exemplar_selection;
      document.getElementById("fieldRecognition").value = j.recognition;
      document.getElementById("fieldExemplarK").value = j.exemplar_k;
      document.getElementById("fieldThr").value = j.confidence_threshold;
      const locked = j.locked;
      [...settingsForm.elements].forEach((el) => {
        if (el.name && el.type !== "submit") el.disabled = locked;
      });
      document.getElementById("btnSaveSettings").disabled = locked;
      if (locked) {
        settingsMsg.textContent = "Settings locked after first enrollment.";
      }
    } catch (e) {
      settingsMsg.textContent = String(e);
      settingsMsg.classList.add("err");
    }
  }

  settingsForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    settingsMsg.textContent = "";
    settingsMsg.classList.remove("err");
    const body = {
      registration: document.getElementById("fieldRegistration").value,
      exemplar_selection: document.getElementById("fieldExemplarSel").value,
      recognition: document.getElementById("fieldRecognition").value,
      exemplar_k: Number(document.getElementById("fieldExemplarK").value),
      confidence_threshold: Number(document.getElementById("fieldThr").value),
    };
    try {
      const r = await fetch("/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = j.detail;
        const msg = typeof d === "string" ? d : JSON.stringify(d || r.statusText);
        throw new Error(msg);
      }
      settingsMsg.textContent = "Saved.";
      await loadSettings();
      await refreshIdentities();
    } catch (e) {
      settingsMsg.textContent = String(e.message || e);
      settingsMsg.classList.add("err");
    }
  });

  async function refreshIdentities() {
    identityList.innerHTML = "";
    try {
      const r = await fetch("/identities");
      const j = await r.json();
      for (const it of j.items || []) {
        const li = document.createElement("li");
        const span = document.createElement("span");
        const ec =
          it.exemplar_count == null ? "Gaussian / no exemplar store" : `${it.exemplar_count} exemplars`;
        span.textContent = `${it.name} · ${ec}`;
        const rm = document.createElement("button");
        rm.type = "button";
        rm.textContent = "Remove";
        rm.addEventListener("click", async () => {
          if (!confirm(`Remove ${it.name}?`)) return;
          const dr = await fetch(`/identities/${encodeURIComponent(it.name)}`, { method: "DELETE" });
          if (!dr.ok) {
            alert(await dr.text());
            return;
          }
          await refreshIdentities();
          await loadSettings();
          await pollStatus();
        });
        li.appendChild(span);
        li.appendChild(rm);
        identityList.appendChild(li);
      }
      if (!j.items || j.items.length === 0) {
        const li = document.createElement("li");
        li.textContent = "No identities yet.";
        identityList.appendChild(li);
      }
    } catch (e) {
      identityList.innerHTML = `<li>${String(e)}</li>`;
    }
  }

  btnRefreshIds.addEventListener("click", refreshIdentities);

  registerForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    regMsg.textContent = "";
    regMsg.classList.remove("err");
    regProgress.hidden = false;
    regBar.style.width = "0%";
    setMode("register");

    const name = document.getElementById("regName").value.trim();
    const n_frames = Number(document.getElementById("regFrames").value);

    btnRegister.disabled = true;
    let es = null;
    try {
      regLog("submit", { name, n_frames });
      const r = await fetch("/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, n_frames }),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = j.detail;
        throw new Error(typeof d === "string" ? d : JSON.stringify(d || r.statusText));
      }
      const startSeq = Number(j.start_seq ?? -1);
      regLog("register accepted", { startSeq });

      es = new EventSource(`/register/stream?after_seq=${encodeURIComponent(startSeq)}`);
      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          regLog("event", data);
          regMsg.textContent = data.message || data.phase || "";
          if (data.phase === "capturing" && data.target) {
            const p = Math.round((100 * (data.current || 0)) / data.target);
            regBar.style.width = `${p}%`;
          }
          if (data.phase === "captured") {
            regBar.style.width = "100%";
            setMode("recognition");
          }
          if (data.phase === "training") {
            regBar.style.width = "100%";
            setMode("recognition");
          }
          if (data.phase === "done") {
            regMsg.textContent = `Done: ${data.identity} · ${data.elapsed_s}s · ${data.total_identities} total · back to recognition`;
            es.close();
            setMode("recognition");
            regProgress.hidden = true;
            refreshIdentities();
            loadSettings();
            pollStatus();
            btnRegister.disabled = false;
          }
          if (data.phase === "error") {
            regMsg.textContent = data.message || "Error";
            regMsg.classList.add("err");
            es.close();
            regProgress.hidden = true;
            btnRegister.disabled = false;
          }
        } catch (err) {
          regLog("event parse failure", err);
        }
      };
      es.onerror = () => {
        regLog("stream error");
        es.close();
        regMsg.textContent = "Registration stream disconnected. Check backend logs and retry.";
        regMsg.classList.add("err");
        btnRegister.disabled = false;
      };
    } catch (e) {
      if (es) es.close();
      regLog("submit failed", e);
      regMsg.textContent = String(e.message || e);
      regMsg.classList.add("err");
      btnRegister.disabled = false;
    }
  });

  setMode("recognition");
  pollStatus();
  pollPipeline();
  loadSettings();
  refreshIdentities();
  setInterval(pollStatus, 4000);
  setInterval(pollPipeline, 900);
})();
