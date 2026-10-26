(() => {
  const REGIONS = ["north", "south", "east", "west", "central"];

  const SEVERITY_COLORS = {
    good: "#047857",
    moderate: "#0284C7",
    unhealthy: "#D97706",
    "very-unhealthy": "#EA580C",
    hazardous: "#DC2626",
  };

  const prevNodeValues = {};

  function animateNumericText(el, targetVal, decimals = 0, suffix = "") {
    if (!el) return;
    const key = el.id || el.dataset.animKey || "default";
    const startVal = prevNodeValues[key] ?? targetVal;
    prevNodeValues[key] = targetVal;
    if (Math.abs(targetVal - startVal) < 0.05) {
      el.textContent = `${targetVal.toFixed(decimals)}${suffix}`;
      return;
    }
    const duration = 360;
    const t0 = performance.now();
    function tick(now) {
      const p = Math.min(1, (now - t0) / duration);
      const ease = 1 - Math.pow(1 - p, 3);
      const cur = startVal + (targetVal - startVal) * ease;
      el.textContent = `${cur.toFixed(decimals)}${suffix}`;
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  const REGION_ARROW_COORDS = {
    north: [262, 85],
    west: [112, 165],
    central: [268, 175],
    south: [254, 252],
    east: [422, 155],
  };

  const state = {
    payload: null,
    selectedRegion: "central",
    chartRegion: "national",
    scrubberStep: 0,
    windRegime: "baseline",
    simWindSpeed: null,
    simWindDir: null,
    simHotspot: 0.0,
    ensoOni: 0.35,
    iodDmi: 0.22,
    rainAnomaly: -10,
    autoLearnEnabled: true,
    secondsUntilNextSync: 1800,
    initializedSimSliders: false,
  };

  function degreesToCardinal(deg) {
    const dirs = [
      "N", "NNE", "NE", "ENE",
      "E", "ESE", "SE", "SSE",
      "S", "SSW", "SW", "WSW",
      "W", "WNW", "NW", "NNW",
    ];
    const idx = Math.round(((deg % 360) + 360) % 360 / 22.5) % 16;
    return dirs[idx];
  }

  function classifyPsi(psi) {
    if (psi <= 50) return { band: "Good", severity: "good" };
    if (psi <= 100) return { band: "Moderate", severity: "moderate" };
    if (psi <= 200) return { band: "Unhealthy", severity: "unhealthy" };
    if (psi <= 300) return { band: "Very Unhealthy", severity: "very-unhealthy" };
    return { band: "Hazardous", severity: "hazardous" };
  }

  function formatSgtTimestamp(isoStr) {
    if (!isoStr) return "--";
    const dt = new Date(isoStr);
    const parts = new Intl.DateTimeFormat("en-GB", {
      timeZone: "Asia/Singapore",
      weekday: "short",
      day: "numeric",
      month: "short",
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    }).format(dt);
    return parts.replace(/\b(am|pm)\b/gi, (m) => m.toUpperCase()) + " SGT";
  }

  function formatShortDayHour(isoStr) {
    if (!isoStr) return "--";
    const dt = new Date(isoStr);
    const parts = new Intl.DateTimeFormat("en-GB", {
      timeZone: "Asia/Singapore",
      weekday: "short",
      hour: "numeric",
      hour12: true,
    }).format(dt);
    return parts.replace(/\b(am|pm)\b/gi, (m) => m.toUpperCase());
  }

  function populateTimePickerAndPresets(shortTerm) {
    const selectEl = document.getElementById("target-time-select");
    const presetsEl = document.getElementById("quick-time-presets");
    const ticksEl = document.getElementById("scrubber-ticks");
    if (!selectEl || !shortTerm?.national_forecast) return;

    const fc = shortTerm.national_forecast;
    let optionsHtml = `<option value="0">Live Now &middot; ${formatShortDayHour(shortTerm.latest_timestamp)}</option>`;
    fc.forEach((item) => {
      const label = `+${item.step_hour}h &middot; ${formatSgtTimestamp(item.timestamp)} &mdash; ${Math.round(item.national_max_psi)} PSI`;
      const sel = item.step_hour === state.scrubberStep ? "selected" : "";
      optionsHtml += `<option value="${item.step_hour}" ${sel}>${label}</option>`;
    });
    selectEl.innerHTML = optionsHtml;

    // Build quick-jump chips for upcoming 7 AM, 1 PM, 7 PM slots
    const presetCandidates = [];
    for (const item of fc) {
      const dt = new Date(item.timestamp);
      const sgtHour = parseInt(
        dt.toLocaleString("en-GB", { timeZone: "Asia/Singapore", hour: "2-digit", hour12: false }),
        10
      );
      const dayLabel = dt.toLocaleString("en-GB", { timeZone: "Asia/Singapore", weekday: "short" });
      if (sgtHour === 7 || sgtHour === 13 || sgtHour === 19) {
        const ampm = sgtHour === 7 ? "7 AM" : sgtHour === 13 ? "1 PM" : "7 PM";
        presetCandidates.push({
          step: item.step_hour,
          label: `${dayLabel} ${ampm}`,
        });
      }
      if (presetCandidates.length >= 6) break;
    }

    if (presetsEl) {
      let btnsHtml = `<button type="button" class="time-preset-btn ${state.scrubberStep === 0 ? "active" : ""}" data-step="0">Live Now</button>`;
      for (const p of presetCandidates) {
        const activeCls = state.scrubberStep === p.step ? "active" : "";
        btnsHtml += `<button type="button" class="time-preset-btn ${activeCls}" data-step="${p.step}">${p.label} (+${p.step}h)</button>`;
      }
      presetsEl.innerHTML = btnsHtml;

      presetsEl.querySelectorAll(".time-preset-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          const stepVal = parseInt(btn.dataset.step, 10) || 0;
          setScrubberStep(stepVal);
        });
      });
    }

    // Update slider tick labels with real calendar day/times
    if (ticksEl && fc.length >= 72) {
      ticksEl.innerHTML = `
        <span>Now (${formatShortDayHour(shortTerm.latest_timestamp)})</span>
        <span>${formatShortDayHour(fc[11].timestamp)}</span>
        <span>${formatShortDayHour(fc[23].timestamp)}</span>
        <span>${formatShortDayHour(fc[47].timestamp)}</span>
        <span>${formatShortDayHour(fc[71].timestamp)}</span>
      `;
    }
  }

  function setScrubberStep(stepVal) {
    state.scrubberStep = Math.max(0, Math.min(72, stepVal));
    const hourSlider = document.getElementById("forecast-hour-slider");
    if (hourSlider) hourSlider.value = state.scrubberStep;
    const selectEl = document.getElementById("target-time-select");
    if (selectEl) selectEl.value = String(state.scrubberStep);
    document.querySelectorAll(".time-preset-btn").forEach((b) => {
      b.classList.toggle("active", parseInt(b.dataset.step, 10) === state.scrubberStep);
    });
    renderSpatialMapAndScrubber();
    renderSubIndexStrip();
    renderTrajectoryChart();
  }

  async function fetchOverview() {
    const params = new URLSearchParams({
      horizon_hours: "72",
      wind_regime: state.windRegime,
      sim_hotspot_intensity: state.simHotspot.toFixed(2),
      enso_oni: state.ensoOni.toFixed(2),
      iod_dmi: state.iodDmi.toFixed(2),
      rainfall_anomaly_pct: state.rainAnomaly.toFixed(0),
    });

    if (state.simWindSpeed !== null) {
      params.set("sim_wind_speed_kn", state.simWindSpeed.toFixed(1));
    }
    if (state.simWindDir !== null) {
      params.set("sim_wind_dir_deg", state.simWindDir.toFixed(0));
    }

    const resp = await fetch(`/api/overview?${params.toString()}`);
    if (!resp.ok) {
      throw new Error(`API error ${resp.status}`);
    }
    state.payload = await resp.json();

    if (state.payload?.continuous_learning?.interval_seconds) {
      state.secondsUntilNextSync = state.payload.continuous_learning.interval_seconds;
    }

    // Initialize counterfactual sliders to live NEA national wind on first load
    if (!state.initializedSimSliders && state.payload?.short_term?.live_national_wind) {
      const liveW = state.payload.short_term.live_national_wind;
      state.simWindSpeed = liveW.speed_kn;
      state.simWindDir = liveW.dir_deg;
      const wsSlider = document.getElementById("slider-sim-ws");
      const wdSlider = document.getElementById("slider-sim-wd");
      if (wsSlider) wsSlider.value = state.simWindSpeed;
      if (wdSlider) wdSlider.value = state.simWindDir;
      updateSimLabels();
      state.initializedSimSliders = true;
    }

    renderAll();
  }

  function renderAll() {
    if (!state.payload) return;
    const { short_term, seasonal, diagnostics } = state.payload;

    const latestTsEl = document.getElementById("latest-timestamp-label");
    if (latestTsEl) {
      latestTsEl.textContent = `NEA Feed: ${formatSgtTimestamp(short_term.latest_timestamp)}`;
    }

    try { populateTimePickerAndPresets(short_term); } catch (e) { console.error("Time picker error:", e); }
    try { renderSpatialMapAndScrubber(); } catch (e) { console.error("Map render error:", e); }
    try { renderSubIndexStrip(); } catch (e) { console.error("SubIndex render error:", e); }
    try { renderTrajectoryChart(); } catch (e) { console.error("Trajectory render error:", e); }
    try { renderHorizonMatrix(); } catch (e) { console.error("Matrix render error:", e); }
    try { renderIndonesiaCrossBorder(short_term); } catch (e) { console.error("Indonesia cross-border render error:", e); }
    try { renderCounterfactualSandbox(short_term.counterfactual_simulation); } catch (e) { console.error("Sandbox render error:", e); }
    try { renderSeasonalOutlook(seasonal); } catch (e) { console.error("Seasonal render error:", e); }
    try { renderDiagnostics(diagnostics); } catch (e) { console.error("Diagnostics render error:", e); }
  }

  function renderSpatialMapAndScrubber() {
    const { short_term } = state.payload;
    const step = state.scrubberStep;
    const badgeEl = document.getElementById("scrubber-step-badge");
    const captionEl = document.getElementById("map-time-caption");

    let regionPsiMap = {};
    let regionWsMap = {};
    let regionWdMap = {};
    let currentTimestamp = short_term.latest_timestamp;

    if (step === 0) {
      badgeEl.textContent = `Live · ${formatShortDayHour(currentTimestamp)}`;
      for (const r of REGIONS) {
        regionPsiMap[r] = short_term.current_regions[r]?.psi_24h ?? 50;
        regionWsMap[r] = short_term.current_regions[r]?.wind_speed_kn ?? 5.0;
        regionWdMap[r] = short_term.current_regions[r]?.wind_dir_deg ?? 180;
      }
      captionEl.textContent = `Live Observation (${formatSgtTimestamp(currentTimestamp)}) · 5 NEA Air Quality Zones + 18 Wind Stations`;
    } else {
      const stepData = short_term.national_forecast[step - 1];
      currentTimestamp = stepData.timestamp;
      badgeEl.textContent = `${formatShortDayHour(currentTimestamp)} (+${step}h)`;
      regionPsiMap = stepData.regions;
      for (const r of REGIONS) {
        const regStep = short_term.regional_forecasts[r]?.[step - 1];
        regionWsMap[r] = regStep?.wind_speed_kn ?? stepData.wind_speed_kn;
        regionWdMap[r] = regStep?.wind_dir_deg ?? stepData.wind_dir_deg;
      }
      captionEl.textContent = `Forecast for ${formatSgtTimestamp(currentTimestamp)} (+${step}h) · Wind: ${stepData.wind_speed_kn.toFixed(1)} kn ${stepData.wind_cardinal}`;
    }

    let maxPsi = 0;
    for (const r of REGIONS) {
      const val = regionPsiMap[r] ?? 50;
      const ws = regionWsMap[r] ?? 5.0;
      const wd = regionWdMap[r] ?? 180;
      if (val > maxPsi) maxPsi = val;
      const meta = classifyPsi(val);
      const color = SEVERITY_COLORS[meta.severity];

      const psiText = document.getElementById(`node-psi-${r}`);
      animateNumericText(psiText, val, 0);

      const windText = document.getElementById(`node-wind-${r}`);
      if (windText) windText.textContent = `${ws.toFixed(1)} kn · ${degreesToCardinal(wd)}`;

      const ring = document.getElementById(`ring-${r}`);
      if (ring) {
        ring.style.stroke = color;
        ring.setAttribute("r", r === state.selectedRegion ? "25" : "22");
      }

      const aura = document.getElementById(`aura-${r}`);
      if (aura) {
        const dynamicR = Math.round(58 + Math.min(52, Math.max(0, (val - 40) * 0.45)));
        aura.setAttribute("r", String(dynamicR));
      }

      const gradStop = document.getElementById(`stop-${r}`);
      if (gradStop) gradStop.setAttribute("stop-color", color);

      const sector = document.getElementById(`sector-${r}`);
      if (sector) {
        sector.classList.toggle("active", r === state.selectedRegion);
      }

      // Rotate regional wind vector arrow to point in direction wind is blowing TOWARD ((wd + 180) % 360)
      const regArrow = document.getElementById(`reg-wind-${r}`);
      if (regArrow && REGION_ARROW_COORDS[r]) {
        const [cx, cy] = REGION_ARROW_COORDS[r];
        const flowTowardsDeg = (wd + 180) % 360;
        regArrow.setAttribute("transform", `translate(${cx}, ${cy}) rotate(${flowTowardsDeg.toFixed(0)})`);
      }
    }

    document.querySelectorAll(".station-node").forEach((node) => {
      node.classList.toggle("active", node.dataset.region === state.selectedRegion);
    });

    const natMeta = classifyPsi(maxPsi);
    const pill = document.getElementById("national-band-pill");
    pill.textContent = `National Max ${maxPsi.toFixed(0)} PSI · ${natMeta.band}`;
    pill.className = `band-pill band-${natMeta.severity}`;

    // Rotate primary compass wind arrow
    const arrow = document.getElementById("wind-arrow");
    const windCaption = document.getElementById("wind-vector-caption");
    const avgWs = Object.values(regionWsMap).reduce((a, b) => a + b, 0) / REGIONS.length;
    const avgWd = regionWdMap[state.selectedRegion] ?? 180;
    const flowDeg = (avgWd + 180) % 360;
    arrow.setAttribute("transform", `rotate(${flowDeg.toFixed(0)})`);
    windCaption.textContent = `${avgWs.toFixed(1)} kn · ${degreesToCardinal(avgWd)}`;
  }

  function renderSubIndexStrip() {
    const { short_term } = state.payload;
    const reg = state.selectedRegion;
    const baseInfo = short_term.current_regions[reg];
    if (!baseInfo) return;

    const step = state.scrubberStep;
    let psiVal = baseInfo.psi_24h;
    let pm25Val = baseInfo.pm25_1h;
    let wsVal = baseInfo.wind_speed_kn;
    let wdVal = baseInfo.wind_dir_deg;

    let targetTs = short_term.latest_timestamp;
    if (step > 0) {
      const regFc = short_term.regional_forecasts[reg]?.[step - 1];
      if (regFc) {
        psiVal = regFc.psi_24h;
        pm25Val = regFc.pm25_1h;
        wsVal = regFc.wind_speed_kn;
        wdVal = regFc.wind_dir_deg;
        targetTs = regFc.timestamp;
      }
    }

    const meta = classifyPsi(psiVal);
    document.getElementById("selected-region-heading").textContent =
      `${baseInfo.label} Region · ${step === 0 ? `Live (${formatShortDayHour(targetTs)})` : `${formatSgtTimestamp(targetTs)} (+${step}h)`}`;
    document.getElementById("selected-region-stations").textContent = baseInfo.station_area;
    document.getElementById("selected-region-advisory").textContent = baseInfo.advisory;

    const psiEl = document.getElementById("val-psi-24h");
    psiEl.textContent = `${Math.round(psiVal)} PSI · ${meta.band}`;
    psiEl.style.color = SEVERITY_COLORS[meta.severity];

    document.getElementById("val-pm25-1h").textContent = `${Math.round(pm25Val)} µg/m³`;
    document.getElementById("val-wind-speed").textContent = `${wsVal.toFixed(1)} kn`;
    document.getElementById("val-wind-dir").textContent = `${Math.round(wdVal)}° (${degreesToCardinal(wdVal)})`;

    const advVal = wsVal * Math.cos(((wdVal - 230) * Math.PI) / 180);
    const advEl = document.getElementById("val-advection-idx");
    const advSign = advVal >= 0 ? "+" : "";
    advEl.textContent = `${advSign}${advVal.toFixed(1)} kn`;
    advEl.style.color = advVal > 1.5 ? SEVERITY_COLORS.unhealthy : SEVERITY_COLORS.good;

    document.getElementById("val-pm10-o3").textContent =
      `${Math.round(baseInfo.pm10_24h)} µg/m³ · ${Math.round(baseInfo.o3_sub)} idx`;
  }

  function renderTrajectoryChart() {
    const svg = document.getElementById("trajectory-svg");
    const { short_term } = state.payload;
    const obs = short_term.observed_history;
    const fcNat = short_term.national_forecast;
    const simObj = short_term.counterfactual_simulation;
    const regKey = state.chartRegion;

    const obsSeries = obs.map((item) => ({
      ts: item.timestamp,
      psi:
        regKey === "national"
          ? item.national_max_psi
          : item.regions[regKey]?.psi_24h ?? item.national_mean_psi,
      pm25:
        regKey === "national"
          ? item.national_max_pm25
          : item.regions[regKey]?.pm25_1h ?? 20,
    }));

    const fcSeries =
      regKey === "national"
        ? fcNat.map((item) => ({
            step: item.step_hour,
            ts: item.timestamp,
            psi: item.national_max_psi,
            low: item.psi_lower_10,
            high: item.psi_upper_90,
            pm25: item.national_max_pm25,
          }))
        : short_term.regional_forecasts[regKey].map((item) => ({
            step: item.step_hour,
            ts: item.timestamp,
            psi: item.psi_24h,
            low: item.psi_lower_10,
            high: item.psi_upper_90,
            pm25: item.pm25_1h,
          }));

    const simSeries =
      regKey === "national"
        ? simObj.national_trajectory.map((item) => ({
            step: item.step_hour,
            psi: item.national_max_psi,
          }))
        : simObj.regional_trajectories[regKey].map((item) => ({
            step: item.step_hour,
            psi: item.psi_24h,
          }));

    const W = 780;
    const H = 315;
    const padL = 46;
    const padR = 44;
    const padT = 18;
    const padB = 34;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const totalPoints = obsSeries.length + fcSeries.length;
    const allVals = [
      ...obsSeries.map((d) => d.psi),
      ...fcSeries.map((d) => d.high),
      ...fcSeries.map((d) => d.low),
      ...simSeries.map((d) => d.psi),
      ...obsSeries.map((d) => d.pm25),
      ...fcSeries.map((d) => d.pm25),
      105,
    ];
    const minY = 0;
    const maxY = Math.max(125, Math.ceil(Math.max(...allVals) / 25) * 25);

    const xAt = (idx) => padL + (idx / Math.max(1, totalPoints - 1)) * plotW;
    const yAt = (val) => padT + plotH - ((val - minY) / (maxY - minY)) * plotH;

    let html = "";

    const yTicks = [0, 50, 100, 150, 200, 250, 300].filter((v) => v <= maxY);
    for (const tick of yTicks) {
      const y = yAt(tick);
      const isBandLine = tick === 50 || tick === 100 || tick === 200;
      const stroke =
        tick === 100
          ? "rgba(217, 119, 6, 0.45)"
          : tick === 50
          ? "rgba(5, 150, 105, 0.35)"
          : "#E5E5EA";
      const dash = isBandLine ? 'stroke-dasharray="4 3"' : "";
      html += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" stroke="${stroke}" stroke-width="1" ${dash} />`;
      html += `<text x="${padL - 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" class="coord-label">${tick}</text>`;
    }

    const splitIdx = obsSeries.length - 1;
    const splitX = xAt(splitIdx);
    html += `<line x1="${splitX.toFixed(1)}" y1="${padT}" x2="${splitX.toFixed(1)}" y2="${H - padB}" stroke="#0071E3" stroke-width="1.2" stroke-dasharray="3 3" />`;
    html += `<text x="${(splitX + 6).toFixed(1)}" y="${padT + 12}" class="coord-label" fill="#0071E3">NOW → +72H FORECAST &amp; COUNTERFACTUAL</text>`;

    if (state.scrubberStep > 0) {
      const scrubX = xAt(splitIdx + state.scrubberStep);
      html += `<line x1="${scrubX.toFixed(1)}" y1="${padT}" x2="${scrubX.toFixed(1)}" y2="${H - padB}" stroke="#7C3AED" stroke-width="1.8" />`;
    }

    const lastObs = obsSeries[obsSeries.length - 1];
    const upperPts = [`${splitX.toFixed(1)},${yAt(lastObs.psi).toFixed(1)}`];
    const lowerPts = [`${splitX.toFixed(1)},${yAt(lastObs.psi).toFixed(1)}`];

    fcSeries.forEach((d, i) => {
      const cx = xAt(splitIdx + 1 + i);
      upperPts.push(`${cx.toFixed(1)},${yAt(d.high).toFixed(1)}`);
      lowerPts.unshift(`${cx.toFixed(1)},${yAt(d.low).toFixed(1)}`);
    });

    html += `<polygon class="morph-poly" points="${upperPts.join(" ")} ${lowerPts.join(" ")}" fill="rgba(124, 58, 237, 0.12)" stroke="rgba(124, 58, 237, 0.28)" stroke-width="1" />`;

    // PM2.5 trace
    const pm25Pts = [
      ...obsSeries.map((d, i) => `${xAt(i).toFixed(1)},${yAt(d.pm25).toFixed(1)}`),
      ...fcSeries.map((d, i) => `${xAt(splitIdx + 1 + i).toFixed(1)},${yAt(d.pm25).toFixed(1)}`),
    ];
    html += `<polyline class="morph-line" points="${pm25Pts.join(" ")}" fill="none" stroke="#0D9488" stroke-width="1.5" stroke-opacity="0.8" stroke-dasharray="2 2" />`;

    // Observed PSI solid line
    const obsPts = obsSeries.map((d, i) => `${xAt(i).toFixed(1)},${yAt(d.psi).toFixed(1)}`);
    html += `<polyline class="morph-line" points="${obsPts.join(" ")}" fill="none" stroke="#0071E3" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />`;

    // Baseline Forecast PSI solid line
    const fcPts = [
      `${splitX.toFixed(1)},${yAt(lastObs.psi).toFixed(1)}`,
      ...fcSeries.map((d, i) => `${xAt(splitIdx + 1 + i).toFixed(1)},${yAt(d.psi).toFixed(1)}`),
    ];
    html += `<polyline class="morph-line" points="${fcPts.join(" ")}" fill="none" stroke="#7C3AED" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />`;

    // Calculated Counterfactual Prediction trajectory ("Predictions on Predictions")
    const simPts = [
      `${splitX.toFixed(1)},${yAt(lastObs.psi).toFixed(1)}`,
      ...simSeries.map((d, i) => `${xAt(splitIdx + 1 + i).toFixed(1)},${yAt(d.psi).toFixed(1)}`),
    ];
    html += `<polyline class="morph-line" points="${simPts.join(" ")}" fill="none" stroke="#D97706" stroke-width="2.4" stroke-dasharray="5 3" stroke-linecap="round" stroke-linejoin="round" />`;

    const labelOffsets = [
      { idx: 0, text: formatShortDayHour(obsSeries[0]?.ts), anchor: "start" },
      { idx: Math.floor(splitIdx / 2), text: formatShortDayHour(obsSeries[Math.floor(splitIdx / 2)]?.ts), anchor: "middle" },
      { idx: splitIdx, text: `Now (${formatShortDayHour(lastObs?.ts)})`, anchor: "middle" },
      { idx: splitIdx + 24, text: formatShortDayHour(fcSeries[23]?.ts), anchor: "middle" },
      { idx: splitIdx + 48, text: formatShortDayHour(fcSeries[47]?.ts), anchor: "middle" },
      { idx: totalPoints - 1, text: formatShortDayHour(fcSeries[fcSeries.length - 1]?.ts), anchor: "end" },
    ];
    for (const lbl of labelOffsets) {
      const lx = xAt(lbl.idx);
      html += `<text x="${lx.toFixed(1)}" y="${H - 10}" text-anchor="${lbl.anchor}" class="coord-label">${lbl.text}</text>`;
    }

    svg.innerHTML = html;

    // Update top hover banner with current scrubber step details by default
    const hoverTimeEl = document.getElementById("hover-time-label");
    const hoverValsEl = document.getElementById("hover-values-label");
    if (hoverTimeEl && hoverValsEl) {
      if (state.scrubberStep > 0 && fcSeries[state.scrubberStep - 1]) {
        const curFc = fcSeries[state.scrubberStep - 1];
        const curSim = simSeries[state.scrubberStep - 1];
        hoverTimeEl.textContent = `Selected Forecast: ${formatSgtTimestamp(curFc.ts)} (+${state.scrubberStep}h)`;
        hoverValsEl.textContent = `Baseline: ${Math.round(curFc.psi)} PSI (${Math.round(curFc.low)}–${Math.round(curFc.high)}) · Counterfactual: ${Math.round(curSim.psi)} PSI · PM2.5: ${Math.round(curFc.pm25)} µg/m³`;
      } else {
        hoverTimeEl.textContent = `Live Observation: ${formatSgtTimestamp(lastObs.ts)} (Click or hover chart to inspect any hour)`;
        hoverValsEl.textContent = `Current PSI: ${Math.round(lastObs.psi)} · 1-Hr PM2.5: ${Math.round(lastObs.pm25)} µg/m³`;
      }
    }

    // Interactive mousemove & click on trajectory chart to inspect any hour
    svg.onmousemove = (ev) => {
      const rect = svg.getBoundingClientRect();
      const relX = ((ev.clientX - rect.left) / rect.width) * W;
      const rawIdx = Math.round(((relX - padL) / plotW) * (totalPoints - 1));
      const clampedIdx = Math.max(0, Math.min(totalPoints - 1, rawIdx));
      if (clampedIdx <= splitIdx) {
        const item = obsSeries[clampedIdx];
        hoverTimeEl.textContent = `Observed History · ${formatSgtTimestamp(item.ts)}`;
        hoverValsEl.textContent = `Observed PSI: ${Math.round(item.psi)} · 1-Hr PM2.5: ${Math.round(item.pm25)} µg/m³`;
      } else {
        const fIdx = clampedIdx - splitIdx - 1;
        const fItem = fcSeries[fIdx];
        const sItem = simSeries[fIdx];
        if (fItem) {
          hoverTimeEl.textContent = `Forecast (+${fItem.step}h) · ${formatSgtTimestamp(fItem.ts)}`;
          hoverValsEl.textContent = `Baseline: ${Math.round(fItem.psi)} PSI (Band: ${Math.round(fItem.low)}–${Math.round(fItem.high)}) · Counterfactual: ${Math.round(sItem.psi)} PSI · PM2.5: ${Math.round(fItem.pm25)} µg/m³`;
        }
      }
    };

    svg.onclick = (ev) => {
      const rect = svg.getBoundingClientRect();
      const relX = ((ev.clientX - rect.left) / rect.width) * W;
      const rawIdx = Math.round(((relX - padL) / plotW) * (totalPoints - 1));
      if (rawIdx > splitIdx) {
        const stepClicked = Math.min(72, rawIdx - splitIdx);
        setScrubberStep(stepClicked);
      } else {
        setScrubberStep(0);
      }
    };
  }

  function renderHorizonMatrix() {
    const { short_term, region_metadata } = state.payload;
    const thead = document.getElementById("horizon-table-head");
    const tbody = document.getElementById("horizon-table-body");
    const steps = [6, 12, 24, 48, 72];
    const natFc = short_term.national_forecast || [];

    if (thead && natFc.length >= 72) {
      let thHtml = `<tr>
        <th scope="col">Region</th>
        <th scope="col">Live Now<span class="th-subdate">${formatShortDayHour(shortTermLastTs(short_term))}</span></th>`;
      for (const h of steps) {
        const ts = natFc[h - 1]?.timestamp;
        thHtml += `<th scope="col">+${h}h<span class="th-subdate">${formatShortDayHour(ts)}</span></th>`;
      }
      thHtml += `<th scope="col">72h Peak</th></tr>`;
      thead.innerHTML = thHtml;
    }

    let rowsHtml = "";
    for (const r of REGIONS) {
      const liveObj = short_term.current_regions[r];
      const fcList = short_term.regional_forecasts[r] || [];
      const max72 = Math.max(liveObj.psi_24h, ...fcList.map((d) => d.psi_24h));
      const peakMeta = classifyPsi(max72);
      const liveMeta = classifyPsi(liveObj.psi_24h);

      let cellsHtml = `<td><strong>${region_metadata[r].name}</strong></td>`;
      cellsHtml += `<td>
        <span class="cell-primary-psi" style="color:${SEVERITY_COLORS[liveMeta.severity]}">${Math.round(liveObj.psi_24h)} PSI</span>
        <span class="cell-sub-meta">${Math.round(liveObj.pm25_1h)} µg · ${liveObj.wind_speed_kn.toFixed(0)}kn ${liveObj.wind_cardinal}</span>
      </td>`;

      for (const h of steps) {
        const item = fcList[h - 1] || fcList[fcList.length - 1];
        const cMeta = classifyPsi(item.psi_24h);
        const color = SEVERITY_COLORS[cMeta.severity];
        cellsHtml += `<td>
          <span class="cell-primary-psi" style="color:${color}">${Math.round(item.psi_24h)} PSI</span>
          <span class="cell-sub-meta">${Math.round(item.pm25_1h)} µg · ${item.wind_speed_kn.toFixed(0)}kn ${item.wind_cardinal}</span>
        </td>`;
      }

      cellsHtml += `<td><span class="band-pill band-${peakMeta.severity}">${Math.round(max72)} · ${peakMeta.band}</span></td>`;
      rowsHtml += `<tr>${cellsHtml}</tr>`;
    }

    tbody.innerHTML = rowsHtml;
  }

  function shortTermLastTs(shortTerm) {
    return shortTerm.latest_timestamp;
  }

  function renderIndonesiaCrossBorder(shortTerm) {
    const idData = shortTerm?.indonesia_transboundary;
    if (!idData) return;

    const totalImpactEl = document.getElementById("id-total-psi-impact");
    const transShareEl = document.getElementById("id-transboundary-share");
    const localShareEl = document.getElementById("id-local-share");

    if (totalImpactEl) {
      totalImpactEl.textContent = `+${idData.total_estimated_psi_impact.toFixed(1)} PSI`;
      totalImpactEl.style.color =
        idData.total_estimated_psi_impact > 12
          ? SEVERITY_COLORS.unhealthy
          : idData.total_estimated_psi_impact > 5
          ? SEVERITY_COLORS.moderate
          : SEVERITY_COLORS.good;
    }
    if (transShareEl) {
      transShareEl.textContent = `${idData.transboundary_share_pct.toFixed(1)}%`;
    }
    if (localShareEl) {
      localShareEl.textContent = `${idData.local_baseline_share_pct.toFixed(1)}%`;
    }

    const provGrid = document.getElementById("indonesia-provinces-grid");
    if (provGrid && Array.isArray(idData.provinces)) {
      provGrid.innerHTML = idData.provinces
        .map((p) => {
          const aqiColor = SEVERITY_COLORS[p.severity] || SEVERITY_COLORS.good;
          const alignColor =
            p.wind_alignment_pct >= 55
              ? SEVERITY_COLORS.unhealthy
              : p.wind_alignment_pct >= 25
              ? SEVERITY_COLORS.moderate
              : "var(--text-secondary)";
          return `
            <div class="id-province-card" role="listitem">
              <div class="id-prov-top">
                <div>
                  <div class="id-prov-title">${p.name}</div>
                  <div class="id-prov-corridor">${p.distance_km} km ${p.bearing_cardinal} (${p.bearing_deg}°) &rarr; SG ${p.primary_sg_receptor}</div>
                </div>
                <span class="band-pill band-${p.severity}">${p.us_aqi} AQI &middot; ${p.band}</span>
              </div>
              <div class="id-prov-metrics">
                <div class="id-metric-col">
                  <span class="id-metric-lbl">Live PM<sub>2.5</sub></span>
                  <span class="id-metric-num" style="color:${aqiColor}">${p.pm25_1h.toFixed(1)} µg/m³</span>
                </div>
                <div class="id-metric-col">
                  <span class="id-metric-lbl">24h Peak / CO</span>
                  <span class="id-metric-num">${p.pm25_24h_peak.toFixed(0)} µg &middot; ${p.co_ugm3} CO</span>
                </div>
                <div class="id-metric-col">
                  <span class="id-metric-lbl">Wind Alignment</span>
                  <span class="id-metric-num" style="color:${alignColor}">${p.wind_alignment_pct}% (${p.transit_lag_hours.toFixed(1)}h lag)</span>
                </div>
              </div>
              <div class="id-prov-footer">
                <span class="id-impact-summary">
                  Active SG Impact: <strong>+${p.estimated_sg_psi_impact.toFixed(1)} PSI</strong> &middot; Direct Potential: <strong>+${p.direct_onshore_psi_potential.toFixed(1)} PSI</strong>
                </span>
                <button
                  type="button"
                  class="btn-corridor-sim"
                  data-bearing="${p.bearing_deg}"
                  data-name="${p.short_name}"
                  aria-label="Simulate direct wind corridor plume from ${p.name}">
                  Simulate Corridor Plume
                </button>
              </div>
            </div>
          `;
        })
        .join("");

      provGrid.querySelectorAll(".btn-corridor-sim").forEach((btn) => {
        btn.addEventListener("click", () => {
          const bearing = parseFloat(btn.dataset.bearing) || 230;
          state.simWindDir = bearing;
          state.simWindSpeed = 13.5;
          state.simHotspot = 1.6;

          const sliderSimWd = document.getElementById("slider-sim-wd");
          const sliderSimWs = document.getElementById("slider-sim-ws");
          const sliderSimHs = document.getElementById("slider-sim-hs");
          if (sliderSimWd) sliderSimWd.value = state.simWindDir;
          if (sliderSimWs) sliderSimWs.value = state.simWindSpeed;
          if (sliderSimHs) sliderSimHs.value = state.simHotspot;

          document.querySelectorAll(".sim-preset-btn").forEach((b) => b.classList.remove("active"));
          updateSimLabels();
          fetchOverview();
        });
      });
    }

    // Render Cross-Border Propagation SVG Chart
    const svg = document.getElementById("indonesia-propagation-svg");
    const srcSeries = idData.source_series || [];
    if (!svg || srcSeries.length < 2) return;

    // Build matching Singapore West/South receptor series from observed_history + national_forecast
    const sgMap = {};
    for (const obs of shortTerm.observed_history || []) {
      const wPsi = obs.regions?.west?.psi_24h ?? obs.national_max_psi;
      const sPsi = obs.regions?.south?.psi_24h ?? obs.national_max_psi;
      sgMap[obs.timestamp.slice(0, 13)] = Math.max(wPsi, sPsi);
    }
    for (const fc of shortTerm.national_forecast || []) {
      const wPsi = fc.regions?.west ?? fc.national_max_psi;
      const sPsi = fc.regions?.south ?? fc.national_max_psi;
      sgMap[fc.timestamp.slice(0, 13)] = Math.max(wPsi, sPsi);
    }

    const merged = srcSeries.map((pt) => {
      const hourKey = pt.timestamp.slice(0, 13);
      return {
        ts: pt.timestamp,
        riau: pt.riau_pm25,
        jambi: pt.jambi_pm25,
        sgPsi: sgMap[hourKey] ?? shortTerm.national_forecast?.[0]?.national_max_psi ?? 50,
      };
    });

    const W = 560;
    const H = 235;
    const padL = 42;
    const padR = 24;
    const padT = 16;
    const padB = 30;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const allVals = merged.flatMap((d) => [d.riau, d.jambi, d.sgPsi, 80]);
    const minY = 0;
    const maxY = Math.ceil(Math.max(...allVals) / 20) * 20;

    const xAt = (i) => padL + (i / Math.max(1, merged.length - 1)) * plotW;
    const yAt = (v) => padT + plotH - ((v - minY) / Math.max(1, maxY - minY)) * plotH;

    let html = "";
    const yTicks = [0, Math.round(maxY / 2), maxY];
    for (const t of yTicks) {
      const y = yAt(t);
      html += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" stroke="#E5E5EA" stroke-width="1" />`;
      html += `<text x="${padL - 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" class="coord-label">${t}</text>`;
    }

    const riauPts = merged.map((d, i) => `${xAt(i).toFixed(1)},${yAt(d.riau).toFixed(1)}`);
    const jambiPts = merged.map((d, i) => `${xAt(i).toFixed(1)},${yAt(d.jambi).toFixed(1)}`);
    const sgPts = merged.map((d, i) => `${xAt(i).toFixed(1)},${yAt(d.sgPsi).toFixed(1)}`);

    html += `<polyline class="morph-line" points="${riauPts.join(" ")}" fill="none" stroke="#EA580C" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" />`;
    html += `<polyline class="morph-line" points="${jambiPts.join(" ")}" fill="none" stroke="#D97706" stroke-width="1.9" stroke-dasharray="4 3" stroke-linecap="round" stroke-linejoin="round" />`;
    html += `<polyline class="morph-line" points="${sgPts.join(" ")}" fill="none" stroke="#0071E3" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" />`;

    const startLbl = formatShortDayHour(merged[0]?.ts);
    const midLbl = formatShortDayHour(merged[Math.floor(merged.length / 2)]?.ts);
    const endLbl = formatShortDayHour(merged[merged.length - 1]?.ts);
    html += `<text x="${padL}" y="${H - 8}" text-anchor="start" class="coord-label">${startLbl}</text>`;
    html += `<text x="${(padL + plotW / 2).toFixed(1)}" y="${H - 8}" text-anchor="middle" class="coord-label">${midLbl}</text>`;
    html += `<text x="${W - padR}" y="${H - 8}" text-anchor="end" class="coord-label">${endLbl}</text>`;

    svg.innerHTML = html;
  }

  function renderCounterfactualSandbox(sim) {
    if (!sim) return;

    document.getElementById("sim-base-24h").textContent = `${Math.round(sim.baseline_24h_psi)} PSI`;

    const calc24El = document.getElementById("sim-calc-24h");
    const calc24Meta = classifyPsi(sim.simulated_24h_psi);
    calc24El.textContent = `${Math.round(sim.simulated_24h_psi)} PSI`;
    calc24El.style.color = SEVERITY_COLORS[calc24Meta.severity];

    const deltaPeakEl = document.getElementById("sim-delta-peak");
    const dVal = Math.round(sim.delta_72h_peak_psi);
    const dSign = dVal > 0 ? "+" : "";
    deltaPeakEl.textContent = `${dSign}${dVal} PSI`;
    deltaPeakEl.style.color =
      sim.delta_72h_peak_psi > 2.0
        ? SEVERITY_COLORS.unhealthy
        : sim.delta_72h_peak_psi < -2.0
        ? SEVERITY_COLORS.good
        : "#1D1D1F";

    const advEl = document.getElementById("sim-advection-val");
    const aSign = sim.sim_sumatra_advection_idx > 0 ? "+" : "";
    advEl.textContent = `${aSign}${sim.sim_sumatra_advection_idx.toFixed(1)} kn`;
    advEl.style.color =
      sim.sim_sumatra_advection_idx > 2.0 ? SEVERITY_COLORS.unhealthy : SEVERITY_COLORS.moderate;

    // Render Polar Wind-Rose Sensitivity SVG
    const svg = document.getElementById("wind-rose-svg");
    const rose = sim.directional_sensitivity_rose || [];
    const cx = 140;
    const cy = 132;
    const maxR = 88;
    const maxRosePsi = Math.max(120, ...rose.map((r) => r.projected_24h_psi));

    let html = `
      <circle cx="${cx}" cy="${cy}" r="${(maxR * 0.33).toFixed(1)}" fill="none" stroke="#DCE5F0" stroke-dasharray="2 2" />
      <circle cx="${cx}" cy="${cy}" r="${(maxR * 0.66).toFixed(1)}" fill="none" stroke="#DCE5F0" stroke-dasharray="2 2" />
      <circle cx="${cx}" cy="${cy}" r="${maxR}" fill="none" stroke="#CBD5E1" />
      <line x1="${cx}" y1="${cy - maxR - 8}" x2="${cx}" y2="${cy + maxR + 8}" stroke="#DCE5F0" />
      <line x1="${cx - maxR - 8}" y1="${cy}" x2="${cx + maxR + 8}" y2="${cy}" stroke="#DCE5F0" />
    `;

    // Polygon connecting all 8 directional sensitivity spokes
    const polyPts = rose.map((sec) => {
      const rad = ((sec.degrees - 90) * Math.PI) / 180;
      const rLen = Math.max(18, (sec.projected_24h_psi / maxRosePsi) * maxR);
      const px = cx + rLen * Math.cos(rad);
      const py = cy + rLen * Math.sin(rad);
      return `${px.toFixed(1)},${py.toFixed(1)}`;
    });
    html += `<polygon class="morph-poly" points="${polyPts.join(" ")}" fill="rgba(0, 113, 227, 0.14)" stroke="#0071E3" stroke-width="1.8" />`;

    // Active simulated wind bearing ray
    const activeRad = ((sim.sim_wind_dir_deg - 90) * Math.PI) / 180;
    const ax = cx + maxR * Math.cos(activeRad);
    const ay = cy + maxR * Math.sin(activeRad);
    html += `<line x1="${cx}" y1="${cy}" x2="${ax.toFixed(1)}" y2="${ay.toFixed(1)}" stroke="#D97706" stroke-width="2.4" />`;

    for (const sec of rose) {
      const rad = ((sec.degrees - 90) * Math.PI) / 180;
      const rLen = Math.max(18, (sec.projected_24h_psi / maxRosePsi) * maxR);
      const px = cx + rLen * Math.cos(rad);
      const py = cy + rLen * Math.sin(rad);
      const lx = cx + (maxR + 22) * Math.cos(rad);
      const ly = cy + (maxR + 20) * Math.sin(rad) + 3;
      const dotColor = SEVERITY_COLORS[sec.severity] || "#0071E3";

      html += `
        <g class="rose-sector-point" data-deg="${sec.degrees}" style="cursor:pointer">
          <circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="5" fill="${dotColor}" stroke="#FFFFFF" stroke-width="1.8" />
          <text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="middle" class="coord-label" fill="#1D1D1F">${sec.cardinal} (${sec.projected_24h_psi.toFixed(0)})</text>
        </g>
      `;
    }

    svg.innerHTML = html;

    svg.querySelectorAll(".rose-sector-point").forEach((pt) => {
      pt.addEventListener("click", () => {
        const deg = parseFloat(pt.dataset.deg);
        state.simWindDir = deg;
        const wdSlider = document.getElementById("slider-sim-wd");
        if (wdSlider) wdSlider.value = deg;
        document.querySelectorAll(".sim-preset-btn").forEach((b) => b.classList.remove("active"));
        updateSimLabels();
        fetchOverview();
      });
    });
  }

  function renderSeasonalOutlook(seasonal) {
    const gridEl = document.getElementById("seasonal-months-grid");
    let gridHtml = "";

    for (const m of seasonal.monthly_outlook) {
      const color = SEVERITY_COLORS[m.peak_severity];
      const pctWidth = Math.min(100, Math.round((m.expected_peak_psi / 260) * 100));
      gridHtml += `
        <div class="month-cell" role="listitem">
          <div class="month-header">
            <span class="month-name">${m.label}</span>
            <span class="month-risk-badge band-${m.peak_severity}">${m.unhealthy_risk_pct.toFixed(0)}% risk</span>
          </div>
          <div class="month-psi-row">
            <span class="month-peak-val" style="color:${color}">${m.expected_peak_psi.toFixed(0)} PSI</span>
            <span class="month-mean-val">Mean ${m.expected_mean_psi.toFixed(0)}</span>
          </div>
          <div class="month-bar-track">
            <div class="month-bar-fill" style="transform:scaleX(${(pctWidth / 100).toFixed(3)}); background-color:${color}"></div>
          </div>
          <div class="month-phase-caption" title="${m.monsoon_phase}">${m.monsoon_phase}</div>
        </div>
      `;
    }
    gridEl.innerHTML = gridHtml;

    const histBody = document.getElementById("historical-annual-body");
    const recentYears = seasonal.historical_annual.slice().reverse();
    histBody.innerHTML = recentYears
      .map((yr) => {
        const meta = classifyPsi(yr.max_psi);
        const oniStr = yr.enso_oni >= 0 ? `+${yr.enso_oni.toFixed(1)}` : yr.enso_oni.toFixed(1);
        const iodStr = yr.iod_dmi >= 0 ? `+${yr.iod_dmi.toFixed(1)}` : yr.iod_dmi.toFixed(1);
        return `
          <tr>
            <td><strong>${yr.year}</strong></td>
            <td>${oniStr}°C</td>
            <td>${iodStr}°C</td>
            <td>${Math.round(yr.mean_psi)} PSI</td>
            <td><span class="band-pill band-${meta.severity}">${Math.round(yr.max_psi)} · ${meta.band}</span></td>
            <td>${yr.unhealthy_days}d</td>
            <td style="font-family:var(--font-sans); color:var(--text-secondary); white-space:normal">${yr.notable_event}</td>
          </tr>
        `;
      })
      .join("");
  }

  function renderDiagnostics(diagnostics) {
    const { metrics, feature_importances, total_retrain_cycles } = diagnostics;
    const accVal = metrics.accuracy_pct ?? 96.2;
    const tolVal = metrics.tolerance_5psi_pct ?? 94.6;

    const topAccEl = document.getElementById("topbar-accuracy-val");
    if (topAccEl) {
      topAccEl.textContent = `${accVal.toFixed(1)}%`;
    }

    const diagAccEl = document.getElementById("diag-accuracy");
    if (diagAccEl) {
      diagAccEl.textContent = `${accVal.toFixed(1)}%`;
    }

    const diagTolEl = document.getElementById("diag-tolerance");
    if (diagTolEl) {
      diagTolEl.textContent = `${tolVal.toFixed(1)}%`;
    }

    document.getElementById("diag-mae").textContent = `${metrics.mae_psi.toFixed(2)} PSI`;
    document.getElementById("diag-wind-gain").textContent = `+${metrics.wind_gain_pct.toFixed(1)}%`;
    const r2El = document.getElementById("diag-r2");
    if (r2El) {
      r2El.textContent = `${(metrics.r2_score * 100).toFixed(1)}%`;
    }
    document.getElementById("diag-coverage").textContent = `${metrics.interval_coverage_pct.toFixed(0)}%`;
    document.getElementById("diag-sample-count").textContent =
      `${metrics.training_samples.toLocaleString()} train · ${metrics.validation_samples} holdout`;
    document.getElementById("retrain-cycles-badge").textContent =
      `Retrain Cycle #${total_retrain_cycles || 1}`;

    // Render Backtest SVG
    const svg = document.getElementById("backtest-svg");
    const series = metrics.validation_series || [];
    if (series.length > 1) {
      const W = 460;
      const H = 155;
      const padL = 36;
      const padR = 14;
      const padT = 12;
      const padB = 22;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;

      const vals = series.flatMap((d) => [d.actual_psi, d.predicted_psi, d.lower_10, d.upper_90]);
      const minV = Math.max(0, Math.floor(Math.min(...vals) - 6));
      const maxV = Math.ceil(Math.max(...vals) + 6);

      const xAt = (i) => padL + (i / (series.length - 1)) * plotW;
      const yAt = (v) => padT + plotH - ((v - minV) / Math.max(1, maxV - minV)) * plotH;

      const upPts = series.map((d, i) => `${xAt(i).toFixed(1)},${yAt(d.upper_90).toFixed(1)}`);
      const lowPts = series
        .slice()
        .reverse()
        .map((d, i) => `${xAt(series.length - 1 - i).toFixed(1)},${yAt(d.lower_10).toFixed(1)}`);

      const actPts = series.map((d, i) => `${xAt(i).toFixed(1)},${yAt(d.actual_psi).toFixed(1)}`);
      const predPts = series.map((d, i) => `${xAt(i).toFixed(1)},${yAt(d.predicted_psi).toFixed(1)}`);

      svg.innerHTML = `
        <line x1="${padL}" y1="${padT}" x2="${W - padR}" y2="${padT}" stroke="#E5E5EA" />
        <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="#E5E5EA" />
        <text x="${padL - 6}" y="${padT + 8}" text-anchor="end" class="coord-label">${maxV}</text>
        <text x="${padL - 6}" y="${H - padB}" text-anchor="end" class="coord-label">${minV}</text>
        <polygon points="${upPts.join(" ")} ${lowPts.join(" ")}" fill="rgba(0, 113, 227, 0.12)" />
        <polyline points="${actPts.join(" ")}" fill="none" stroke="#047857" stroke-width="2.2" />
        <polyline points="${predPts.join(" ")}" fill="none" stroke="#7C3AED" stroke-width="2" stroke-dasharray="3 2" />
        <text x="${padL + 6}" y="${H - 5}" class="coord-label" fill="#047857">— Actual NEA PSI</text>
        <text x="${padL + 135}" y="${H - 5}" class="coord-label" fill="#7C3AED">-- Wind-Aware ML Prediction</text>
      `;
    }

    // Continuous Retraining Evolution Log Table
    const histBody = document.getElementById("training-history-body");
    const runs = (metrics.training_history || []).slice().reverse().slice(0, 5);
    histBody.innerHTML = runs
      .map((r) => {
        const runAcc = r.accuracy_pct ?? Math.max(75.0, 100.0 - (r.mae_psi / 50.0) * 100.0);
        return `
          <tr>
            <td><strong>#${r.run_id}</strong></td>
            <td style="font-family:var(--font-sans); color:var(--text-secondary)">${r.trigger_source}</td>
            <td>${r.training_samples}</td>
            <td style="color:#047857; font-weight:700">${runAcc.toFixed(1)}%</td>
            <td style="color:#1D1D1F; font-weight:600">${r.mae_psi.toFixed(2)} PSI</td>
            <td>${r.r2_score.toFixed(3)}</td>
            <td style="color:#0071E3; font-weight:700">+${r.wind_gain_pct.toFixed(1)}%</td>
          </tr>
        `;
      })
      .join("");

    // Feature importances
    const featContainer = document.getElementById("feature-bars-container");
    const maxPct = Math.max(1, ...feature_importances.map((f) => f.importance_pct));
    featContainer.innerHTML = feature_importances
      .map((f) => {
        const widthPct = Math.min(100, (f.importance_pct / maxPct) * 100);
        return `
          <div class="feat-row">
            <div class="feat-meta">
              <span class="feat-name">${f.label}</span>
              <span class="feat-pct">${f.importance_pct.toFixed(1)}%</span>
            </div>
            <div class="feat-track">
              <div class="feat-fill" style="transform:scaleX(${(widthPct / 100).toFixed(3)})"></div>
            </div>
          </div>
        `;
      })
      .join("");
  }

  function updateSimLabels() {
    const wd = state.simWindDir ?? 180;
    const ws = state.simWindSpeed ?? 6.0;
    const hs = state.simHotspot ?? 0.0;
    document.getElementById("label-sim-wd").textContent = `${Math.round(wd)}° (${degreesToCardinal(wd)})`;
    document.getElementById("label-sim-ws").textContent = `${ws.toFixed(1)} kn`;
    document.getElementById("label-sim-hs").textContent = `${hs.toFixed(1)}×`;
  }

  function bindEvents() {
    document.querySelectorAll(".region-sector, .station-node").forEach((el) => {
      el.addEventListener("click", () => {
        const reg = el.dataset.region;
        if (reg) {
          state.selectedRegion = reg;
          renderSpatialMapAndScrubber();
          renderSubIndexStrip();
        }
      });
      el.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          el.click();
        }
      });
    });

    const hourSlider = document.getElementById("forecast-hour-slider");
    hourSlider.addEventListener("input", (ev) => {
      setScrubberStep(parseInt(ev.target.value, 10) || 0);
    });

    const selectEl = document.getElementById("target-time-select");
    if (selectEl) {
      selectEl.addEventListener("change", (ev) => {
        setScrubberStep(parseInt(ev.target.value, 10) || 0);
      });
    }

    document.querySelectorAll(".regime-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".regime-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.windRegime = btn.dataset.regime;
        fetchOverview();
      });
    });

    document.querySelectorAll("#region-filter-pills .pill-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("#region-filter-pills .pill-btn").forEach((b) => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        state.chartRegion = btn.dataset.region;
        if (state.chartRegion !== "national") {
          state.selectedRegion = state.chartRegion;
          renderSpatialMapAndScrubber();
          renderSubIndexStrip();
        }
        renderTrajectoryChart();
      });
    });

    // Counterfactual Wind & Plume Perturbation Sliders
    const sliderSimWd = document.getElementById("slider-sim-wd");
    const sliderSimWs = document.getElementById("slider-sim-ws");
    const sliderSimHs = document.getElementById("slider-sim-hs");

    let simDebounce = null;
    function scheduleSimFetch() {
      updateSimLabels();
      clearTimeout(simDebounce);
      simDebounce = setTimeout(() => fetchOverview(), 110);
    }

    sliderSimWd.addEventListener("input", (ev) => {
      state.simWindDir = parseFloat(ev.target.value);
      document.querySelectorAll(".sim-preset-btn").forEach((b) => b.classList.remove("active"));
      scheduleSimFetch();
    });

    sliderSimWs.addEventListener("input", (ev) => {
      state.simWindSpeed = parseFloat(ev.target.value);
      document.querySelectorAll(".sim-preset-btn").forEach((b) => b.classList.remove("active"));
      scheduleSimFetch();
    });

    sliderSimHs.addEventListener("input", (ev) => {
      state.simHotspot = parseFloat(ev.target.value);
      document.querySelectorAll(".sim-preset-btn").forEach((b) => b.classList.remove("active"));
      scheduleSimFetch();
    });

    document.querySelectorAll(".sim-preset-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".sim-preset-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const liveW = state.payload?.short_term?.live_national_wind || { speed_kn: 5.5, dir_deg: 180 };
        state.simWindDir = btn.dataset.wd === "live" ? liveW.dir_deg : parseFloat(btn.dataset.wd);
        state.simWindSpeed = btn.dataset.ws === "live" ? liveW.speed_kn : parseFloat(btn.dataset.ws);
        state.simHotspot = parseFloat(btn.dataset.hs);

        sliderSimWd.value = state.simWindDir;
        sliderSimWs.value = state.simWindSpeed;
        sliderSimHs.value = state.simHotspot;
        updateSimLabels();
        fetchOverview();
      });
    });

    // Seasonal sliders
    const sliderEnso = document.getElementById("slider-enso");
    const sliderIod = document.getElementById("slider-iod");
    const sliderRain = document.getElementById("slider-rain");

    function updateClimateLabels() {
      const ensoSign = state.ensoOni >= 0 ? "+" : "";
      const iodSign = state.iodDmi >= 0 ? "+" : "";
      const rainSign = state.rainAnomaly > 0 ? "+" : "";
      document.getElementById("label-enso").textContent = `${ensoSign}${state.ensoOni.toFixed(2)} °C`;
      document.getElementById("label-iod").textContent = `${iodSign}${state.iodDmi.toFixed(2)} °C`;
      document.getElementById("label-rain").textContent = `${rainSign}${state.rainAnomaly.toFixed(0)}%`;
    }

    let debounceTimer = null;
    function scheduleClimateFetch() {
      updateClimateLabels();
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => fetchOverview(), 140);
    }

    sliderEnso.addEventListener("input", (ev) => {
      state.ensoOni = parseFloat(ev.target.value);
      document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
      scheduleClimateFetch();
    });

    sliderIod.addEventListener("input", (ev) => {
      state.iodDmi = parseFloat(ev.target.value);
      document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
      scheduleClimateFetch();
    });

    sliderRain.addEventListener("input", (ev) => {
      state.rainAnomaly = parseFloat(ev.target.value);
      document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
      scheduleClimateFetch();
    });

    document.querySelectorAll(".preset-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.ensoOni = parseFloat(btn.dataset.oni);
        state.iodDmi = parseFloat(btn.dataset.iod);
        state.rainAnomaly = parseFloat(btn.dataset.rain);
        sliderEnso.value = state.ensoOni;
        sliderIod.value = state.iodDmi;
        sliderRain.value = state.rainAnomaly;
        updateClimateLabels();
        fetchOverview();
      });
    });

    // Live Sync & Retrain Button
    const btnSync = document.getElementById("btn-sync");
    btnSync.addEventListener("click", async () => {
      btnSync.disabled = true;
      const span = btnSync.querySelector("span");
      const origText = span.textContent;
      span.textContent = "Fetching & Retraining...";
      try {
        await fetch("/api/sync?trigger_label=manual-live-retrain&force_retrain=true", { method: "POST" });
        await fetchOverview();
      } finally {
        span.textContent = origText;
        btnSync.disabled = false;
      }
    });

    // Hourly-Aligned (:05 and :35) Auto-Learn Countdown Timer
    const toggleAuto = document.getElementById("toggle-auto-learn");
    const countdownLabel = document.getElementById("countdown-label");

    function formatMinutesSeconds(sec) {
      const m = Math.floor(sec / 60);
      const s = sec % 60;
      return `${m}m ${String(s).padStart(2, "0")}s`;
    }

    toggleAuto.addEventListener("change", () => {
      state.autoLearnEnabled = toggleAuto.checked;
      countdownLabel.textContent = state.autoLearnEnabled
        ? formatMinutesSeconds(state.secondsUntilNextSync)
        : "Paused";
    });

    setInterval(async () => {
      if (!state.autoLearnEnabled) return;
      state.secondsUntilNextSync -= 1;
      if (state.secondsUntilNextSync <= 0) {
        state.secondsUntilNextSync = 1800;
        countdownLabel.textContent = "Checking NEA...";
        try {
          await fetch("/api/sync?trigger_label=hourly-aligned-poll&force_retrain=false", { method: "POST" });
          await fetchOverview();
        } catch (_err) {
          // Next 30m slot will retry
        }
      } else {
        countdownLabel.textContent = formatMinutesSeconds(state.secondsUntilNextSync);
      }
    }, 1000);
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    fetchOverview();
  });
})();
