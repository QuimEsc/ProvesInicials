(() => {
  const dom = {
    tickerSelect: document.getElementById('tickerSelect'),
    timeframeSelect: document.getElementById('timeframeSelect'),
    toggleBlueZones: document.getElementById('toggleBlueZones'),
    toggleRedZones: document.getElementById('toggleRedZones'),
    toggleBuyLines: document.getElementById('toggleBuyLines'),
    toggleSellLines: document.getElementById('toggleSellLines'),
    resetZoomBtn: document.getElementById('resetZoomBtn'),
    statusText: document.getElementById('statusText'),
    metricName: document.getElementById('metricName'),
    metricClose: document.getElementById('metricClose'),
    metricBuyLine: document.getElementById('metricBuyLine'),
    metricPct: document.getElementById('metricPct'),
    mainChartSubtitle: document.getElementById('mainChartSubtitle'),
    ratioChartSubtitle: document.getElementById('ratioChartSubtitle'),
    timeframeBadge: document.getElementById('timeframeBadge'),
    summaryUpdatedBadge: document.getElementById('summaryUpdatedBadge'),
    summaryTableBody: document.getElementById('summaryTableBody'),
    summaryEmpty: document.getElementById('summaryEmpty'),
    mainChartWrapper: document.getElementById('mainChartWrapper'),
    ratioChartWrapper: document.getElementById('ratioChartWrapper'),
    mainZoneOverlay: document.getElementById('mainZoneOverlay'),
  };

  const state = {
    manifest: null,
    summary: null,
    tickerData: null,
    activeSlug: null,
    syncingRange: false,
    zoneRedrawQueued: false,
  };

  const BUY_LINE_KEYS = ['c1', 'c2', 'c3', 'c4'];
  const SELL_LINE_KEYS = ['v1', 'v2', 'v3', 'v4'];

  const charts = createCharts();
  attachEvents();
  loadInitialData();

  function createCharts() {
    const commonOptions = {
      layout: {
        background: { color: '#131722' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
        horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: 'rgba(197, 203, 206, 0.2)',
      },
      timeScale: {
        borderColor: 'rgba(197, 203, 206, 0.2)',
        timeVisible: false,
        secondsVisible: false,
      },
      localization: {
        locale: 'ca-ES',
      },
    };

    const mainChart = LightweightCharts.createChart(document.getElementById('mainChart'), {
      ...commonOptions,
      width: dom.mainChartWrapper.clientWidth,
      height: dom.mainChartWrapper.clientHeight,
      handleScroll: true,
      handleScale: true,
    });

    const ratioChart = LightweightCharts.createChart(document.getElementById('ratioChart'), {
      ...commonOptions,
      width: dom.ratioChartWrapper.clientWidth,
      height: dom.ratioChartWrapper.clientHeight,
      handleScroll: true,
      handleScale: true,
    });

    const mainSeries = mainChart.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      priceLineVisible: false,
      lastValueVisible: true,
    });

    const ratioSeries = ratioChart.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: '#60a5fa',
      downColor: '#f87171',
      borderVisible: false,
      wickUpColor: '#60a5fa',
      wickDownColor: '#f87171',
      priceLineVisible: false,
      lastValueVisible: true,
    });

    const lineColors = {
      c1: '#93c5fd',
      c2: '#60a5fa',
      c3: '#3b82f6',
      c4: '#1d4ed8',
      v1: '#fca5a5',
      v2: '#f87171',
      v3: '#ef4444',
      v4: '#b91c1c',
    };

    const lineSeries = {};
    [...BUY_LINE_KEYS, ...SELL_LINE_KEYS].forEach((key) => {
      lineSeries[key] = mainChart.addSeries(LightweightCharts.LineSeries, {
        color: lineColors[key],
        lineWidth: 2,
        crosshairMarkerVisible: false,
        priceLineVisible: false,
        lastValueVisible: false,
      });
    });

    const resizeObserver = new ResizeObserver(() => {
      mainChart.resize(dom.mainChartWrapper.clientWidth, dom.mainChartWrapper.clientHeight);
      ratioChart.resize(dom.ratioChartWrapper.clientWidth, dom.ratioChartWrapper.clientHeight);
      queueZoneRedraw();
    });
    resizeObserver.observe(dom.mainChartWrapper);
    resizeObserver.observe(dom.ratioChartWrapper);

    mainChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (state.syncingRange || !range) return;
      state.syncingRange = true;
      ratioChart.timeScale().setVisibleLogicalRange(range);
      state.syncingRange = false;
      queueZoneRedraw();
    });

    ratioChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (state.syncingRange || !range) return;
      state.syncingRange = true;
      mainChart.timeScale().setVisibleLogicalRange(range);
      state.syncingRange = false;
      queueZoneRedraw();
    });

    mainChart.timeScale().subscribeVisibleTimeRangeChange(queueZoneRedraw);
    mainChart.subscribeCrosshairMove(queueZoneRedraw);

    return { mainChart, ratioChart, mainSeries, ratioSeries, lineSeries };
  }

  function attachEvents() {
    dom.tickerSelect.addEventListener('change', async (event) => {
      await setActiveTicker(event.target.value, true);
    });
    dom.timeframeSelect.addEventListener('change', renderActiveTicker);
    dom.toggleBlueZones.addEventListener('change', renderActiveTicker);
    dom.toggleRedZones.addEventListener('change', renderActiveTicker);
    dom.toggleBuyLines.addEventListener('change', renderActiveTicker);
    dom.toggleSellLines.addEventListener('change', renderActiveTicker);
    dom.resetZoomBtn.addEventListener('click', () => {
      charts.mainChart.timeScale().fitContent();
      charts.ratioChart.timeScale().fitContent();
      queueZoneRedraw();
    });
  }

  async function loadInitialData() {
    try {
      setStatus('Carregant manifest i resum...');
      const [manifest, summary] = await Promise.all([
        fetchJson('./data/manifest.json'),
        fetchJson('./data/summary.json'),
      ]);
      state.manifest = manifest;
      state.summary = summary;
      populateTickerSelect();
      renderSummaryTable();

      const slugFromUrl = new URLSearchParams(window.location.search).get('ticker');
      const storedSlug = localStorage.getItem('dashboard-active-slug');
      const firstSlug = manifest?.tickers?.[0]?.slug || null;
      const initialSlug = [slugFromUrl, storedSlug, firstSlug].find((slug) => slug && hasTicker(slug)) || null;

      if (!initialSlug) {
        setStatus('Encara no hi ha dades generades. Puja el projecte a GitHub i deixa que el workflow cree els JSON.');
        dom.summaryEmpty.hidden = false;
        return;
      }

      await setActiveTicker(initialSlug, false);
    } catch (error) {
      console.error(error);
      showError(`No s'han pogut carregar les dades inicials: ${error.message}`);
    }
  }

  function hasTicker(slug) {
    return Boolean(state.manifest?.tickers?.some((item) => item.slug === slug));
  }

  function populateTickerSelect() {
    const items = state.manifest?.tickers || [];
    dom.tickerSelect.innerHTML = '';
    items.forEach((item) => {
      const option = document.createElement('option');
      option.value = item.slug;
      option.textContent = `${item.name} (${item.ticker})`;
      dom.tickerSelect.appendChild(option);
    });
  }

  async function setActiveTicker(slug, updateUrl) {
    if (!hasTicker(slug)) return;
    state.activeSlug = slug;
    dom.tickerSelect.value = slug;
    localStorage.setItem('dashboard-active-slug', slug);
    if (updateUrl) {
      const url = new URL(window.location.href);
      url.searchParams.set('ticker', slug);
      history.replaceState({}, '', url);
    }

    setStatus('Carregant actiu seleccionat...');
    state.tickerData = await fetchJson(`./data/tickers/${slug}.json`);
    renderActiveTicker();
    renderSummaryTable();
  }

  function renderActiveTicker() {
    const data = state.tickerData;
    if (!data) return;

    const timeframe = dom.timeframeSelect.value || 'D';
    const isDaily = timeframe === 'D';
    const tfKey = isDaily ? 'daily' : 'weekly';
    const tfLabel = isDaily ? 'Diari' : 'Setmanal';

    const mainCandles = data[tfKey]?.candles || [];
    const ratioCandles = data[tfKey]?.ratio || [];
    charts.mainSeries.setData(mainCandles);
    charts.ratioSeries.setData(ratioCandles);

    const showBuy = dom.toggleBuyLines.checked && isDaily;
    const showSell = dom.toggleSellLines.checked && isDaily;
    BUY_LINE_KEYS.forEach((key) => {
      charts.lineSeries[key].setData(showBuy ? (data.daily?.lines?.[key] || []) : []);
    });
    SELL_LINE_KEYS.forEach((key) => {
      charts.lineSeries[key].setData(showSell ? (data.daily?.lines?.[key] || []) : []);
    });

    charts.mainChart.timeScale().fitContent();
    charts.ratioChart.timeScale().fitContent();

    const meta = data.meta || {};
    const summary = data.summary || {};
    const tickerLabel = `${meta.name || '-'} (${meta.ticker || '-'})`;

    dom.metricName.textContent = tickerLabel;
    dom.metricClose.textContent = formatNumber(summary.close);
    dom.metricBuyLine.textContent = summary.line ? `${summary.line} · ${formatNumber(summary.buy)}` : '-';
    dom.metricPct.textContent = formatPct(summary.pct);
    dom.metricPct.classList.toggle('positive', Number(summary.pct) >= 0);
    dom.metricPct.classList.toggle('negative', Number(summary.pct) < 0);

    dom.mainChartSubtitle.textContent = tickerLabel;
    dom.ratioChartSubtitle.textContent = `Base 100 vs ${meta.denominator || '-'}`;
    dom.timeframeBadge.textContent = tfLabel;

    const updatedText = meta.generated_at ? `Actualitzat: ${formatDateTime(meta.generated_at)}` : 'Actualitzat: -';
    const cadenceText = meta.refresh_interval_minutes ? ` · refresc cada ${meta.refresh_interval_minutes} min` : '';
    setStatus(`${tickerLabel} · ${updatedText}${cadenceText}`);
    dom.summaryUpdatedBadge.textContent = updatedText;

    queueZoneRedraw();
  }

  function renderSummaryTable() {
    const rows = state.summary?.rows || [];
    dom.summaryTableBody.innerHTML = '';
    dom.summaryEmpty.hidden = rows.length > 0;

    rows.forEach((row) => {
      const tr = document.createElement('tr');
      if (row.slug === state.activeSlug) {
        tr.classList.add('is-active');
      }
      tr.innerHTML = `
        <td>${escapeHtml(row.name || '-')}</td>
        <td>${escapeHtml(row.ticker || '-')}</td>
        <td>${formatNumber(row.close)}</td>
        <td>${escapeHtml(row.line || '-')}</td>
        <td>${formatNumber(row.buy)}</td>
        <td class="${Number(row.pct) >= 0 ? 'positive' : 'negative'}">${formatPct(row.pct)}</td>
      `;
      tr.addEventListener('click', async () => {
        await setActiveTicker(row.slug, true);
      });
      dom.summaryTableBody.appendChild(tr);
    });
  }

  function queueZoneRedraw() {
    if (state.zoneRedrawQueued) return;
    state.zoneRedrawQueued = true;
    requestAnimationFrame(() => {
      state.zoneRedrawQueued = false;
      drawZones();
    });
  }

  function drawZones() {
    dom.mainZoneOverlay.innerHTML = '';
    const data = state.tickerData;
    if (!data || dom.timeframeSelect.value !== 'D') {
      return;
    }

    const zones = [];
    if (dom.toggleBlueZones.checked) {
      zones.push(...(data.daily?.zones?.blue || []));
    }
    if (dom.toggleRedZones.checked) {
      zones.push(...(data.daily?.zones?.red || []));
    }
    if (!zones.length) {
      return;
    }

    const overlayHeight = Math.max(0, dom.mainChartWrapper.clientHeight - charts.mainChart.timeScale().height());
    dom.mainZoneOverlay.style.height = `${overlayHeight}px`;
    dom.mainZoneOverlay.style.bottom = `${charts.mainChart.timeScale().height()}px`;

    const width = dom.mainChartWrapper.clientWidth;
    const height = overlayHeight;

    zones.forEach((zone) => {
      const x1 = charts.mainChart.timeScale().timeToCoordinate(zone.start);
      const x2 = charts.mainChart.timeScale().timeToCoordinate(zone.end);
      const yTopRaw = charts.mainSeries.priceToCoordinate(Number(zone.high));
      const yBottomRaw = charts.mainSeries.priceToCoordinate(Number(zone.low));

      if ([x1, x2, yTopRaw, yBottomRaw].some((value) => value === null || Number.isNaN(value))) {
        return;
      }

      const left = clamp(Math.min(x1, x2), 0, width);
      const right = clamp(Math.max(x1, x2), 0, width);
      const top = clamp(Math.min(yTopRaw, yBottomRaw), 0, height);
      const bottom = clamp(Math.max(yTopRaw, yBottomRaw), 0, height);
      const rectWidth = right - left;
      const rectHeight = bottom - top;

      if (rectWidth <= 1 || rectHeight <= 1) {
        return;
      }

      const rect = document.createElement('div');
      rect.className = `zone-rect ${zone.active_now ? 'active-now' : ''}`;
      rect.style.left = `${left}px`;
      rect.style.top = `${top}px`;
      rect.style.width = `${rectWidth}px`;
      rect.style.height = `${rectHeight}px`;
      rect.style.background = zone.color || 'rgba(255,255,255,0.12)';
      rect.title = `${zone.role || 'zone'} · ${zone.start} → ${zone.end}`;
      dom.mainZoneOverlay.appendChild(rect);
    });
  }

  async function fetchJson(url) {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText} en ${url}`);
    }
    return response.json();
  }

  function setStatus(message) {
    dom.statusText.textContent = message;
  }

  function showError(message) {
    dom.statusText.innerHTML = `<span class="error-state">${escapeHtml(message)}</span>`;
  }

  function formatNumber(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return '-';
    }
    return new Intl.NumberFormat('ca-ES', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(Number(value));
  }

  function formatPct(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return '-';
    }
    return `${new Intl.NumberFormat('ca-ES', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(Number(value))}%`;
  }

  function formatDateTime(value) {
    if (!value) return '-';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return new Intl.DateTimeFormat('ca-ES', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function escapeHtml(text) {
    return String(text)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#039;');
  }
})();