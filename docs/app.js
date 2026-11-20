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
    strategySelect: document.getElementById('strategySelect'),
    officialMemorySelect: document.getElementById('officialMemorySelect'),
    macroPanel: document.getElementById('macroPanel'),
    macroStatusBadge: document.getElementById('macroStatusBadge'),
    macroUpdatedBadge: document.getElementById('macroUpdatedBadge'),
    macroSummaryText: document.getElementById('macroSummaryText'),
    macroCards: document.getElementById('macroCards'),
    rebalanceUpdatedBadge: document.getElementById('rebalanceUpdatedBadge'),
    rebalanceTableBody: document.getElementById('rebalanceTableBody'),
    rebalanceEmpty: document.getElementById('rebalanceEmpty'),
    mainChartSubtitle: document.getElementById('mainChartSubtitle'),
    ratioChartSubtitle: document.getElementById('ratioChartSubtitle'),
    vixChartSubtitle: document.getElementById('vixChartSubtitle'),
    timeframeBadge: document.getElementById('timeframeBadge'),
    summaryUpdatedBadge: document.getElementById('summaryUpdatedBadge'),
    summaryTableBody: document.getElementById('summaryTableBody'),
    summaryEmpty: document.getElementById('summaryEmpty'),
    mainChartWrapper: document.getElementById('mainChartWrapper'),
    ratioChartWrapper: document.getElementById('ratioChartWrapper'),
    vixChartWrapper: document.getElementById('vixChartWrapper'),
    mainZoneOverlay: document.getElementById('mainZoneOverlay'),
  };

  const AUTO_REFRESH_CHECK_MS = 60 * 1000;
  const MIN_VISIBILITY_CHECK_MS = 20 * 1000;
  // Heartbeat limit: generated_at is rewritten by every run, even when prices do not change.
  const STALE_AFTER_MS = 45 * 60 * 1000;
  const TOGGLE_STORAGE_KEYS = {
    blueZones: 'dashboard-toggle-blue-zones',
    redZones: 'dashboard-toggle-red-zones',
    buyLines: 'dashboard-toggle-buy-lines',
    sellLines: 'dashboard-toggle-sell-lines',
  };

  const state = {
    manifest: null,
    summary: null,
    rebalance: null,
    macro: null,
    // The frozen observation protocol always opens on the ordinary391*.
    rebalanceStrategy: 'secondpass_391',
    officialMemory: localStorage.getItem('dashboard-official-memory') || 'friday',
    timeframe: localStorage.getItem('dashboard-timeframe') === 'W' ? 'W' : 'D',
    tickerData: null,
    vixData: null,
    activeSlug: null,
    publishedAt: null,
    lastCheckedAt: null,
    lastRefreshError: null,
    refreshInFlight: null,
    autoRefreshTimerId: null,
    tickerRequestSeq: 0,
    syncingRange: false,
    zoneRedrawQueued: false,
  };

  const BUY_LINE_KEYS = ['c1', 'c2', 'c3', 'c4'];
  const SELL_LINE_KEYS = ['v1', 'v2', 'v3', 'v4'];

  restoreUserPreferences();
  const charts = createCharts();
  attachEvents();
  loadInitialData();
  startAutoRefresh();

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

    const vixChart = LightweightCharts.createChart(document.getElementById('vixChart'), {
      ...commonOptions,
      width: dom.vixChartWrapper.clientWidth,
      height: dom.vixChartWrapper.clientHeight,
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

    const vixSeries = vixChart.addSeries(LightweightCharts.LineSeries, {
      color: '#f59e0b',
      lineWidth: 2,
      crosshairMarkerVisible: true,
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
      vixChart.resize(dom.vixChartWrapper.clientWidth, dom.vixChartWrapper.clientHeight);
      queueZoneRedraw();
    });
    resizeObserver.observe(dom.mainChartWrapper);
    resizeObserver.observe(dom.ratioChartWrapper);
    resizeObserver.observe(dom.vixChartWrapper);

    mainChart.timeScale().subscribeVisibleTimeRangeChange((range) => syncVisibleTimeRange(mainChart, range));
    ratioChart.timeScale().subscribeVisibleTimeRangeChange((range) => syncVisibleTimeRange(ratioChart, range));
    vixChart.timeScale().subscribeVisibleTimeRangeChange((range) => syncVisibleTimeRange(vixChart, range));
    mainChart.timeScale().subscribeVisibleLogicalRangeChange((range) => syncVisibleLogicalRange(mainChart, range));
    ratioChart.timeScale().subscribeVisibleLogicalRangeChange((range) => syncVisibleLogicalRange(ratioChart, range));
    vixChart.timeScale().subscribeVisibleLogicalRangeChange((range) => syncVisibleLogicalRange(vixChart, range));

    mainChart.timeScale().subscribeVisibleTimeRangeChange(queueZoneRedraw);
    mainChart.subscribeCrosshairMove(queueZoneRedraw);

    return { mainChart, ratioChart, vixChart, mainSeries, ratioSeries, vixSeries, lineSeries };
  }

  function restoreUserPreferences() {
    if (dom.timeframeSelect) {
      dom.timeframeSelect.value = state.timeframe;
    }
    const togglePairs = [
      [dom.toggleBlueZones, TOGGLE_STORAGE_KEYS.blueZones],
      [dom.toggleRedZones, TOGGLE_STORAGE_KEYS.redZones],
      [dom.toggleBuyLines, TOGGLE_STORAGE_KEYS.buyLines],
      [dom.toggleSellLines, TOGGLE_STORAGE_KEYS.sellLines],
    ];
    togglePairs.forEach(([input, storageKey]) => {
      if (!input) return;
      const saved = localStorage.getItem(storageKey);
      if (saved === '0' || saved === 'false') {
        input.checked = false;
      } else if (saved === '1' || saved === 'true') {
        input.checked = true;
      }
    });
  }

  function saveTogglePreference(storageKey, checked) {
    localStorage.setItem(storageKey, checked ? '1' : '0');
  }

  function attachEvents() {
    dom.tickerSelect.addEventListener('change', async (event) => {
      await setActiveTicker(event.target.value, true);
    });
    dom.timeframeSelect.addEventListener('change', () => {
      state.timeframe = dom.timeframeSelect.value === 'W' ? 'W' : 'D';
      localStorage.setItem('dashboard-timeframe', state.timeframe);
      renderActiveTicker({ preserveViewport: false });
    });
    dom.toggleBlueZones.addEventListener('change', () => {
      saveTogglePreference(TOGGLE_STORAGE_KEYS.blueZones, dom.toggleBlueZones.checked);
      renderActiveTicker({ preserveViewport: true });
    });
    dom.toggleRedZones.addEventListener('change', () => {
      saveTogglePreference(TOGGLE_STORAGE_KEYS.redZones, dom.toggleRedZones.checked);
      renderActiveTicker({ preserveViewport: true });
    });
    dom.toggleBuyLines.addEventListener('change', () => {
      saveTogglePreference(TOGGLE_STORAGE_KEYS.buyLines, dom.toggleBuyLines.checked);
      renderActiveTicker({ preserveViewport: true });
    });
    dom.toggleSellLines.addEventListener('change', () => {
      saveTogglePreference(TOGGLE_STORAGE_KEYS.sellLines, dom.toggleSellLines.checked);
      renderActiveTicker({ preserveViewport: true });
    });
    dom.resetZoomBtn.addEventListener('click', () => {
      fitAllCharts();
      queueZoneRedraw();
    });
    dom.officialMemorySelect.addEventListener('change', (event) => {
      state.officialMemory = event.target.value || 'friday';
      localStorage.setItem('dashboard-official-memory', state.officialMemory);
      renderRebalanceTable();
    });
    dom.strategySelect.addEventListener('change', (event) => {
      state.rebalanceStrategy = event.target.value || 'secondpass_391';
      localStorage.setItem('dashboard-rebalance-strategy', state.rebalanceStrategy);
      renderRebalanceTable();
    });
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        maybeCheckForUpdates('visibility');
      }
    });
    window.addEventListener('focus', () => {
      maybeCheckForUpdates('focus');
    });
  }

  function startAutoRefresh() {
    if (state.autoRefreshTimerId !== null) {
      clearInterval(state.autoRefreshTimerId);
    }
    state.autoRefreshTimerId = setInterval(() => {
      checkForUpdates('interval');
    }, AUTO_REFRESH_CHECK_MS);
  }

  function maybeCheckForUpdates(reason) {
    if (!state.manifest) return;
    const lastMs = state.lastCheckedAt ? Date.parse(state.lastCheckedAt) : 0;
    if (Number.isFinite(lastMs) && (Date.now() - lastMs) < MIN_VISIBILITY_CHECK_MS) {
      return;
    }
    checkForUpdates(reason);
  }

  async function loadInitialData() {
    try {
      setStatus('Carregant manifest i resum...');
      const initialBust = Date.now();
      const manifest = await fetchJson('./data/manifest.json', { bustKey: initialBust, captureHeaders: true });
      state.lastCheckedAt = new Date().toISOString();
      state.lastRefreshError = null;

      const dataBust = manifest?.generated_at || initialBust;
      const [summary, rebalance, macro] = await Promise.all([
        fetchJson('./data/summary.json', { bustKey: dataBust }),
        fetchOptionalJson('./data/rebalance.json', { bustKey: dataBust }),
        fetchOptionalJson('./data/macro.json', { bustKey: dataBust }),
      ]);
      state.manifest = manifest;
      state.summary = summary;
      state.rebalance = rebalance;
      state.macro = macro;
      populateTickerSelect();
      renderSummaryTable();
      renderMacroPanel();
      renderRebalanceTable();
      await loadVixData(dataBust);

      const slugFromUrl = new URLSearchParams(window.location.search).get('ticker');
      const storedSlug = localStorage.getItem('dashboard-active-slug');
      const firstSlug = manifest?.tickers?.[0]?.slug || null;
      const initialSlug = [slugFromUrl, storedSlug, firstSlug].find((slug) => slug && hasTicker(slug)) || null;

      if (!initialSlug) {
        setStatus('Encara no hi ha dades generades. Puja el projecte a GitHub i deixa que el workflow cree els JSON.');
        dom.summaryEmpty.hidden = false;
        return;
      }

      await setActiveTicker(initialSlug, false, { preserveViewport: false, bustKey: dataBust });
    } catch (error) {
      console.error(error);
      showError(`No s'han pogut carregar les dades inicials: ${error.message}`);
    }
  }

  async function checkForUpdates(reason = 'interval') {
    if (state.refreshInFlight) {
      return state.refreshInFlight;
    }

    const task = (async () => {
      try {
        const probeBust = Date.now();
        const nextManifest = await fetchJson('./data/manifest.json', { bustKey: probeBust, captureHeaders: true });
        state.lastCheckedAt = new Date().toISOString();
        state.lastRefreshError = null;

        const prevGen = state.manifest?.generated_at || null;
        const nextGen = nextManifest?.generated_at || null;
        const prevExec = state.manifest?.diagnostics?.executed_at || null;
        const nextExec = nextManifest?.diagnostics?.executed_at || null;
        const hasNewData = !state.manifest || nextGen !== prevGen || nextExec !== prevExec || !state.tickerData;

        if (!hasNewData) {
          state.manifest = nextManifest;
          updateStatusDiagnostic();
          return false;
        }

        const dataBust = nextGen || probeBust;
        const slugToLoad = [state.activeSlug, nextManifest?.tickers?.[0]?.slug]
          .find((slug) => slug && nextManifest?.tickers?.some((item) => item.slug === slug)) || null;

        const vixItem = nextManifest?.tickers?.find((item) => item.ticker === '^VIX' || item.name?.toUpperCase() === 'VIX');
        const [nextSummary, nextRebalance, nextMacro, nextVix, nextTickerData] = await Promise.all([
          fetchJson('./data/summary.json', { bustKey: dataBust }),
          fetchOptionalJson('./data/rebalance.json', { bustKey: dataBust }),
          fetchOptionalJson('./data/macro.json', { bustKey: dataBust }),
          vixItem?.slug ? fetchOptionalJson(`./data/tickers/${vixItem.slug}.json`, { bustKey: dataBust }) : Promise.resolve(null),
          slugToLoad ? fetchJson(`./data/tickers/${slugToLoad}.json`, { bustKey: dataBust }) : Promise.resolve(null),
        ]);

        state.manifest = nextManifest;
        state.summary = nextSummary;
        if (nextRebalance) state.rebalance = nextRebalance;
        if (nextMacro) state.macro = nextMacro;
        if (nextVix) state.vixData = nextVix;

        populateTickerSelect();
        if (slugToLoad) {
          state.activeSlug = slugToLoad;
          dom.tickerSelect.value = slugToLoad;
        }
        if (nextTickerData) {
          state.tickerData = nextTickerData;
        }

        renderSummaryTable();
        renderMacroPanel();
        renderRebalanceTable();
        renderActiveTicker({ preserveViewport: true });
        return true;
      } catch (error) {
        console.warn(`Error en auto-refresc (${reason}):`, error);
        state.lastCheckedAt = new Date().toISOString();
        state.lastRefreshError = error?.message || 'Error de connexió';
        updateStatusDiagnostic();
        return false;
      } finally {
        state.refreshInFlight = null;
      }
    })();

    state.refreshInFlight = task;
    return task;
  }

  async function loadVixData(bustKey) {
    const vixItem = state.manifest?.tickers?.find((item) => item.ticker === '^VIX' || item.name?.toUpperCase() === 'VIX');
    if (!vixItem?.slug) return;

    try {
      state.vixData = await fetchJson(`./data/tickers/${vixItem.slug}.json`, { bustKey });
    } catch (error) {
      console.warn('No s\'ha pogut carregar el VIX', error);
      state.vixData = null;
    }
  }

  function hasTicker(slug) {
    return Boolean(state.manifest?.tickers?.some((item) => item.slug === slug));
  }

  function populateTickerSelect() {
    const items = state.manifest?.tickers || [];
    const currentSlug = state.activeSlug;
    dom.tickerSelect.innerHTML = '';
    items.forEach((item) => {
      const option = document.createElement('option');
      option.value = item.slug;
      option.textContent = `${item.name} (${item.ticker})`;
      dom.tickerSelect.appendChild(option);
    });
    if (currentSlug && hasTicker(currentSlug)) {
      dom.tickerSelect.value = currentSlug;
    }
  }

  async function setActiveTicker(slug, updateUrl, options = {}) {
    if (!hasTicker(slug)) return;
    const preserveViewport = Boolean(options.preserveViewport);
    const bustKey = options.bustKey ?? state.manifest?.generated_at ?? Date.now();
    const requestSeq = ++state.tickerRequestSeq;

    state.activeSlug = slug;
    dom.tickerSelect.value = slug;
    localStorage.setItem('dashboard-active-slug', slug);
    if (updateUrl) {
      const url = new URL(window.location.href);
      url.searchParams.set('ticker', slug);
      history.replaceState({}, '', url);
    }

    setStatus('Carregant actiu seleccionat...');
    try {
      const nextTickerData = await fetchJson(`./data/tickers/${slug}.json`, { bustKey });
      if (requestSeq !== state.tickerRequestSeq) {
        return;
      }
      state.tickerData = nextTickerData;
      state.lastRefreshError = null;
      renderActiveTicker({ preserveViewport });
      renderSummaryTable();
    } catch (error) {
      if (requestSeq !== state.tickerRequestSeq) {
        return;
      }
      console.error(error);
      state.lastRefreshError = error?.message || 'Error carregant actiu';
      if (state.tickerData) {
        updateStatusDiagnostic();
      } else {
        showError(`No s'ha pogut carregar ${slug}: ${error.message}`);
      }
    }
  }

  function captureViewportRanges() {
    try {
      const mainRange = charts.mainChart.timeScale().getVisibleRange?.();
      if (mainRange?.from !== undefined && mainRange?.to !== undefined) {
        return mainRange;
      }
    } catch (error) {
      console.debug('No s ha pogut capturar el rang visible', error);
    }
    return null;
  }

  function restoreViewportRanges(range) {
    if (!range || range.from === undefined || range.to === undefined) return false;
    try {
      state.syncingRange = true;
      [charts.mainChart, charts.ratioChart, charts.vixChart].forEach((chart) => {
        chart.timeScale().setVisibleRange(range);
      });
      state.syncingRange = false;
      return true;
    } catch (error) {
      state.syncingRange = false;
      console.debug('No s ha pogut restaurar el rang visible', error);
      return false;
    }
  }

  function renderActiveTicker(options = {}) {
    const data = state.tickerData;
    if (!data) return;

    const preserveViewport = Boolean(options.preserveViewport);
    const savedRange = preserveViewport ? captureViewportRanges() : null;

    const timeframe = dom.timeframeSelect.value || 'D';
    const isDaily = timeframe === 'D';
    const tfKey = isDaily ? 'daily' : 'weekly';
    const tfLabel = isDaily ? 'Diari' : 'Setmanal';

    const mainCandles = data[tfKey]?.candles || [];
    const ratioCandles = data[tfKey]?.ratio || [];
    const vixCandles = state.vixData?.[tfKey]?.candles || [];
    charts.mainSeries.setData(mainCandles);
    charts.ratioSeries.setData(ratioCandles);
    charts.vixSeries.setData(candlesToLine(vixCandles));

    const showBuy = dom.toggleBuyLines.checked && isDaily;
    const showSell = dom.toggleSellLines.checked && isDaily;
    BUY_LINE_KEYS.forEach((key) => {
      charts.lineSeries[key].setData(showBuy ? (data.daily?.lines?.[key] || []) : []);
    });
    SELL_LINE_KEYS.forEach((key) => {
      charts.lineSeries[key].setData(showSell ? (data.daily?.lines?.[key] || []) : []);
    });

    if (!preserveViewport || !restoreViewportRanges(savedRange)) {
      fitAllCharts();
    }

    const meta = data.meta || {};
    const summary = data.summary || {};
    const tickerLabel = `${meta.name || '-'} (${meta.ticker || '-'})`;

    dom.metricName.textContent = tickerLabel;
    dom.metricClose.textContent = formatNumber(summary.close);
    dom.metricClose.title = summary.quote_date ? `Data real cotització: ${formatDateOnly(summary.quote_date)}` : '';
    dom.metricBuyLine.textContent = summary.line ? `${summary.line} · ${formatNumber(summary.buy)}` : '-';
    dom.metricPct.textContent = formatPct(summary.pct);
    dom.metricPct.classList.toggle('positive', Number(summary.pct) >= 0);
    dom.metricPct.classList.toggle('negative', Number(summary.pct) < 0);

    dom.mainChartSubtitle.textContent = tickerLabel;
    dom.ratioChartSubtitle.textContent = `Base 100 vs ${meta.denominator || '-'}`;
    dom.vixChartSubtitle.textContent = state.vixData?.meta?.ticker ? `${state.vixData.meta.name} (${state.vixData.meta.ticker})` : 'Volatilitat';
    dom.timeframeBadge.textContent = tfLabel;

    updateStatusDiagnostic();
    queueZoneRedraw();
  }

  function updateStatusDiagnostic() {
    const data = state.tickerData;
    const meta = data?.meta || {};
    const manifestEntry = (state.manifest?.tickers || []).find((item) => item.slug === state.activeSlug) || {};
    const diag = state.manifest?.diagnostics || {};
    const tickerLabel = meta.name ? `${meta.name} (${meta.ticker || '-'})` : (state.activeSlug || 'Dashboard');

    const executedAt = diag.executed_at || state.manifest?.generated_at || null;
    const generatedAt = state.manifest?.generated_at || diag.generated_at || meta.generated_at || null;
    const publishedAt = state.publishedAt || null;
    const checkedAt = state.lastCheckedAt || null;
    const quoteDate = manifestEntry.quote_date || data?.summary?.quote_date || meta.quote_date || getLastCandleDate(data);
    const dataUpdatedAt = manifestEntry.data_updated_at || generatedAt;
    const refreshStatus = manifestEntry.refresh_status || null;

    const quoteAgeText = dataUpdatedAt ? formatRelativeAge(dataUpdatedAt) : '';
    const quoteAgeSuffix = quoteAgeText ? ` (${quoteAgeText})` : '';
    const statusReason = refreshStatus === 'unchanged'
      ? 'sense canvis de preu'
      : refreshStatus === 'cached_outside_window'
        ? 'mercat tancat · conservada última dada'
        : refreshStatus === 'fallback_error'
          ? 'error font · conservada última dada'
          : refreshStatus === 'updated'
            ? 'nou preu'
            : '';

    const generatedMs = generatedAt ? Date.parse(generatedAt) : NaN;
    const heartbeatAgeMs = Number.isFinite(generatedMs) ? Date.now() - generatedMs : null;
    const isStale = heartbeatAgeMs !== null && heartbeatAgeMs > STALE_AFTER_MS;
    const generatedAge = generatedAt ? formatRelativeAge(generatedAt) : '';

    const parts = [
      tickerLabel,
      `Sessió de cotització: ${quoteDate ? formatDateOnly(quoteDate) : '-'}`,
      `Últim canvi de preu: ${dataUpdatedAt ? formatDateTime(dataUpdatedAt) : '-'}${quoteAgeSuffix}${statusReason ? ` [${statusReason}]` : ''}`,
      `Última execució: ${executedAt ? formatDateTime(executedAt) : '-'}${generatedAge ? ` (${generatedAge})` : ''}`,
      `Publicat: ${publishedAt ? formatDateTime(publishedAt) : '-'}`,
      `Comprovat: ${formatShortTime(checkedAt)}`,
    ];
    if (isStale) {
      parts.push(`AVÍS: dades generades fa més de 45 min (retard del sistema d'actualització)`);
    }
    setStatus(parts.join(' · '));
    dom.statusText.title = [
      `Actiu: ${tickerLabel}`,
      `Data real cotització: ${quoteDate ? formatDateOnly(quoteDate) : '-'}`,
      `Últim canvi de cotització detectat: ${dataUpdatedAt ? formatDateTime(dataUpdatedAt) : '-'}`,
      `Última execució workflow: ${executedAt ? formatDateTime(executedAt) : '-'}${diag.run_number ? ` (#${diag.run_number})` : ''}`,
      `Última generació JSON: ${generatedAt ? formatDateTime(generatedAt) : '-'}`,
      `Última publicació Pages (Last-Modified): ${publishedAt ? formatDateTime(publishedAt) : '-'}`,
      `Última comprovació navegador: ${checkedAt ? formatDateTime(checkedAt) : '-'}`,
      `Estat refresc: ${refreshStatus || '-'}`,
    ].join('\n');

    const badgeTime = dataUpdatedAt || generatedAt;
    const badgeQuote = quoteDate ? `Cot. ${formatDateOnly(quoteDate)} · ` : '';
    dom.summaryUpdatedBadge.textContent = badgeTime ? `${badgeQuote}Gen. ${formatDateTime(badgeTime)}` : 'Actualitzat: -';
    dom.summaryUpdatedBadge.title = dom.statusText.title;
  }

  function getLastCandleDate(data) {
    const candles = data?.daily?.candles || [];
    if (!candles.length) return null;
    return candles[candles.length - 1]?.time || null;
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
      const rowMeta = [
        row.quote_date ? `Cotització: ${formatDateOnly(row.quote_date)}` : null,
        row.data_updated_at ? `Actualitzat: ${formatDateTime(row.data_updated_at)}` : null,
        row.refresh_status ? `Estat: ${row.refresh_status}` : null,
      ].filter(Boolean).join(' · ');
      if (rowMeta) {
        tr.title = rowMeta;
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

  function renderMacroPanel() {
    const macro = state.macro;
    if (!dom.macroPanel) return;
    dom.macroPanel.hidden = !macro;
    if (!macro) return;

    const status = macro.status || {};
    const tone = status.tone || 'neutral';
    dom.macroStatusBadge.className = `macro-status-badge tone-${tone}`;
    dom.macroStatusBadge.textContent = status.label || '-';
    dom.macroSummaryText.textContent = status.summary || '-';

    const updatedAt = macro.meta?.generated_at || state.manifest?.generated_at;
    dom.macroUpdatedBadge.textContent = updatedAt ? `Actualitzat: ${formatDateTime(updatedAt)}` : '-';

    dom.macroCards.innerHTML = '';
    (macro.items || []).forEach((item) => {
      dom.macroCards.appendChild(renderMacroCard(item));
    });

    const activeCell = macro.matrix?.active_cell || status.key;
    document.querySelectorAll('.matrix-cell').forEach((cell) => {
      cell.classList.remove('is-active', 'tone-good', 'tone-warning', 'tone-danger');
      cell.removeAttribute('aria-current');
      cell.querySelector('.matrix-current-tag')?.remove();
      if (cell.dataset.cell === activeCell) {
        cell.classList.add('is-active', `tone-${tone}`);
        cell.setAttribute('aria-current', 'true');
        const tag = document.createElement('em');
        tag.className = 'matrix-current-tag';
        tag.textContent = 'Situació actual';
        cell.appendChild(tag);
      }
    });
  }

  function renderMacroCard(item) {
    const card = document.createElement('article');
    const stateKey = item.state || 'unknown';
    card.className = `macro-card state-${stateKey}`;

    const changeValue = item.kind === 'dxy' ? item.change_26w_pct : item.change_26w_pp;
    const changeText = item.kind === 'dxy' ? formatSignedPct(changeValue) : formatSignedPp(changeValue);
    const valueText = item.kind === 'dxy' ? formatNumber(item.value) : `${formatNumber(item.value)}%`;
    const meter = Number.isFinite(Number(item.meter)) ? clamp(Number(item.meter), 0, 100) : 0;

    card.innerHTML = `
      <div class="macro-card-top">
        <span class="macro-card-label">${escapeHtml(item.label || '-')}</span>
        <span class="macro-pill">${escapeHtml(item.state_label || '-')}</span>
      </div>
      <div class="macro-card-value">${valueText}</div>
      <div class="macro-card-change ${Number(changeValue) > 0 ? 'negative' : 'positive'}">${changeText} / 26s</div>
      <div class="macro-meter" aria-hidden="true">
        <div class="macro-meter-fill" style="width: ${meter}%"></div>
      </div>
    `;
    return card;
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
    const visibleRange = charts.mainChart.timeScale().getVisibleRange?.();
    const visiblePriceRange = getVisiblePriceRange(data.daily?.candles || [], visibleRange);

    zones.forEach((zone) => {
      const zoneLow = Number(zone.low);
      const zoneHigh = Number(zone.high);
      if (!Number.isFinite(zoneLow) || !Number.isFinite(zoneHigh)) {
        return;
      }
      if (visiblePriceRange && (zoneHigh < visiblePriceRange.min || zoneLow > visiblePriceRange.max)) {
        return;
      }

      const yTopRaw = charts.mainSeries.priceToCoordinate(zoneHigh);
      const yBottomRaw = charts.mainSeries.priceToCoordinate(zoneLow);

      if ([yTopRaw, yBottomRaw].some((value) => value === null || Number.isNaN(value))) {
        return;
      }

      const left = 0;
      const right = width;
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
      rect.title = `${zone.role || 'zone'} · ${zone.timeframes || 'D'} · ${formatNumber(zone.low)} - ${formatNumber(zone.high)}`;
      dom.mainZoneOverlay.appendChild(rect);
    });
  }

  function renderRebalanceTable() {
    const strategyOptions = state.rebalance?.strategy_options || [];
    const strategiesByKey = state.rebalance?.rows_by_strategy || {};
    const validStrategyValues = strategyOptions.map((item) => item.value);
    const defaultStrategy = state.rebalance?.meta?.default_strategy || 'secondpass_391';
    if (!validStrategyValues.includes(state.rebalanceStrategy) || !strategiesByKey[state.rebalanceStrategy]) {
      state.rebalanceStrategy = strategiesByKey[defaultStrategy] ? defaultStrategy : (validStrategyValues[0] || defaultStrategy);
    }
    if (dom.strategySelect) {
      if (strategyOptions.length) {
        dom.strategySelect.innerHTML = '';
        strategyOptions.forEach((item) => {
          const option = document.createElement('option');
          option.value = item.value;
          option.textContent = item.label;
          dom.strategySelect.appendChild(option);
        });
      }
      dom.strategySelect.value = state.rebalanceStrategy;
    }

    const strategyData = strategiesByKey[state.rebalanceStrategy] || state.rebalance || {};
    dom.strategySelect.title = strategyData.research_status || '';
    document.querySelector('.rebalance-table thead th:nth-child(7)').textContent = strategyData.execution ? 'Pesos efectius' : 'Pesos objectiu';
    const options = state.rebalance?.official_review_options || [];
    const validValues = options.map((item) => item.value);
    if (!validValues.includes(state.officialMemory)) {
      state.officialMemory = state.rebalance?.meta?.default_official_review || 'friday';
    }
    dom.officialMemorySelect.value = state.officialMemory;

    const rowsByOfficial = strategyData?.rows_by_official_review || state.rebalance?.rows_by_official_review || {};
    const rows = rowsByOfficial[state.officialMemory] || strategyData?.rows || state.rebalance?.rows || [];
    dom.rebalanceTableBody.innerHTML = '';
    dom.rebalanceEmpty.hidden = rows.length > 0;

    const updatedAt = state.rebalance?.meta?.generated_at || state.manifest?.generated_at;
    const selectedStrategy = strategyOptions.find((item) => item.value === state.rebalanceStrategy);
    const strategyLabel = selectedStrategy?.label || strategyData?.label || state.rebalance?.meta?.default_strategy_label || 'Estratègia';
    const selectedOption = options.find((item) => item.value === state.officialMemory);
    const memoryLabel = selectedOption?.label || (state.officialMemory === 'thursday' ? 'Dijous' : 'Divendres');
    const updatedText = updatedAt ? `Actualitzat: ${formatDateTime(updatedAt)}` : '-';
    dom.rebalanceUpdatedBadge.textContent = `${updatedText} · ${strategyLabel} · memòria ${memoryLabel.toLowerCase()}`;

    rows.forEach((row) => {
      if (!row?.available) return;
      const tr = document.createElement('tr');
      const needsAction = Boolean(row.action_required || row.ab3_pending_action);
      tr.classList.toggle('needs-action', needsAction);
      const protectionName = row.vp1_variant === 'VP1_2MA' ? 'VP1_2MA' : row.vp1_variant ? 'VP1 original' : 'AB3';
      const protection = row.ab3_active ? `${protectionName} actiu · ` : row.ab3_pending_action === 'proteccio' ? `${protectionName} pendent · ` : '';
      const finalRegime = `${protection}${row.final_regime || '-'}`;
      const weightKind = row.ab3_reentry_rule ? 'Pesos efectius' : 'Pesos objectiu';
      const pendingWeights = row.ab3_pending_weights ? ` Objectiu pendent: ${formatWeights(row.ab3_pending_weights)}.` : '';
      const weightsTitle = row.ordinary_weights ? `Règim ordinari ${row.ordinary_regime || row.final_regime}: ${formatWeights(row.ordinary_weights)}. ${weightKind}: ${formatWeights(row.weights)}.${pendingWeights}` : '';
      const dateTitle = row.ab3_reentry_rule ? 'Tancament diari confirmat; les noves ordres són per al pròxim tancament comú.' : row.is_provisional ? 'Estimació actual; no és una nova confirmació setmanal.' : `Data de revisió. Execució al tancament comú següent${row.execution_date ? `: ${row.execution_date}` : ' disponible'}.`;
      const pending = row.ab3_pending_action ? `Pendent al pròxim tancament comú: ${row.ab3_pending_order || row.ab3_pending_action}` : '';
      const actionText = pending && !row.action_required ? pending : [row.action || '-', pending].filter(Boolean).join(' · ');
      tr.innerHTML = `
        <td>${escapeHtml(row.label || '-')}</td>
        <td title="${escapeHtml(dateTitle)}">${escapeHtml(row.date || '-')}</td>
        <td>${formatScore(row.score)}</td>
        <td>${escapeHtml(row.base_regime || '-')}</td>
        <td>${row.crash_fast_armed ? 'Sí' : 'No'}</td>
        <td>${escapeHtml(finalRegime)}</td>
        <td title="${escapeHtml(weightsTitle)}">${formatWeights(row.weights)}</td>
        <td title="${escapeHtml(actionText)}" class="${needsAction ? 'negative' : 'positive'}">${escapeHtml(actionText)}</td>
      `;
      dom.rebalanceTableBody.appendChild(tr);
    });

    if (!dom.rebalanceTableBody.children.length) {
      dom.rebalanceEmpty.hidden = false;
    }
  }

  function syncVisibleTimeRange(sourceChart, range) {
    if (state.syncingRange || !range || range.from === undefined || range.to === undefined) return;
    state.syncingRange = true;
    [charts.mainChart, charts.ratioChart, charts.vixChart].forEach((chart) => {
      if (chart === sourceChart) return;
      try {
        chart.timeScale().setVisibleRange(range);
      } catch (error) {
        console.debug('No s ha pogut sincronitzar per data', error);
      }
    });
    state.syncingRange = false;
    queueZoneRedraw();
  }

  function syncVisibleLogicalRange(sourceChart, range) {
    if (state.syncingRange || !range || range.from === undefined || range.to === undefined) return;
    const visibleTimeRange = sourceChart.timeScale().getVisibleRange?.();
    if (visibleTimeRange?.from !== undefined && visibleTimeRange?.to !== undefined) {
      syncVisibleTimeRange(sourceChart, visibleTimeRange);
      return;
    }

    state.syncingRange = true;
    [charts.mainChart, charts.ratioChart, charts.vixChart].forEach((chart) => {
      if (chart === sourceChart) return;
      try {
        chart.timeScale().setVisibleLogicalRange(range);
      } catch (error) {
        console.debug('No s ha pogut sincronitzar per zoom', error);
      }
    });
    state.syncingRange = false;
    queueZoneRedraw();
  }

  function fitAllCharts() {
    state.syncingRange = true;
    [charts.mainChart, charts.vixChart, charts.ratioChart].forEach((chart) => {
      chart.timeScale().fitContent();
    });
    state.syncingRange = false;
  }

  function getVisiblePriceRange(candles, visibleRange) {
    const visibleCandles = (candles || []).filter((bar) => {
      if (!bar?.time) return false;
      if (!visibleRange?.from || !visibleRange?.to) return true;
      return compareChartTimes(bar.time, visibleRange.from) >= 0 && compareChartTimes(bar.time, visibleRange.to) <= 0;
    });
    if (!visibleCandles.length) return null;

    let min = Infinity;
    let max = -Infinity;
    visibleCandles.forEach((bar) => {
      min = Math.min(min, Number(bar.low));
      max = Math.max(max, Number(bar.high));
    });
    if (!Number.isFinite(min) || !Number.isFinite(max) || min >= max) {
      return null;
    }

    const padding = (max - min) * 0.08;
    return {
      min: min - padding,
      max: max + padding,
    };
  }

  function compareChartTimes(a, b) {
    return chartTimeToStamp(a) - chartTimeToStamp(b);
  }

  function chartTimeToStamp(value) {
    if (typeof value === 'number') return value;
    if (typeof value === 'string') return Date.parse(`${value}T00:00:00Z`) / 1000;
    if (value && typeof value === 'object') {
      return Date.UTC(Number(value.year), Number(value.month) - 1, Number(value.day)) / 1000;
    }
    return NaN;
  }

  function candlesToLine(candles) {
    return (candles || [])
      .filter((bar) => bar?.time && bar.close !== null && bar.close !== undefined && !Number.isNaN(Number(bar.close)))
      .map((bar) => ({
        time: bar.time,
        value: Number(bar.close),
      }));
  }

  function withCacheBust(url, bustKey) {
    if (bustKey === undefined || bustKey === null || bustKey === '') {
      return url;
    }
    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}v=${encodeURIComponent(String(bustKey))}`;
  }

  async function fetchJson(url, options = {}) {
    const targetUrl = withCacheBust(url, options.bustKey);
    const response = await fetch(targetUrl, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText} en ${url}`);
    }
    if (options.captureHeaders) {
      const lastModified = response.headers.get('last-modified');
      if (lastModified) {
        const parsed = new Date(lastModified);
        if (!Number.isNaN(parsed.getTime())) {
          state.publishedAt = parsed.toISOString();
        }
      }
    }
    return response.json();
  }

  async function fetchOptionalJson(url, options = {}) {
    try {
      return await fetchJson(url, options);
    } catch (error) {
      console.warn(`No s'ha pogut carregar ${url}`, error);
      return null;
    }
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

  function formatSignedPct(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return '-';
    }
    const number = Number(value);
    const sign = number > 0 ? '+' : '';
    return `${sign}${new Intl.NumberFormat('ca-ES', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(number)}%`;
  }

  function formatSignedPp(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return '-';
    }
    const number = Number(value);
    const sign = number > 0 ? '+' : '';
    return `${sign}${new Intl.NumberFormat('ca-ES', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(number)} pp`;
  }

  function formatScore(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return '-';
    }
    return new Intl.NumberFormat('ca-ES', {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    }).format(Number(value));
  }

  function formatWeights(weights) {
    if (!weights) return '-';
    return [
      `World ${formatWeight(weights.world)}`,
      `Nasdaq ${formatWeight(weights.nasdaq)}`,
      `Monetari ${formatWeight(weights.cash)}`,
    ].join(' / ');
  }

  function formatWeight(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return '-';
    }
    return `${new Intl.NumberFormat('ca-ES', {
      maximumFractionDigits: 1,
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

  function formatShortTime(value) {
    if (!value) return '-';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat('ca-ES', {
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  }

  function formatDateOnly(value) {
    if (!value) return '-';
    const match = String(value).match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (match) {
      return `${match[3]}/${match[2]}/${match[1]}`;
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat('ca-ES', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    }).format(date);
  }

  function formatRelativeAge(value) {
    if (!value) return '';
    const ms = Date.parse(value);
    if (!Number.isFinite(ms)) return '';
    const diffMin = Math.max(0, Math.round((Date.now() - ms) / 60000));
    if (diffMin < 1) return 'fa <1 min';
    if (diffMin < 60) return `fa ${diffMin} min`;
    const hours = Math.floor(diffMin / 60);
    const mins = diffMin % 60;
    if (hours < 24) return mins ? `fa ${hours} h ${mins} min` : `fa ${hours} h`;
    const days = Math.floor(hours / 24);
    const remHours = hours % 24;
    return remHours ? `fa ${days} d ${remHours} h` : `fa ${days} d`;
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
