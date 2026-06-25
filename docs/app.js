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

  const state = {
    manifest: null,
    summary: null,
    rebalance: null,
    macro: null,
    rebalanceStrategy: localStorage.getItem('dashboard-rebalance-strategy') || 'secondpass_391_airbag_s3',
    officialMemory: localStorage.getItem('dashboard-official-memory') || 'friday',
    tickerData: null,
    vixData: null,
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
      fitAllCharts();
      queueZoneRedraw();
    });
    dom.officialMemorySelect.addEventListener('change', (event) => {
      state.officialMemory = event.target.value || 'friday';
      localStorage.setItem('dashboard-official-memory', state.officialMemory);
      renderRebalanceTable();
    });
    dom.strategySelect.addEventListener('change', (event) => {
      state.rebalanceStrategy = event.target.value || 'secondpass_391_airbag_s3';
      localStorage.setItem('dashboard-rebalance-strategy', state.rebalanceStrategy);
      renderRebalanceTable();
    });
  }

  async function loadInitialData() {
    try {
      setStatus('Carregant manifest i resum...');
      const [manifest, summary, rebalance, macro] = await Promise.all([
        fetchJson('./data/manifest.json'),
        fetchJson('./data/summary.json'),
        fetchOptionalJson('./data/rebalance.json'),
        fetchOptionalJson('./data/macro.json'),
      ]);
      state.manifest = manifest;
      state.summary = summary;
      state.rebalance = rebalance;
      state.macro = macro;
      populateTickerSelect();
      renderSummaryTable();
      renderMacroPanel();
      renderRebalanceTable();
      await loadVixData();

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

  async function loadVixData() {
    const vixItem = state.manifest?.tickers?.find((item) => item.ticker === '^VIX' || item.name?.toUpperCase() === 'VIX');
    if (!vixItem?.slug) return;

    try {
      state.vixData = await fetchJson(`./data/tickers/${vixItem.slug}.json`);
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

    fitAllCharts();

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
    dom.vixChartSubtitle.textContent = state.vixData?.meta?.ticker ? `${state.vixData.meta.name} (${state.vixData.meta.ticker})` : 'Volatilitat';
    dom.timeframeBadge.textContent = tfLabel;

    const updatedAt = state.manifest?.generated_at || meta.generated_at;
    const updatedText = updatedAt ? `Actualitzat: ${formatDateTime(updatedAt)}` : 'Actualitzat: -';
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
    const defaultStrategy = state.rebalance?.meta?.default_strategy || 'secondpass_391_airbag_s3';
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
      tr.classList.toggle('needs-action', Boolean(row.action_required));
      tr.innerHTML = `
        <td>${escapeHtml(row.label || '-')}</td>
        <td>${escapeHtml(row.date || '-')}</td>
        <td>${formatScore(row.score)}</td>
        <td>${escapeHtml(row.base_regime || '-')}</td>
        <td>${row.crash_fast_armed ? 'Sí' : 'No'}</td>
        <td>${escapeHtml(row.final_regime || '-')}</td>
        <td>${formatWeights(row.weights)}</td>
        <td class="${row.action_required ? 'negative' : 'positive'}">${escapeHtml(row.action || '-')}</td>
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

  async function fetchJson(url) {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText} en ${url}`);
    }
    return response.json();
  }

  async function fetchOptionalJson(url) {
    try {
      return await fetchJson(url);
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
      maximumFractionDigits: 0,
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
