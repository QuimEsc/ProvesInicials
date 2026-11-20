# Versions tècniques abans de l'inici prospectiu

## v2 ·08/11/2026

Es conserva v1 al commit5dab23ad31ff33fa2a760b247c19ae627c87ab4f i l'etiqueta
391-freeze-2026-11-08. Identitat, manifest i paràmetres previs a `archive/v1/`.

Canvi: el registre rebutja una execució retrospectiva si la captura que aporta
la decisió anterior es va generar després del tancament d'execució. Es desa
la vintage amb qualitat `late_information_after_execution_close`, sense crear
una sessió de rendibilitat aparentment prospectiva.

Impacte: **cap canvi de senyals, paràmetres, pesos, costos, models o resultats
històrics**. Cap sessió prospectiva existent: encara és08/11/2026; la captura
conservada és del06/11 i està etiquetada com a coneguda després delbacktest.

La identitat i l'etiqueta v2 tenen el seu propi SHA i hora efectiva. La prova
nova utilitza només dates i preus històrics2020 dins d'un directori temporal,
mai dades futures inexistents. S'hereta la mateixa captura inicialv1 perquè
els estats de model no han canviat; la cadena nova usa identitatv2.

## v3 · 20/11/2026

Es conserva v2 a `freeze/archive/v2/`. Canvi tècnic exclusiu d'actualització,
transport i diagnòstic:

- `.github/workflows/update-pages-data.yml`: fraccionament de `schedule` en blocs de 4 hores (`7,37`), `cancel-in-progress: false`, `fetch-depth: 1`, caché `pip`, metadades d'execució i `git pull --rebase` abans de `push`.
- `data_manager.py`: registre de telemetria de descàrrega (`get_last_fetch_info`) mantenint intacte el fallback al CSV local en cas d'error.
- `build_pages_data.py`: salvaguarda post-tancament de mercat (`_needs_post_close_refresh`) perquè les execucions retardades capturen el tancament oficial diari fora de la finestra `07:00–23:00`, refresc automàtic diari de tipus BCE/FED si fa >18 h, i metadades diagnòstiques (`diagnostics`, `quote_date`, `data_updated_at`, `last_checked_at`, `refresh_status`).
- `docs/app.js`: sondeig periòdic automàtic cada 60 s (`visibilitychange` / `focus` + `setInterval`), cache-busting amb `generated_at`, control de concurrència/fora d'ordre, preservació de ticker, timeframe, toggles i zoom, i visualització de les marques temporals d'execució, generació, publicació, comprovació i cotització real.

Impacte: **cap canvi en fórmules, paràmetres, senyals, pesos, costos, models financers ni sessions prospectives arxivades**.

### v3.1 · 20/11/2026

Supervisió i claredat d'estat, sense cap canvi de models: `docs/app.js` separa
«Última execució» (batec del sistema) d'«Últim canvi de preu» i avisa si el batec
supera 45 min. Nous fitxers no congelats: `monitor_freshness.py`,
`.github/workflows/watch-data-freshness.yml` i `tests/test_monitor_freshness.py`.

### v3.2 · 20/11/2026

Ajust als criteris operatius reals (objectiu 30 min amb tolerància de fins a
60 min en horari de mercat `07:00–23:00 Europe/Madrid`, dl-dv), sense canvis
de models financers ni estratègies:

- `.github/workflows/update-pages-data.yml`: simplificació del `cron` a
  `7,37 6-22 * * 1-5` i `12 3 * * *`, més `workflow_dispatch.inputs.refresh_rates`
  per defecte a `false`.
- `monitor_freshness.py` i `.github/workflows/watch-data-freshness.yml`: avaluació
  en horari de mercat respecte de l'última consulta efectiva a la font, auto-reparació
  silenciosa a partir de `45 min`, tolerància fins a `60 min`, alerta persistent
  només a partir de `120 min` i silenci complet fora d'horari o amb preus sense canvis.
- `docs/app.js`: llindar d'avís a `60 min` només dins d'horari de mercat i
  visualització explícita d'«Última consulta font».
