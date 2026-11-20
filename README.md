# ProvesInicials — Dashboard de Borsa i Manteniment de l'Automatització

- **Web pública:** https://quimesc.github.io/ProvesInicials/docs/?ticker=nasdaq
- **Repositori:** https://github.com/QuimEsc/ProvesInicials

Aquest document centralitza el funcionament operatiu del sistema d'actualització automàtica de dades, el registre de caducitat i renovació del token de GitHub per a `cron-job.org`, les comprovacions de salut i la resolució d'incidències.

> [!NOTE]
> Per a la documentació científica, regles congelades (`391` / `391*` / `VP1`), registre prospectiu i protocol de validació anual (novembre 2027), consulta exclusivament els fitxers existents dins [`freeze/`](freeze/README.md) (incloent [`freeze/research_handoff.md`](freeze/research_handoff.md) i [`freeze/corrections.md`](freeze/corrections.md)) i [`tracking/README.md`](tracking/README.md). **No s'han d'alterar fórmules financeres, estratègies de cartera ni registres prospectius.**

---

## 1. Funcionament de l'Actualització Automàtica

### Circuit complet de dades

```text
cron-job.org
  ──(HTTP POST cada 30 min, dl–dv 07:07–22:37 Europe/Madrid)──▶ GitHub REST API
  ──(workflow_dispatch)──▶ GitHub Actions (.github/workflows/update-pages-data.yml)
  ──(build_pages_data.py + data_manager.py + tests + commit)──▶ JSON a docs/data/ + CSV a dades_historiques/
  ──(push a main)──▶ GitHub Pages (pages build and deployment)
  ──(sondeig cada 60 s de docs/app.js)──▶ Refresc automàtic al navegador sense recarregar
```

### Configuració operativa de la tasca externa

| Camp | Valor configurat |
| :--- | :--- |
| **Servei extern** | `cron-job.org` |
| **Nom de la tasca** | `Actualitzar Dashboard Borsa` |
| **Repositori** | `QuimEsc/ProvesInicials` |
| **Workflow activat** | `.github/workflows/update-pages-data.yml` (`Update dashboard data`) |
| **Endpoint** | `https://api.github.com/repos/QuimEsc/ProvesInicials/actions/workflows/update-pages-data.yml/dispatches` |
| **Mètode HTTP** | `POST` |
| **Cos JSON** | `{"ref":"main"}` |
| **Horari previst** | Dilluns a divendres, cada 30 minuts, minuts **`07`** i **`37`**, hores **`07–22`** incloses |
| **Zona horària** | `Europe/Madrid` (gestiona automàticament canvis d'estiu **CEST** i hivern **CET**) |
| **Primera execució diària** | `07:07 Europe/Madrid` |
| **Última execució diària** | `22:37 Europe/Madrid` |
| **Objectiu de frescor** | Actualització habitual cada **30 minuts**, acceptant retards ocasionals de fins a **60 minuts** dins l'horari de mercat |

### Mecanismes complementaris a GitHub Actions

GitHub Actions manté dues capes pròpies de suport i supervisió sobre la branca `main`:

1. **Cron de suport natiu (`.github/workflows/update-pages-data.yml`):**
   - `7,37 5-21 * * 1-5`: franja UTC fixa que cobreix tant l'estiu (`05:07–20:37 UTC` = `07:07–22:37 CEST`) com l'hivern (`06:07–21:37 UTC` = `07:07–22:37 CET`) com a reserva passiva si `cron-job.org` fallara puntualment.
   - `12 3 * * *`: execució nocturna diària amb `--refresh-rates` per actualitzar tipus d'interés (BCE/FED) i actius `24h` (`BTC-USD`, `GC=F`, `BZ=F`).
2. **Watchdog de frescor (`.github/workflows/watch-data-freshness.yml` + `monitor_freshness.py`):**
   - Programat a `22,52 5-21 * * 1-5` i activable per `workflow_dispatch`.
   - Avalua la frescor dins l'horari operatiu respecte de l'última consulta efectiva a la font: tolera fins a `60 min` sense alarma, llança una auto-reparació silenciosa per `workflow_dispatch` si supera `45 min`, i només obri o edita un únic Issue (`Actualització de dades endarrerida (>60 min en horari de mercat)`) si el retard persisteix `>= 120 min` dins l'horari de mercat. Fora d'horari o amb preus sense canvis no genera falses alarmes.
3. **Control de concurrència i anti-duplicats:**
   - `concurrency: group: update-dashboard-data` amb `cancel-in-progress: false` evita tallar execucions en curs.
   - El pas inicial `Skip duplicate or out-of-window backup run` omet automàticament execucions redundants en pocs segons (sense descarregar ni recomputar):
     - Per a `schedule` (excepte `12 3 * * *`): si ja s'ha generat `docs/data/manifest.json` fa **< 20 min** o si l'hora queda fora de `07:00–23:00 Europe/Madrid` (franges límit del canvi d'hora estiu/hivern).
     - Per a `workflow_dispatch` (quan `refresh_rates` no és `true`): si ja s'ha generat `manifest.json` fa **< 12 min**.

### Verificació contra el codi real i estat actual

- **Coincidència verificada:** la finestra `07:07–22:37 Europe/Madrid` (dl–dv) coincideix amb `build_pages_data.py` (`MARKET_REFRESH_START_HOUR = 7`, `MARKET_REFRESH_END_HOUR = 23`, `REFRESH_TIMEZONE = "Europe/Madrid"`).
- **Matisos observats al codi real:**
  - En enviar `{"ref":"main"}`, `inputs.refresh_rates` queda en `false`; `build_pages_data.py` igualment refresca les sèries BCE/FED quan han passat `>= 18 h` des de `rates_updated_at` (a més del cron nocturn `12 3 * * *`).
  - Un `workflow_dispatch` enviat fora de `07:00–23:00 Europe/Madrid` **no** és blocat pel filtre anti-duplicats: s'executa conservant en `cached_fresh` els actius `market` (si el tancament anterior ja es va capturar) i refrescant els actius `24h`.
- **Evidència inicial de connexió i estat de verificació:**
  - S'ha verificat que `cron-job.org` dispara correctament `.github/workflows/update-pages-data.yml` amb una petició HTTP `POST` autenticada: **Execució `#2633`** (`Run ID: 44834439800`, trigger `workflow_dispatch`, resultat **`Success`**, duració **3 minuts i 22 segons**, desplegat per `pages build and deployment #2643`).
  - **Nota:** aquella prova manual inicial (`2026-11-20T23:07:38Z`) acredita la connectivitat i permisos del token; la continuïtat sostinguda de les execucions programades en dies laborables s'ha de comprovar a l'historial operatiu de cron-job.org i GitHub Actions.

---

## 2. Caducitat del Token GitHub

**TOKEN GITHUB — CRON-JOB.ORG**

- **Nom identificatiu:** `cron-job-provesinicials`
- **Finalitat:** permetre que `cron-job.org` active les actualitzacions de GitHub Actions.
- **Tipus:** Fine-grained personal access token.
- **Resource owner:** `QuimEsc`
- **Repositori autoritzat:** `QuimEsc/ProvesInicials`
- **Permís necessari:** `Actions: Read and write`
- **Caducitat:** **18 de febrer de 2027, a les 22:49:49 UTC**
- **Hora de Madrid:** **18 de febrer de 2027, a les 23:49:49 CET**
- **Ubicació d'ús:** capçalera `Authorization` de la tasca `Actualitzar Dashboard Borsa` dins de `cron-job.org`.

> [!IMPORTANT]
> **IMPORTANT: Abans del 18/02/2027 cal renovar el token de GitHub i actualitzar-lo a cron-job.org. Si caduca, les peticions autenticades deixaran de funcionar i l'actualització externa quedarà interrompuda.**

*(El valor real del token no s'emmagatzema mai en aquest fitxer ni en cap arxiu del repositori.)*

---

## 3. Procediment Exacte de Renovació del Token

1. Entrar a:
   https://github.com/settings/personal-access-tokens
2. Localitzar el token `cron-job-provesinicials` i comprovar les opcions de regeneració o renovació que GitHub oferisca.
3. Renovar-lo o generar-ne un de nou, mantenint els permisos mínims:
   - **Resource owner:** `QuimEsc`.
   - **Repository access:** `Only select repositories` → `QuimEsc/ProvesInicials`.
   - **Repository permissions:** `Actions` — **`Read and write`** (i `Metadata: Read-only`, obligatori per defecte).
   - Sense permisos addicionals innecessaris.
4. Copiar el token nou de manera segura (GitHub només el mostra una vegada en pantalla).
5. Entrar a:
   https://cron-job.org
6. Localitzar la tasca `Actualitzar Dashboard Borsa`.
7. Obrir la configuració de la petició HTTP (*Advanced → Headers*).
8. Substituir exclusivament el valor de la capçalera:
   ```text
   Authorization: Bearer TOKEN_NOU
   ```
9. Guardar la tasca.
10. Executar una prova manual amb **Test run**.
11. Confirmar que GitHub accepta la petició amb una resposta HTTP satisfactòria (`204 No Content`).
12. Verificar que apareix una nova execució `workflow_dispatch` en:
    https://github.com/QuimEsc/ProvesInicials/actions
13. Comprovar que el workflow acaba correctament, tenint en compte que el filtre anti-duplicats (`Skip duplicate or out-of-window backup run`) pot ometre la regeneració en ~25 s si ja existeix una actualització recent (< 12 min); això també es considera `Success`.
14. Revocar el token antic a GitHub **únicament després** de confirmar que el nou funciona.
15. Actualitzar la nova data de caducitat documentada en aquest `README.md` i renovar els recordatoris del calendari (i, si escau, les dates de [`recordatoris_renovacio_token_cron_job.ics`](recordatoris_renovacio_token_cron_job.ics)).

### Errors habituals durant el Test run de renovació

- **HTTP `401` (Unauthorized):** token absent, mal copiat (espais sobrants o falta el prefix `Bearer `), caducat o revocat. Torna a enganxar `Bearer TOKEN_NOU` sense espais ni salts de línia.
- **HTTP `403` (Forbidden):** el token no té `Actions: Read and write`, no inclou el repositori `QuimEsc/ProvesInicials` o el compte té alguna restricció d'accés/PAT. Revisa permisos i abast del token a GitHub.
- **HTTP `404` (Not Found):** URL incorrecta, repositori privat sense accés assignat al token o nom de fitxer del workflow incorrecte (`update-pages-data.yml`). Comprova que el token té seleccionat explícitament `QuimEsc/ProvesInicials` i verifica l'endpoint exacte.

---

## 4. Recordatoris de Renovació (`.ics`)

Com que el repositori no disposa d'accés directe a cap compte extern de calendari, s'ha generat el fitxer estàndard [`recordatoris_renovacio_token_cron_job.ics`](recordatoris_renovacio_token_cron_job.ics) a l'arrel del projecte amb els tres avisos en zona horària `Europe/Madrid`:

| Data i hora (`Europe/Madrid`) | Avís | Títol de l'esdeveniment |
| :--- | :--- | :--- |
| **04/02/2027 a les 09:00** | Primer avís (14 dies abans) | `Renovar token GitHub — cron-job.org — ProvesInicials` |
| **11/02/2027 a les 09:00** | Segon avís (7 dies abans) | `Renovar token GitHub — cron-job.org — ProvesInicials` |
| **17/02/2027 a les 09:00** | Avís urgent (caduca l'endemà) | `Renovar token GitHub — cron-job.org — ProvesInicials` |

> [!WARNING]
> **Estat dels recordatoris:** el fitxer `.ics` està desat al repositori, però **els recordatoris no quedaran actius fins que importes aquest fitxer manualment** al teu calendari.

### Com importar `recordatoris_renovacio_token_cron_job.ics`

- **Google Calendar (web):**
  1. Descarrega [`recordatoris_renovacio_token_cron_job.ics`](recordatoris_renovacio_token_cron_job.ics) des del repositori (botó *Raw* / *Download raw file*).
  2. Obre [Google Calendar](https://calendar.google.com) → roda de **Configuració** → **Importa i exporta**.
  3. Selecciona el fitxer `.ics`, tria el teu calendari i fes clic a **Importa**.
- **Outlook / Apple Calendar / Thunderbird:**
  - Fes doble clic sobre el fitxer `.ics` descarregat (o **Fitxer → Obre i exporta → Importa un fitxer iCalendar (.ics)** a Outlook) i confirma la importació dels 3 esdeveniments.

---

## 5. Comprovació i Resolució d'Incidències

### Llista de comprovació ràpida (de punta a punta)

Per verificar que tot el circuit continua sa en qualsevol moment dins l'horari operatiu (dl–dv, `07:07–22:37 Europe/Madrid`):

1. **Última execució a `cron-job.org`:** entra a `https://cron-job.org` → tasca `Actualitzar Dashboard Borsa` → pestanya **History** i comprova que s'ha executat a l'últim minut `:07` o `:37`.
2. **Resultat HTTP retornat:** verifica que l'estat a l'historial de `cron-job.org` és **`204 No Content`** (o `2xx` satisfactori).
3. **Execució a GitHub Actions:** obre `https://github.com/QuimEsc/ProvesInicials/actions/workflows/update-pages-data.yml` i confirma que hi ha una execució `workflow_dispatch` recent acabada en **`Success`** (en ~3–5 min quan actualitza dades, o ~25 s si el filtre anti-duplicats l'ha omés perquè ja havia acabat una altra execució feia menys de 12 min).
4. **Nous JSON generats al repositori:** comprova que hi ha commits recents de `github-actions[bot]` (`chore: update dashboard data and historical csv`) i inspecciona [`docs/data/manifest.json`](docs/data/manifest.json) (`generated_at`, `diagnostics.executed_at`, `diagnostics.run_number`, `diagnostics.error_count` i `tickers[].last_checked_at`).
5. **Publicació a GitHub Pages:** obre `https://github.com/QuimEsc/ProvesInicials/actions/workflows/pages/pages-build-deployment` (o comprova la capçalera `Last-Modified` de `https://quimesc.github.io/ProvesInicials/docs/data/manifest.json`) per verificar que el desplegament de Pages posterior al commit del bot ha acabat en **`Success`**.
6. **Marques temporals al navegador:** obri `https://quimesc.github.io/ProvesInicials/docs/?ticker=nasdaq` i revisa la barra **Estat** superior dreta:
   - `Sessió de cotització` (data real de l'última vela diària).
   - `Últim canvi de preu` (`[nou preu]` o `[sense canvis de preu]`).
   - `Última consulta font` (última resposta efectiva de Yahoo Finance per a l'actiu).
   - `Última execució` (batec de GitHub Actions), `Publicat` (`Last-Modified` de Pages) i `Comprovat` (darrer sondeig automàtic cada 60 s del navegador).

> [!NOTE]
> **Una cotització sense canvis no indica fallada:** fora d'horari de la borsa corresponent, caps de setmana, festius o sessions planes, `Últim canvi de preu` conservarà l'hora del darrer moviment real i mostrarà `[sense canvis de preu]` o `[mercat tancat · conservada última dada]`, mentre `Última consulta font` i `Última execució` continuaran avançant dins de la finestra operativa.

### Diagnòstic ràpid d'errors

| Símptoma | Causa probable i acció recomanada |
| :--- | :--- |
| **HTTP `401`** a cron-job.org | Token absent, mal format o caducat (revisa la data **18/02/2027**). Segueix el procediment de renovació de la Secció 3 i comprova `Authorization: Bearer <nou_token>`. |
| **HTTP `403`** a cron-job.org | Permisos insuficients al token (cal `Actions: Read and write` sobre `QuimEsc/ProvesInicials`) o restricció temporal de l'API de GitHub. |
| **HTTP `404`** a cron-job.org | Repositori o ruta del workflow errònies, o el token no té seleccionat `QuimEsc/ProvesInicials`. Revisa l'URL exacta i l'abast del token. |
| **HTTP `204` / OK però sense dades noves** | 1) Comprova el log de `Update dashboard data` a GitHub Actions.<br>2) Verifica si `Skip duplicate or out-of-window backup run` va ometre l'execució (< 12 min des de l'anterior).<br>3) Revisa `diagnostics.error_count` i `refresh_status` (`fallback_error` = error puntual de Yahoo Finance; `cached_outside_window` = fora d'horari).<br>4) Comprova si `pages build and deployment` ha fallat o té retard respecte al commit del bot. |
| **Retards ocasionals (30–60 min)** | Distingeix la capa on es produeix mirant les marques temporals:<br>- **Retard del programador:** `cron-job.org` o el `schedule` de suport s'ha disparat més tard (`diagnostics.executed_at`). Fins a 60 min dins l'horari de mercat és acceptable.<br>- **Temps d'execució:** el job de GitHub Actions triga ~3–5 min (descàrrega + 51 tests).<br>- **Publicació / caché navegador:** `pages build and deployment` triga ~1 min extra, i la web pot tardar fins a 60 s (interval de sondeig de `docs/app.js`) o mostrar retard CDN de Pages. |

---

## 6. Seguretat i Integritat

- **Zero secrets al repositori:** no inclogues mai tokens, contrasenyes ni capçaleres `Authorization` amb valors reals en `README.md`, fitxers `.ics`, issues, commits o logs.
- **Principi de mínim privilegi:** el token de `cron-job.org` només ha de tindre accés al repositori `QuimEsc/ProvesInicials` i únicament amb `Actions: Read and write`. No utilitzes un Classic PAT global ni dones permisos d'escriptura de codi (`Contents`) o administració.
- **Rotació segura:** prova sempre el token nou a `cron-job.org` amb **Test run** abans de revocar l'antic, per evitar talls accidentals.
- **Integritat dels models i seguiment:** la programació i el manteniment documentats ací només orquestren la descàrrega i publicació periòdica. No modifiques `portfolio_rebalance.py`, `logic_*.py`, `freeze/` ni `tracking/` fora del protocol formal descrit a [`freeze/README.md`](freeze/README.md).
