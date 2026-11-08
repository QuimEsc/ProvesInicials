# HANDOFF GENERAL 391

Actualitzat: 2026-11-08T20:45:31+01:00. Commit de les regles: `5dab23ad31ff33fa2a760b247c19ae627c87ab4f`. Congelació efectiva: **2026-11-08T20:45:20+01:00**. Primera sessió admissible: **2026-11-09**. Publicació: **en curs; encara no declarada completada**.

## A. Origen de la 391 i de la 391*

391 és el conjunt de paràmetres Second Pass 391 del motor original. El score suma cinc blocs: tendència World, tendència Nasdaq, momentum World de 252 sessions, volatilitat World de 42 sessions i mitjana dels tipus BCE/FED. Un estat positiu aporta tot el pes del bloc; neutral, la meitat; negatiu, zero. Els pesos respectius són 18,4005/28,0396/22,8988/10,7878/19,8733. Els valors complets, buffers i llindars estan a `freeze/parameters.json`; no s’han de copiar arrodonits d’aquest resum.

La 391 confirma els canvis durant quatre setmanes i conserva la histèresi. Entrades 391: risc-on 70,5519943575, defensiu 45,4822879079; retenció 58,4133314095/59,9912313194. El crash ordinari i la seua reentrada amb score/MA75 continuen formant part del model. **391* només canvia el risc-on 40/55/5 a 10/85/5.** Defensiu 45/15/40, semidefensiu 55/20/25 i normal 60/25/15 es conserven.

1207 manté els seus propis paràmetres, scores i transicions. Que haja coincidit amb 391 en moltes dates no permet substituir-ne la lògica. El motor responsable de producció és `portfolio_rebalance.py`.

## B. Investigació AB3

AB3 es va dissenyar per reaccionar a una caiguda ràpida abans de la confirmació setmanal. Nasdaq PR: rendiment de 10 sessions≤−8% i dos tancaments consecutius sota MA20; preparació amb MA200 i sigma 63 mostral retardada. Retira el 50% de cadascun dels dos objectius d’equity i el transfereix al monetari. Execució al següent tancament comú, mínim 3 sessions i cooldown 5 després de restaurar. Una segona alarma activa no reinicia l’episodi.

RE5 restaura amb dos tancaments sobre MA10 i sigma 5/sigma 63 retardada<1,5, o amb dos sobre MA20. **No significa cinc dies estables.** AB3_S3 utilitza el detector AB3 i la reentrada setmanal S3: DDWorld respecte d’ATH≤5%, o score>55,45843288247754 i World o Nasdaq sobre MA75.

Airbag S3 original és diferent: DDWorld≥8%, màxim 70 dies des del pic, ruptura MA200 amb buffer i volatilitat World 42≥25%. Quan està armat imposa 55/20/25 només si l’objectiu ordinari té menys 25%cash. Conserva el detector, la reentrada i el calendari originals; no rep els timers AB3.

COVID: AB3 executa 28/02/2020; S3 original 16/03/2020. RE5 pot reentrar massa prompte en baixistes escalonats, com 2022; S3 pot perdre recuperacions per protecció llarga. AB3_S3 continua amb menor MDD P95 modern principal. No s’incorporen D7/D9/D10/RB/RD/VOL. Els 49 cicles antics,37 falsos de preu i 16 falsos econòmics tenen definicions diferents; els marginals i els oracles no són estalvis executables ni additius.

## C. Història des de 1926

Les fonts French/CRSP són carteres de recerca TR revisables. El moderat és mercat ampli dels EUA, no MSCIWorld internacional. A:Small Growth des 01/07/1926. B:tres de dotze indústries amb major volatilitat passada de 252 sessions, selecció anual a juny i execució a juliol; inici 01/07/1927. C:mercat 1,5×, finançament 0,5×RF+100 pb ACT/365,2425 i reajustament diari. Es preserven els costos interns B/C i el RF publicat.

Les carteres són passives 10/85/5,20/80/0 i 30/70/0, amb reequilibri al primer tancament de gener. No són la 391 completa des 1926 ni un Nasdaq antic. S3 històric és **AB3_S3_DD_ONLY**, sense la branca de scores. El senyal TR dels proxies és una adaptació declarada del PR original.

1929 mostra protecció parcial amb pèrdues encara catastròfiques.1973 mostra vulnerabilitat davant caigudes lentes;1987 pot executar després del gap. World TR no cobreix el pic dotcom de març 2000. Les proves amb Nasdaq real des 1985 es mantenen separades dels proxies. Les revisions CRSP/CIZ no certifiquen una vintage antiga point-in-time ni invertibilitat 1930.

## D. Sonnet/Gemini

SONET va reproduir independentment 1453 setmanes 391/S3, el monetari i els controls. SONET/Airbag va provar 16 candidats de 8 famílies: VT(volatilitat),TR(tendència),DD(drawdown),CS(CUSUM),SY(sistèmic),PG(progressiu),VP(pic de volatilitat),VX(VIX), més 4 controls. Cap candidat va passar tots els filtres de Categoria A. VP1 i VT1 van justificar més observació, no una promoció universal. La resta queda arxivada, sense afegir-la alweb.

Cal llegir les correccions de l’auditoria Codex: el text antic descriu malament RE5, presenta 24/35 com a sortides VT1 quan el codi usa 25,50%/38,25%, i atribueix efectes a bandes 20% inexistents. Les mostres S&P152 i World 131 no són comparables sense les dates comunes. No es reescriu l’informe antic: les correccions i el seu impacte estan documentats separadament.

## E. Auditoria independent Codex

Va reconstruir indicadors, màquines d’estat i comptabilitat. VP1 original comprova dues sigmes contra els seus pics causals, però només el preu actual sobre MA10. VP1_2MA afegeix únicament el preu anterior sobre la MA10 anterior. És una variant descoberta després d’observar la història.

VT1 real baixa sota 25,50%/38,25% després de tres confirmacions; no aplica timers globals. Es van contrastar costos, reentrades, deriva de posicions, cohorts, falses alarmes i bootstrap condicional. No hi ha un percentatge de confiança calibrat ni una prova de superioritat futura.

## F. Doble verificació definitiva

Font de veritat: `CODEX_VP1_DOBLE_VERIFICACIO`, amb 22 tests,2520 contrastos A/B,528 controls,120 trajectòries i 230.448 finestres descriptives de 10/15 anys. Les finestres solapades no són observacions independents.

**World TRUSD,15 anys,131 finestres, cost 50 pb:**

| model        | cagr_p50   | mdd_p50   | mdd_p95   |   annual_turnover_mean |   annual_orders_mean | annual_cost_mean   |
|:-------------|:-----------|:----------|:----------|-----------------------:|---------------------:|:-------------------|
| AB3_RE5      | 11.6537%   | 37.6041%  | 37.6041%  |               1.0612   |                 9.36 | 1.0612%            |
| AB3_S3       | 10.9772%   | 31.8211%  | 31.8211%  |               0.615064 |                 4.96 | 0.6151%            |
| BASE         | 11.5429%   | 41.5350%  | 41.6506%  |               0.33373  |                 1.44 | 0.3337%            |
| VP1_2MA      | 11.2488%   | 31.9881%  | 31.9881%  |               0.710851 |                 6.24 | 0.7109%            |
| VP1_ORIGINAL | 11.3255%   | 34.0958%  | 34.0958%  |               0.748012 |                 6.56 | 0.7480%            |
| VT1          | 11.6510%   | 37.2752%  | 37.3223%  |               0.51605  |                 6.4  | 0.5160%            |

La rotació és la mitjana dels anys de calendari complets 2001–2025. Volum brut equity=2×rotació. Cost de 25 anys de calendari 02/01/2001–02/01/2026, per 100.000USD inicials: BASE37.415USD, RE598.825, S369.485, VP1 original 72.140,2MA69.338, VT159.109. Són càrrecs nominals sobre patrimoni creixent, no taxes anuals ni percentatges de patrimoni final.

2MA redueix el MDDP95 i la rotació, però té menys CAGR mediana a 15 anys. A10 anys la seua mediana de MDD empitjora lleument i la de CAGR puja lleument. En COVID, original reentra 03/04/2020 i 2MA09/04: quatre sessions més tard i 5,66 pp menys retorn del rang. El 2022 també empitjora 2MA(0,386 pp de retorn i 0,357 pp de MDD); en GFC millora 2,108 pp de MDD i 2,546 pp de retorn.2010/2024 quasi empaten;2015–16 perd 0,424 pp més amb 2MA;2011 millora. Les negatives formen part de la conclusió.

En A/B/C,2MA redueix rotació i MDD continu, però només B millora CAGR continu; en C empitjora MDDP95. Sense 1929/2008, tots dos VP1 perden CAGR i poden empitjorar MDD en B/C. No hi ha dominància universal ni garantia de límit de pèrdua. BASE continua sent referència;2MA té prioritat per observar protecció/rotació, original és el comparador essencial, RE5 el candidat de retorn/reentrada ràpida i S3 el control defensiu.

## G. Producció

Repositori de treball: `C:\Users\User\CodexProjects\ProvesInicials-VP1-freeze`. Origin:https://github.com/QuimEsc/ProvesInicials. Branca: `codex/vp1-freeze-2026-11-08`. Estat anterior recuperable: `59d19342114318f25eb6b5de2c0938c312cf8f9b`. Les còpies anteriors amb CSV modificats no s’han sobreescrit.

`build_pages_data.py` genera els JSON i l’històric CSV de `docs/data/`. El motor ordinari continua setmanal; AB3 i VP1 són diaris amb tancaments confirmats. `docs/app.js` consumeix `rows_by_strategy`, pesos ordinaris/efectius i ordres pendents. L’HTML mostra set opcions en l’ordre sol·licitat. Sempre obri 391*, fins i tot amb una preferència antiga d’estratègia guardada. L’actiu, timeframe i memòria continuen funcionant.

Canvis: `portfolio_rebalance.py`, `build_pages_data.py`, `docs/index.html`, `docs/app.js`, els dos fitxers de rebalanç generats i elworkflow. Afegits: `prospective_tracking.py`, `freeze/`, `tracking/`, fixtures i tests VP1. CSS, línies, zones, indicadors i altres apartats no s’han redissenyat.

Les cinc estratègies antigues reprodueixen els seus 4208/6997 registres amb inputs idèntics, llevat de l’etiqueta autoritzada.51 contrastos de producció reprodueixen pesos, estats, operacions, fees, rotació,NAV i les 131 finestres congelades.34 tests locals passats. Navegador: set opcions i canvis, default després de recarregar,22 actius, gràfics, macro, timeframes i toggles;1440/430 px, sense errors JS ni desbordament global.

Elworkflow `update-pages-data.yml` manté la programació existent. Abans del commit del bot verifica les regles congelades, el registre i els tests. Les captures de fonts poden actualitzar-se; els hashes de les regles no. Consultar `REGISTRE_PUBLICACIO_GITHUB.md` per l’estat públic real, elcommit publicat i els runs Actions.

## H. Congelació

No canviar AB3,sigma 10,0,60,MA10,dues confirmacions 2MA,mínim 3,cooldown 5,pesos,histèresi,reequilibris ni costos durant l’observació. Una correcció tècnica exigeix conservar la versió antiga, documentar impacte i crear una identitat nova.4075 fitxers de recerca estan inventariats per SHA256; els experiments acabats romanen de només lectura.

`freeze/manifest.json` verifica els fitxers essencials en format de text canònic Git LF; els manifests de dades/recerca són de bytes exactes. `freeze/identity.json` referencia elcommit immutable de regles sense autoreferències. La data real no s’antidata. Les dades 06/11 fins a la congelació són posteriors albacktest però ja conegudes.

## I. Novembre 2027

Llegir `PROTOCOL_VALIDACIO_2027.md`. Comparar només observacions estrictament noves, amb capital igual i estats heretats.50 pb principal i 10 pb separat. El feed automàtic actual és **PRUSD**, diferent del TR auditat: no comparar directament els dos CAGR. Les revisions es conserven separades de la primera observació.

No convertir un any en una estimació fiable del CAGR de llarg termini ni calcular MDDP95 de 15 anys amb un any. Si no hi ha cap crisi, no concloure protecció extrema. Si falten captures o hi ha incidències, declarar-ho abans d’avaluar. Qualsevol millora temptadora és una hipòtesi per a una investigació nova, sense alterar els models observats.

Ordre de lectura: `LLEGIR_PRIMER.md` → aquest HANDOFF → models congelats → índex i informes necessaris → protocol 2027 → registre de publicació i manifests. No repetir experiments ni descàrregues sobre els lliuraments antics per defecte.
