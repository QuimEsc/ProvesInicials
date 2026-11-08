# Protocol de validació 2027

Preparat: 2026-11-08T20:41:42+01:00. Commit de regles: `pendent de commit`. Congelació efectiva: **pendent de fixació immutable**. Primera sessió admissible: **2026-11-09 prevista, condicionada a congelació/publicació**.

## 1. Dates i separació de l’evidència

Tall modern auditat:05/11/2026; World TR comença 29/12/2000(6501 closes) i no cobreix el pic dotcom 2000. Tall French:30/09/2026. Les passives amb proxies no són 391 completa.2019–2026 no és validació verge.

Dates previstes: primera sessió 09/11/2026, última 05/11/2027, revisió 08/11/2027. Si la congelació immutable o publicació acaba després 08/11/2026, endarrerir la primera sessió estricta a una sessió comuna posterior a la congelació efectiva. No antidatar.06/11 fins aleshores es classifica com a postbacktest conegut. L’inici real és la primera captura admissible completa; si arriba tard, no declarar l’any complet.

## 2. Models i fonts

Ordre:391*,391*+VP1_2MA,391*+AB3_RE5,391*+VP1 original,391*+AB3_S3,391*+Airbag S3,1207. IDs i regles exactes a `MODELS_CONGELATS_2026.md` i `freeze/parameters.json`.391* sempre és la predeterminada.

**Feed prospectiu automàtic: PRUSD delweb**, World i Nasdaq, sense reinversió de dividends. El TR auditat no està disponible en el feed actual. No inventar-lo via ETF ni anomenar PR com TR. Si en el futur es captura TR, mantindre una branca separada amb dates reals de primera recepció; no reemplaçar la història principal. Aquesta diferència limita la comparació directa dels CAGR nous amb els CAGRTR anteriors.

Cash reconstruït amb DFF, retard conservador de dos dies hàbils EUA(USFederal Holiday Calendar), tipus disponible del dia natural anterior, ACT/360 i capitalització diària. No és un monetari contractual. Els set models comparteixen exactament les mateixes captures.

## 3. Capital i execució

Capital inicial 1 per model i per escenari, primera sessió admissible confirmada. Compra inicial autofinançada amb el seu cost. Heretar règim, protecció, indicadors, pic, cooldown i confirmacions de tota la història causal. Registrar objectius, posicions, caixa, fee i execució inicial. Les dades conegudes de Nov 6 poden contribuir a l’estat heretat sense convertir-se en rendibilitat prospectiva.

Senyals amb tancaments comuns confirmats; no usar una cotització intradia com a close. Decisió alclose t, execució alclose comú següent. El retorn de la sessió d’execució correspon a les posicions anteriors. Un canvi ordinari i d’Airbag simultani produeix un sol objectiu i una sola operació de cartera. Les cames ETF no són cicles.

Elledger nou executa les ordres degudes a l’última sessió observada i conserva les posicions, sense liquidació final. No necessita saber que una data serà l’extrem d’una finestra. L’omissió d’ordres just a l’extrem dels backtests antics és una convenció d’aquelles finestres, preservada en la reproducció històrica.

## 4. Costos i rotació

Principal 50 pb, secundari 10 pb, en taules i informes separats. Fee=pb/10000×nominal absolut de compres+vendes equity. Cash sensefee.50 pb és una hipòtesi conservadora no verificada de transacció, no TER. No es coneix el desglossament de broker, bid-ask i FX; no inventar impostos o TER addicional.

Rotació=0,5×volum equity/pre NAV. Guardar també volum brut, cames de compra/venda i dates d’operació. Costs simultanis es paguen una vegada i es marquen JOINT; no atribuir tot el diferencial contra BASE causalment a una sola alarma.

## 5. Registre immutable i revisions

`tracking/initial.json` conserva l’estat heretat sense rendibilitat prospectiva fictícia. `tracking/blobs/` desa les entrades per contingut SHA256. `tracking/vintages/<sessió>/` conserva observacions i correccions amb timestamps, fonts, runtime, commits de regles/generació i decisions recalculades. `tracking/sessions/<sessió>.json` conserva una primera observació per sessió, cadena SHA anterior, preus, score, règim, protecció, pesos, pending simulat, execució, holdings, fees i NAV50/10 separats.

Els valors de preu ja observats no es reemplaçen en la branca principal. Les noves publicacions macro s’incorporen quan s’observen, sense fer-les retroactives en els comptadors de divendres. Les correccions van en vintages separades; elweb mostra la vintage actual i el registre principal preserva la primera observació. Cal explicar qualsevol divergència.

Senseclose nou no hi ha una nova operació ni fee. Si hi ha sessions intermèdies no capturades, el sistema marca `late_gap_not_strictly_observed`; no les ompli com si s’hagueren observat puntualment. Cal resoldre/documentar la incidència abans de declarar un any complet. No basta arribar a l’última data.

Git i els hashes permeten auditar canvis. No prometen immunitat a un esborrat deliberat o force-push del propietari. Els fitxers de regles es verifiquen en format Git LF; els manifests de fonts/dades són de bytes exactes. Qualsevol canvi tècnic exigeix una versió explícita, impacte i conservació de l’antiga.

## 6. Mètriques de la revisió

Per als set models: retorn net; CAGR efectiu ACT/365,2425; MDD; durada màxima sota màxim en sessions i dies naturals; volatilitat√252; Calmar; rotació; ordres i costos; activacions/reentrades; sessions i fracció protegida; retorn guanyat/perdut i MDD estalviat respecte 391*; recuperació després de caigudes; exposició mitjana World/Nasdaq/cash.

El generador prepara CSV diari, comparativa de models, catàleg d’episodis, comparacions aparellades per episodi i informes HTML separats 50/10 pb. Guarda retorns,DD, primeres activacions, pèrdua abans d’activar, reentrada, compra real simulada, dies protegits, fees i recuperació.

Un any no estima amb fiabilitat un CAGR de llarg termini. No calcular MDDP95 de quinze anys amb un any. Una ampliació retrospectiva descriptiva ha d’anar separada del període nou. Cap puntuació composta o confiança percentual basada en finestres correlacionades.

## 7. Crisis curtes i intenses: classificació fixada

La hipòtesi personal és que poden ser més curtes i intenses que 2008. No és una predicció confirmada i no modifica cap model.

Sèrie comuna Nasdaq PR. Màxim corrent des del començament prospectiu; episodi quan la pèrdua arriba al 10%, fins recuperar el mateixpic o al tall si continua obert. Primera arribada al−10%: ràpida si≤20 sessions des delpic; lenta si>60; intermèdia 21–60. Recuperació en V si mínim→recuperació delpic≤20 sessions; lenta si>60; intermèdia altrament; oberta si encara no observable.

Rebotfals descriptiu: després del mínim, pujada≥5% des del mínim corrent seguida, abans de recuperar elpic, d’una caiguda≥5% des del màxim corrent. Es desarma després de cada caiguda per no comptar cada sessió com un altre rebot. Els episodis niats no són independents.

Per episodi comparar detecció i execució, pèrdua prèvia, reentrada després del mínim, temps fora d’equity, retorn/MDD, ordres/costs i recuperació del model. No confondre recuperació del mercat amb la de la cartera. Si no hi ha crisi rellevant, dir-ho i no fabricar evidència de protecció extrema.

## 8. Comandes exactes

Des de `C:\Users\User\CodexProjects\ProvesInicials-VP1-freeze`, amb les dependències originals:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python prospective_tracking.py verify --base-dir .
python -m unittest discover -s tests -v
python prospective_tracking.py annual --base-dir . --output-dir tracking/review_2027
```

Només per un informe intermedi explícit, afegir `--allow-partial`. Sense sessions o amb inici/final incomplets, la comparació anual per defecte falla. No s’ha executat una validació amb dades futures inexistents.

## 9. Revisió 08/11/2027 i prohibició de reajustament

Primer comprovar manifests, dates, versions, runtimes, cobertura i incidències/revisions. Després comparar els set models a 50 pb;10 pb separat. Revisar totes les crisis i els períodes tranquils, inclosos els desfavorables. Valorar retorn, protecció, rotació i cost sense anunciar un guanyador universal.

No alterar llindars AB3,sigma 10,0,60,MA10,confirmacions 2MA,mínim 3,cooldown 5,pesos,histèresi,reequilibris o costos durant l’any. Actualitzar les dades per aplicar les regles no és recalibrar. Una idea temptadora nascuda després d’observar dades noves es documenta per a una investigació futura; no s’incorpora al model observat.
