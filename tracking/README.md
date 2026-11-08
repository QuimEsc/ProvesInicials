# Seguiment amb dades noves

Registres de recerca, sense ordres reals. Les regles estan fixades a `freeze/`.
El generador desa vintages d'entrada comprimides per contingut, revisions
separades i una cadena de sessions originals. No sobreescriu decisions anteriors.

El feed de producció és **PR USD**, no el TR dels backtests moderns. Els set
models reben les mateixes dades; els costos50pb/10pb romanen separats. Cash DFF
amb retard conservador de dos dies hàbils EUA i ACT/360, sense fee sobre cash.

Les dades conegudes entre06/11/2026 i la congelació no són prospectives.
`initial.json` preserva l'estat heretat i no inicia una rendibilitat fictícia.
La primera sessió admissible consta a `freeze/identity.json`; l'inici real serà
la primera sessió completa registrada, amb capital1 i compra inicial.

Si falta la captura d'una sessió existent, s'emet `late_gap_not_strictly_observed`.
No s'ompli el buit com si els senyals s'hagueren registrat puntualment. Cal
documentar la incidència abans de declarar un any complet. Les revisions de
fonts van a `vintages/`, amb decisions recalculades separades de les originals.

Verificar: `python prospective_tracking.py verify --base-dir .`

Revisió anual, quan hi haja dades fins05/11/2027:
`python prospective_tracking.py annual --base-dir . --output-dir tracking/review_2027`

Un informe intermedi exigeix explícitament `--allow-partial`. Un any tranquil
no verifica protecció extrema ni permet estimar un MDD P95 de quinze anys.
