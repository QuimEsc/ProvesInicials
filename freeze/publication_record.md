# Registre de publicació Git Hub

**Publicació funcional verificada: 2026-11-08T21:00:44+01:00.** URL: https://quimesc.github.io/ProvesInicials/docs/

- Estat anterior: `59d19342114318f25eb6b5de2c0938c312cf8f9b`.
- Commit funcional v 2 publicat: `bf999dfedc2bd02280420190aff14c7c5474e2e4`.
- Commit immutable de regles v 2: `3723b933780691f9c54bb6723f44da454ba38af3`.
- Etiqueta activa: `391-freeze-2026-11-08-v2`. v 1 conservada al tag 391-freeze-2026-11-08 i a freeze/archive/v 1.
- Branca de treball: `codex/vp1-freeze-2026-11-08`; branca publicada: `main`.
- Fixació efectiva de v 2: **2026-11-08T20:52:13+01:00**, després de publicar i verificar el tag al remot. No s’antidata.
- Primera sessió admissible: **2026-11-09**, encara futura en aquesta verificació. Sessions prospectives observades: **0**. Les dades Nov 6 són posteriors albacktest però ja conegudes, sense NAV prospectiu.
- Revisió prevista 08/11/2027, període previst fins 05/11/2027.

## Proves i Actions

51 contrastos quantitatius de producció contra el motor congelat, error relatiu màxim 2,31 e−15. Mateixes 131 finestres 50 pb i trajectòries diàries de BASE,RE5,S3,VP1 original i 2MA. Les cinc estratègies prèvies conserven els seus registres 4208/6997 amb els mateixos inputs, llevat d’etiquetes autoritzades.

34 tests del primer lliurament i la comprovació addicional de recepció tardana;35 tests en la versió final, més comprovació local/publicada de 7 opcions,7 canvis, default 391* després de recarregar amb preferència guardada, pesos visibles contra JSON,22 actius, memòria, timeframes, línies/zones, gràfics i macro. Mòbil 430 px i escriptori 1440 px, cap error Java Script, caprequest fallida ni desbordament global. CSS i disseny conservats.

Workflow de dades: https://github.com/QuimEsc/ProvesInicials/actions/runs/42517205129 · **success**.
Pages: https://github.com/QuimEsc/ProvesInicials/actions/runs/42517650430 · **success**.
Els commits posteriors del bot només actualitzen dades i captures, amb les mateixes regles verificades perhash. El commit documental posterior conserva els rebuts de publicació sense redefinir models.

## Set opcions públiques confirmades

| value                       | label               |
|:----------------------------|:--------------------|
| secondpass_391              | 391*                |
| secondpass_391_vp1_2ma      | 391* + VP1_2MA      |
| secondpass_391_ab3_re5      | 391* + AB3_RE5      |
| secondpass_391_vp1_original | 391* + VP1 original |
| secondpass_391_ab3_s3       | 391* + AB3_S3       |
| secondpass_391_airbag_s3    | 391* + Airbag S3    |
| secondpass_1207             | 1207                |

## Preservació i incidències

**4075 fitxers originals de recerca sense canvis SHA256**. Quatre fitxers amb modificacions no relacionades de les còpies prèvies també preservats. Cap optimització nova ni operació real. La correcció v 2 només impedeix atribuir una execució a unclose anterior a la recepció de la informació; cap canvi de model o resultat històric i cap sessió prospectiva afectada.

Cap incidència tècnica bloquejant pendent. Limitacions científiques conservades: feed prospectiu PRUSD diferent del TR auditat, proveïdors revisables, captures eventualment tardanes que s’han de declarar, i cap validació anual encara possible.50 pb és hipòtesi conservadora de transacció, no TER o cost contractual verificat.

## Fitxers del canvi funcional i del registre

- `.github/workflows/update-pages-data.yml`
- `build_pages_data.py`
- `dades_historiques/BTC-USD.csv`
- `docs/app.js`
- `docs/data/macro.json`
- `docs/data/manifest.json`
- `docs/data/rebalance.json`
- `docs/data/rebalance_history.csv`
- `docs/data/summary.json`
- `docs/data/tickers/bitcoin.json`
- `docs/index.html`
- `freeze/README.md`
- `freeze/archive/v1/identity.json`
- `freeze/archive/v1/manifest.json`
- `freeze/archive/v1/parameters.json`
- `freeze/corrections.md`
- `freeze/identity.json`
- `freeze/manifest.json`
- `freeze/models.md`
- `freeze/parameters.json`
- `freeze/production_validation.json`
- `freeze/research_handoff.md`
- `freeze/research_index.md`
- `freeze/research_sources_sha256.csv`
- `freeze/validation_2027.md`
- `portfolio_rebalance.py`
- `prospective_tracking.py`
- `tests/fixtures/vp1_frozen_reference.csv.gz`
- `tests/fixtures/vp1_ordinary.csv`
- `tests/test_portfolio_ab3.py`
- `tests/test_prospective.py`
- `tests/test_vp1.py`
- `tracking/README.md`
- `tracking/blobs/134c67410c59ba9ad956aa7288a2352a3e0bf9cb20f497880daffa90fc041671.json.gz`
- `tracking/blobs/1ae971d65c02c88622d87ea09252afb8580e2773630e29b994bb5aa6bd625a84.json.gz`
- `tracking/initial.json`
- `tracking/vintages/2026-11-06/85db687148b8466fdeca143e932ae95e0ff6903ae67130ca752379003c193ab6.json`
- `tracking/vintages/2026-11-06/a46b9dff86dfa66fd63c4696cac68b2a01267f6daf07827975790a243a04bdb1.json`
- `tracking/vintages/2026-11-06/c70e0dd46e798a9d6c1018f31bd709d9291ebce521e464789eb87c53a0e28d4f.json`
