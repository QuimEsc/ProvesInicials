# Regles391 congelades

`parameters.json` conté les set configuracions i totes les convencions.
`manifest.json` verifica les fonts essencials amb SHA256 del text canònic GitLF.
Els manifests de dades i recerca mantenen hashes dels bytes originals.

`identity.json` referencia el commit immutable de les regles i la data efectiva.
Es grava en un commit posterior perquè un fitxer no pot contenir el SHA del
mateix commit que el crea. L'etiqueta `391-freeze-2026-11-08` conserva v1.
La versió activa v2 usa `391-freeze-2026-11-08-v2`, amb la correcció tècnica
documentada a `corrections.md`; la identitat en dona el SHA i l'hora exactes.

No modificar silenciosament els fitxers congelats. Un error tècnic exigeix
conservar la versió antiga, documentar l'impacte i crear una versió nova.
Les captures de mercat poden actualitzar-se: les regles no.

Lectura: `research_handoff.md`, `research_index.md`, `models.md` i
`validation_2027.md`. El projecte principal conserva els documents mestres
amb els noms sol·licitats i tots els informes antics en les seues rutes.

Tall modern05/11/2026, proxies30/09/2026. VP1_2MA és exploratori post hoc.
El seguiment automàtic usa el feedPRUSD actual delweb i no s'anomenaTR.
Les dades conegudes abans de la congelació no són prospectives.

`production_validation.json` documenta la reproducció de51mètriques/trajectòries
contra els resultats congelats i la regressió de les cinc estratègies antigues.
