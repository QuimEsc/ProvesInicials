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
