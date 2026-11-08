# Models congelats 2026

Congelació efectiva: 2026-11-08T20:52:13+01:00. Commit `3723b933780691f9c54bb6723f44da454ba38af3`. Primera sessió admissible: 2026-11-09.

| id                          | nom                 | versio                   | base   | reentrada    | commit                                   |
|:----------------------------|:--------------------|:-------------------------|:-------|:-------------|:-----------------------------------------|
| secondpass_391              | 391*                | 391-freeze-2026-11-08-v2 | 391*   | ordinària    | 3723b933780691f9c54bb6723f44da454ba38af3 |
| secondpass_391_vp1_2ma      | 391* + VP1_2MA      | 391-freeze-2026-11-08-v2 | 391*   | VP1_2MA      | 3723b933780691f9c54bb6723f44da454ba38af3 |
| secondpass_391_ab3_re5      | 391* + AB3_RE5      | 391-freeze-2026-11-08-v2 | 391*   | RE5          | 3723b933780691f9c54bb6723f44da454ba38af3 |
| secondpass_391_vp1_original | 391* + VP1 original | 391-freeze-2026-11-08-v2 | 391*   | VP1_ORIGINAL | 3723b933780691f9c54bb6723f44da454ba38af3 |
| secondpass_391_ab3_s3       | 391* + AB3_S3       | 391-freeze-2026-11-08-v2 | 391*   | S3           | 3723b933780691f9c54bb6723f44da454ba38af3 |
| secondpass_391_airbag_s3    | 391* + Airbag S3    | 391-freeze-2026-11-08-v2 | 391*   | S3 original   | 3723b933780691f9c54bb6723f44da454ba38af3 |
| secondpass_1207             | 1207                | 391-freeze-2026-11-08-v2 | 1207   | ordinària    | 3723b933780691f9c54bb6723f44da454ba38af3 |

La font normativa és `portfolio_rebalance.py`, configuració exacta `freeze/parameters.json` i SHA256 a `freeze/manifest.json`. Identitat delcommit de regles a `freeze/identity.json` (fitxer separat per evitar una autoreferència impossible). La configuració JSON conté tots els pesos/llindars de 391 i 1207 sense arrodonir-los. Fonts PRYahoo World/NDX i BCE/FED originals; cash DFF. Les observacions posteriors a 05/11/2026 no formenpart delbacktest auditat.

391* només canvia risc-on a 10/85/5; defensiu 45/15/40, semi 55/20/25, normal 60/25/15.391 i 1207 mantenen scores, confirmació 4 setmanes, histèresi, crash ordinari i calendari. S3 original conserva detector, cap 55/20/25 i reentrada originals, no es converteix en AB3.

AB3: r 10≤−8%,2 closes sota MA20, MA200 i sigma 63 mostral retardada. Tall 50% delsdos objectius RV, no de les posicions derivades, a cash. Mínim 3 sessions, cooldown 5, segonaalarma activa ignorada. RE5:2 sobre MA10 i sigma 5/sigma 63 retardada<1,5, o 2 sobre MA20. S3 reentrada AB3:divendres DDWorld ATH≤5%, o score>55,45843288247754 i World/Nasdaq sobre MA75.

VP1 sigma 10:std(ddof 1)×√252; pic causal de l’episodi a cada data. Reentrada si sigma[t]≤0,60×pic[t] i sigma[t−1]≤0,60×pic[t−1], preuactual>MA10, mínimcomplert.2MA afegeix únicamentpreuanterior>MA10 anterior. Execucióalnextclosecomú; reentradausa règim 391 vigent, no el precrisi. VP1_2MA és posthoc i experimental, sense superioritatfutura demostrada.

50 pb principal/10 pb secundari són costos simulats sobre compres+vendes equity, no TER. Les sèries podenactualitzar-se; els paràmetres no. Capnoufiltre, optimització o ajust a crisiscurtes. Correcciótècnica: conservarversió, impacte, noucommit i nova identitat. No canviar la història silenciosament.
