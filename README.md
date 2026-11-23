# Hanzi Lens

Lit et traduit les caractères chinois vus par la caméra du téléphone ou de l'ordinateur.

**Ouvrir l'app : https://leophalippou-cpu.github.io/hanzi-lens/**

Tout est fait maison : un réseau de neurones convolutif entraîné pour ce projet (3 755 caractères simplifiés, manuscrits ou imprimés), exécuté dans la page par notre propre code (WebGPU sur la carte graphique, sinon JavaScript sur le processeur), un découpage des caractères écrit à la main, un traducteur neuronal maison et un dictionnaire pour le mot à mot. Aucun service extérieur : après la première ouverture, l'app marche hors connexion.

Précision mesurée sur les 223 991 caractères manuscrits de test de CASIA-HWDB1.1 (60 personnes jamais vues à l'entraînement) : 94,1 % du premier coup, 98,4 % parmi les 3 propositions.

## Traducteur neuronal

Les phrases reconnues sont traduites par notre propre Transformer chinois → français (11 millions de paramètres), entraîné sur 9 millions de phrases traduites des corpus libres OPUS, et exécuté dans la page par notre code JavaScript. Le mot à mot du dictionnaire est affiché en dessous pour vérifier.

## Données et licences

- Écriture manuscrite : CASIA-HWDB1.1, Institut d'automatisation de l'Académie chinoise des sciences. Usage de recherche et personnel, pas d'usage commercial. Référence : C.-L. Liu, F. Yin, D.-H. Wang, Q.-F. Wang, « CASIA Online and Offline Chinese Handwriting Databases », ICDAR 2011.
- Texte imprimé : images générées avec les polices chinoises de Windows et des polices libres OFL (Noto Sans/Serif SC, LXGW WenKai, ZCOOL, Ma Shan Zheng, Long Cang), seulement pour l'entraînement.
- Phrases traduites : corpus OPUS (https://opus.nlpl.eu) — OpenSubtitles, CCMatrix, MultiParaCrawl, TED2020, NeuLab-TedTalks, News-Commentary, WikiMatrix, TICO-19 — chacun sous sa propre licence.
- `dict/zh.json` est tiré de **CFDICT** (chinois-français, https://chine.in, CC BY-SA 3.0) et de **CC-CEDICT** (chinois-anglais, MDBG, https://www.mdbg.net, CC BY-SA 4.0) ; il est distribué sous la même licence CC BY-SA.
