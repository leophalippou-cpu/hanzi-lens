# Hanzi Lens

Lit et traduit les caractères chinois vus par la caméra du téléphone ou de l'ordinateur.

**Ouvrir l'app : https://leophalippou-cpu.github.io/hanzi-lens/**

Tout est fait maison : un réseau de neurones convolutif entraîné pour ce projet (3 755 caractères simplifiés, manuscrits ou imprimés), exécuté dans la page par notre propre code (WebGPU sur la carte graphique, sinon JavaScript sur le processeur), un découpage des caractères écrit à la main et une traduction mot à mot par dictionnaire. Aucun service extérieur : après la première ouverture, l'app marche hors connexion.

Précision mesurée sur les 223 991 caractères manuscrits de test de CASIA-HWDB1.1 (60 personnes jamais vues à l'entraînement) : 94,3 % du premier coup, 98,5 % parmi les 3 propositions.

## Données et licences

- Écriture manuscrite : CASIA-HWDB1.1, Institut d'automatisation de l'Académie chinoise des sciences. Usage de recherche et personnel, pas d'usage commercial. Référence : C.-L. Liu, F. Yin, D.-H. Wang, Q.-F. Wang, « CASIA Online and Offline Chinese Handwriting Databases », ICDAR 2011.
- Texte imprimé : images générées avec les polices chinoises de Windows, seulement pour l'entraînement.
- `dict/zh.json` est tiré de **CFDICT** (chinois-français, https://chine.in, CC BY-SA 3.0) et de **CC-CEDICT** (chinois-anglais, MDBG, https://www.mdbg.net, CC BY-SA 4.0) ; il est distribué sous la même licence CC BY-SA.
