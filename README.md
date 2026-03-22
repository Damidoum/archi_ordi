# Projet Architecture des Ordinateurs - 2025-2026

**Auteurs** : Tom Mariani, Damien Rouchouse

Ce projet implémente un calcul de distance multi-threadé et vectorisé (AVX) en langage C. L'objectif est de comparer les performances entre une approche scalaire séquentielle et des optimisations parallèles.

## Compilation et Exécution

Le projet utilise un `Makefile` pour automatiser la compilation avec les flags `-mavx`, `-mavx2`, `-pthread` et `-lm`.

**Pour les utilisateurs de Mac Silicon (M1, M2, M3...) :**
Pour compiler correctement les instructions x86_64 sur architecture ARM, utilisez :

```bash
arch -x86_64 make
```

Pour les autres systèmes (Linux/Windows) :

```bash
make
```

Cela génère deux exécutables :
- `./main_float` : Version utilisant des flottants simple précision.
- `./main_double` : Version de test utilisant la double précision pour la vectorisation.

## Choix techniques et versions

Nous avons implémenté deux versions pour mettre en évidence les limites de l'architecture :

- **Version Float (`main_float`)** : C'est la version de base demandée pour des vecteurs en simple précision. On a remarqué que le résultat AVX diffère légèrement du scalaire sur $1024^2$ éléments. C'est dû à l'accumulation dans les registres AVX 32 bits qui saturent leur précision.
- **Version Double (`main_double`)** : On a fait une version en double précision pour corriger ce problème. Les résultats sont alors plus précis, mais on ne traite que 4 éléments par registre AVX au lieu de 8, ce qui est moins performant.

**Note sur le Multi-threading** : On a observé que la version multithreadée est plus précise que la version AVX mono-thread. Comme on découpe le calcul en plusieurs blocs, chaque registre AVX accumule moins de valeurs pour chaque thread, ce qui limite la dérive de l'erreur d'arrondi avant la somme finale.
