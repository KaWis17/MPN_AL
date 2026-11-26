# Eksperyment: który model SSL najlepiej wybiera kafle do adnotacji (Active Learning)

Protokół zapisany **przed wynikami** (26.11.2026). Dane: jeden slajd (jeden pacjent), przygotowanie `prep_33de9068`,
2668 kafli 1024 × 1024 px, 0,22087 µm/px (pole 226 × 226 µm), artefakty ≤ 2%, udział tkanki ≥ 50%. Krok kafli 512 px,
więc **sąsiednie kafle nakładają się w połowie** (99,7% kafli dzieli powierzchnię z sąsiadem); 672 kafle z `x` i `y`
podzielnymi przez 1024 tworzą siatkę bez nakładania. 23 rozłączne fragmenty tkanki.

## Cel

Embeddingi mają oddawać różnorodność obrazów (struktury tkankowe, morfologia, komórkowość, barwienie), tak aby wybór po
jednym kaflu z klastra dawał zróżnicowany i reprezentatywny zestaw do adnotacji. Liczba klastrów = liczba kafli w rundzie AL.
Nie chodzi o klastrowanie według klas megakariocytów (jeden kafel zawiera wiele klas).

## Modele

`lunit-dino-s16` (ViT-S/16, 384-d), `phikon-v2` (ViT-L/16, 1024-d), `midnight-12k` (ViT-g/14, 3072-d `cls+mean`),
`dinov2-s14` (ogólny, 384-d), oraz `noise-control` (wektory losowe) jako poziom przypadku.

## Wejście i przestrzeń embeddingów

1. **Ten sam kafel i te same wycinki dla wszystkich modeli.** Kafel 1024 px nie trafia do modelu w całości: 9 okien po 507 × 507 px
   (112 µm), każde przeskalowane do 224 × 224 px przy 0,5 µm/px (`extract_embeddings.py`, tryb `grid`). Embedding kafla = średnia
   9 wektorów. Patch 14 i 16 daje różną siatkę tokenów wewnątrz modelu (16 × 16 i 14 × 14), ale obraz wejściowy jest identyczny.
   Okna nakładają się o około połowę, więc środkowe pasy kafla liczą się w średniej wielokrotnie, rogi raz (jednakowo dla wszystkich).
   Tryb `full` (cały kafel w jednym przejściu) jest ewentualną analizą wrażliwości.
2. **Kolejność:** uśrednienie okien (bez normalizacji pojedynczych okien) → normalizacja L2 wektora kafla w oryginalnym wymiarze
   → dopiero potem KMeans, silhouette, kNN, UMAP i wybór reprezentantów. Bez PCA i bez centrowania. KMeans (sklearn) liczy odległość
   euklidesową: na znormalizowanych wektorach jest to bliskie, ale nie identyczne z klastrowaniem kosinusowym (centroidy nie mają
   długości 1); wybór kafla najbliższego danemu centroidowi jest w obu miarach taki sam.
3. **Która miara w której przestrzeni:**

| miara | przestrzeń i odległość |
|---|---|
| silhouette, nakładanie sąsiedztw kNN, UMAP, wybór reprezentanta | embeddingi po L2, kosinusowa |
| Davies-Bouldin, Calinski-Harabasz | embeddingi po L2, euklidesowa; tylko do porównań K w obrębie modelu |
| pokrycie i różnorodność selekcji | cechy z pikseli, poza przestrzenią modeli |

W miarach sąsiedztwa kafle nakładające się przestrzennie są pomijane jako sąsiedzi (inaczej najbliższym sąsiadem bywa kafel z połową
tego samego obrazu); silhouette sprawdzana jest dodatkowo na 672 kaflach bez nakładania.

## Klastrowanie

K ∈ {5, 10, 20, 30, 50, 80, 100}, 10 seedów (`n_init=1`). Miary: silhouette (kosinusowa), Davies-Bouldin, Calinski-Harabasz, ARI między
seedami (średnia ± odchylenie). Analiza wrażliwości: 672 kafle bez nakładania; opcjonalnie PCA 128. Porównanie modeli: ARI partycji i
nakładanie sąsiedztw (k = 10). Bez CKA, tablic kontyngencji i bootstrapu po pacjentach (jeden pacjent).

## Selekcja

- Z każdego klastra kafel najbliższy centroidowi (kosinusowo).
- **Wariant główny: bez nakładania przestrzennego** (klastry od największego; jeśli kandydat nakłada się z już wybranym, bierzemy następny
  najbliższy z klastra; gdy cały klaster się nakłada, bierzemy najbliższy wolny kafel z całego slajdu i to odnotowujemy). Wariant bez
  ograniczenia raportowany obok z liczbą nakładających się par.
- Punkty odniesienia: **losowy** (1000 losowań z tym samym ograniczeniem), **przestrzenny** (kafle równomiernie po preparacie: najdalszy
  punkt w układzie slajdu, wiele punktów startowych), **KMeans na cechach z pikseli** (tylko jako pułap: optymalizuje miarę oceny).

## Ocena selekcji (cechy niezależne od modeli)

Z pikseli każdego kafla liczone są: udział szkła/tłuszczu, erytrocytów, zbitej tkanki eozynofilowej (kość), barwnika, udział i gęstość jąder
z barwienia hematoksyliną, mediany stężeń hematoksyliny i eozyny, ich rozrzut, tekstura (krawędzie). Cechy są standaryzowane. To
przybliżenia: nie uchwycą całej morfologii (np. atypii megakariocytów); potrzebna jest ocena patomorfologa.

| pytanie | miara |
|---|---|
| pokrycie różnorodności | średnia i 95. percentyl odległości każdego kafla do najbliższego wybranego (mniej = lepiej) |
| rzadkie struktury | liczba pokrytych rzadkich kategorii: kafle z górnych 5% każdej z cech (kość, erytrocyty, barwnik, gęstość jąder, szkło, tekstura) |
| różnorodność wybranych | średnia i minimalna odległość między wybranymi |
| reprezentatywność | rozbieżność Jensena-Shannona rozkładu grup cech: wybrane wobec wszystkich |
| przestrzeń | pokryte fragmenty tkanki i komórki siatki, średnia odległość do najbliższego wybranego sąsiada na slajdzie |
| stabilność | zgodność zbiorów między seedami: kafle identyczne i nakładające się |

Wyniki: średnia ± odchylenie po 10 seedach oraz percentyl w rozkładzie losowym.

## Reguła wyboru modelu (zamrożona)

1. **Kryterium główne:** średnia odległość do najbliższego wybranego kafla (pokrycie) dla K = 50 i K = 80, wariant bez nakładania,
   uśredniona po seedach i obu K.
2. **Pomocnicze:** rzadkie struktury, rozmieszczenie przestrzenne, stabilność; miary klastrowania jako wsparcie.
3. Różnica mniejsza niż odchylenie między seedami to remis; wtedy wygrywa tańszy model.
4. Jeśli żaden model nie bije selekcji losowej albo przestrzennej, raportujemy to wprost.

Dwa najlepsze modele według punktu 1 są dodatkowo badane dla K ∈ {50, 80} z `n_init=10` (10 seedów), a ich selekcje porównywane na
planszach (po jednym kaflu na klaster, K = 50, seed 0), z selekcją losową, mapą slajdu i planszami anonimowymi (A, B, C) dla patomorfologa.

## Ograniczenia

Jeden preparat, jedno barwienie, jeden skaner: wynik trzeba potwierdzić na kolejnych pacjentach. Kafel najbliższy centroidowi jest z natury
typowy, więc przy małym K rzadkie struktury mogą wypaść. Czy różnorodna selekcja poprawia model w AL, pokaże dopiero uczenie na adnotacjach.

## Wyniki (26.11.2026, joby 3582982–3582985 i 3582993, wszystkie `COMPLETED`)

Pliki: `outputs/selection/24070_25/` (`report.md`, `figures/index.html`, tabele CSV). Embeddingi: 2668 kafli na model, po 9 okien; A100: od 2,4 min (`dinov2-s14`)
do 4,7 min (`midnight-12k`), analiza 15 min. Doprecyzowanie liczb z nagłówka: w kodzie siatka bez nakładania ma 675 kafli, a fragmentów tkanki jest 9, gdy łączy się
kafle nakładające się **lub stykające** (23 przy samym nakładaniu).

### Ranking według zamrożonej reguły (pokrycie dla K = 50 i 80, mniej = lepiej)

| miejsce | model | pokrycie (± seedy) | bije selekcję losową w % losowań (K=50 / K=80) |
|---|---|---|---|
| 1 | `lunit-dino-s16` | 1,550 ± 0,010 | 100 / 100 |
| 2 | `midnight-12k` | 1,561 ± 0,011 | 99,9 / 99,1 |
| 3 | `phikon-v2` | 1,575 ± 0,009 | 96,7 / 99,8 |
| 4 | `dinov2-s14` | 1,601 ± 0,009 | 87,8 / 82,0 |

Punkty odniesienia: KMeans na cechach (pułap) 1,389; losowa 1,634; kontrola szumem 1,646; przestrzenna 1,677.

- **Remis `lunit-dino-s16` / `midnight-12k` jest na samej granicy** (różnica 0,01052 wobec progu 0,01057). Reguła daje zwycięstwo tańszemu `lunit-dino-s16` (22 mln wobec 1,1 mld
  parametrów), który i tak ma niższe pokrycie. Dwa najlepsze: `lunit-dino-s16` i `midnight-12k`.
- **Zysk jest umiarkowany:** najlepszy model ma pokrycie o ok. 5% niższe niż selekcja losowa i zamyka ok. jedną trzecią drogi od losowej do pułapu.
- Wszystkie cztery modele biją selekcję losową; kontrola szumem wypada jak losowa, a równomierne rozłożenie po preparacie gorzej niż losowe.
- **Sąsiedzi w przestrzeni modelu wyglądają podobnie** (odległość cech do 10 nienakładających się sąsiadów): `lunit` 2,14, `midnight` 2,23, `dinov2` 2,41, `phikon` 2,44; losowa para 4,43.
- **Rzadkie struktury** (z 6 kategorii) różnicują przy małym K: dla K = 20 `lunit` 5,5, `midnight` 5,4, `dinov2` 4,5, `phikon` 3,3, losowa 3,9. Dla K = 50 i 80 prawie wszyscy mają 5,6–6.
- **Reprezentatywność** (rozbieżność rozkładu grup cech; mniej = lepiej): `phikon` 0,017, `midnight` 0,019, losowa 0,021, `lunit` 0,040, `dinov2` 0,059. Selekcje modeli nie są więc
  jednocześnie najbardziej pokrywające i najbardziej proporcjonalne.
- **Rozmieszczenie na slajdzie** jest podobne do losowego (zajęte komórki siatki i odległość do sąsiada), z obu rdzeni tkanki; wybory nie skupiają się w jednym miejscu.

### Stabilność i analizy wrażliwości

- **Wybór mocno zależy od seeda:** między seedami identyczne jest tylko 20–29% wybranych kafli, a identyczne lub nakładające się 36–46% (K = 50 i 80). ARI klastrowań 0,35–0,44.
  Pokrycie jest przy tym stabilne (odchylenie ok. 0,01–0,02), czyli różne zestawy są podobnie dobre. `n_init=10` zmienia to niewiele (`lunit`, K = 50: dopasowanie 0,41 → 0,44, pokrycie bez zmian).
- **Siatka bez nakładania (675 kafli):** dla K = 50 `lunit` nadal najlepszy (1,635), dla K = 80 trzy najlepsze są nierozróżnialne (1,500–1,504); `dinov2` ostatni z modeli.
- **PCA 128:** różnice między modelami prawie znikają (K = 50: 1,656–1,665). **Kolejność trzech najlepszych nie jest więc odporna**; odporniejsze jest, że modele histologiczne wypadają
  nie gorzej niż `dinov2-s14`, szczególnie przy pełnych wymiarach.
- Bez ograniczenia nakładania w wyborze trafia się 2–8 nakładających się par, przy tym samym pokryciu.

### Dobór cech oceny (analiza dodatkowa, po wynikach)

Po usunięciu dowolnej jednej cechy albo jednej z sześciu grup cech kolejność modeli się nie zmienia (20 z 20 wariantów z pełnym zestawem włącznie:
`lunit` < `midnight` < `phikon` < `dinov2`; `outputs/selection/24070_25/robustness_descriptors.csv`). Cechy są skorelowane, więc to słaby test i nie zastępuje oceny patomorfologa.

### Plansze (K = 50, seed 0)

`lunit` i `midnight`: 4 kafle identyczne, 16 identycznych lub nakładających się (32%), 62% ma odpowiednik o podobnych cechach (wobec 52–56% przy porównaniu z losowymi). Na oko obie
plansze modeli zawierają więcej wariantów z beleczkami kostnymi i krwinkami niż losowa; to wrażenie z jednego seeda, nie pomiar. Plansze dla patomorfologa: `figures/blind_A.png`,
`blind_B.png`, `blind_C.png`; klucz w `blind_key.json` (nie pokazywać oceniającemu).

### Czego te wyniki nie mówią

- Jeden slajd i jeden pacjent: rozrzut po seedach nie obejmuje zmienności między preparatami.
- Miarą są przybliżone cechy z pikseli (głównie kolor, komórkowość, struktury), nie ocena patomorfologa ani atypia komórek. Model, który lepiej odpowiada tym cechom, nie musi lepiej
  wspierać Active Learning.
- Czy selekcja poprawia uczenie, pokaże dopiero trening na adnotacjach.

## Uruchomienie i stan

Kod: `src/evaluation/tile_descriptors.py` (cechy), `selection.py` (selekcje i miary), `selection_figures.py`, `selection_cli.py`,
`scripts/run_tile_selection.py`; testy `tests/test_selection.py` (12, na syntetycznym slajdzie z czterema typami tkanki).

Na Atenie (26.11.2026): kafle w `$SCRATCH/mpn_al_data/slide_24070_25/`; embeddingi jobami 3582982 (`dinov2-s14`), 3582983 (`lunit-dino-s16`),
3582984 (`phikon-v2`), 3582985 (`midnight-12k`); analiza jobem 3582993 (zależnym `afterok` od czterech poprzednich), wyniki w
`$SCRATCH/mpn_al_runs/selection_24070_25/analysis/`.

```bash
R='$SCRATCH/mpn_al_runs/selection_24070_25'; D='$SCRATCH/mpn_al_data/slide_24070_25'
bash scripts/plgrid/remote.sh submit scripts/extract_embeddings.py --base-model <model> --images "$D/tiles"     --mpp 0.22087244616234125 --batch-size 64 --out "$R/<model>/embeddings.parquet"
bash scripts/plgrid/remote.sh submit --dependency=afterok:<id1>:<id2>:<id3>:<id4> scripts/run_tile_selection.py     --tiles-parquet "$D/tiles.parquet" --images "$D/tiles" --embeddings "<model>=$R/<model>/embeddings.parquet" ...     --out "$R/analysis" --workers 8
bash scripts/plgrid/remote.sh fetch '$SCRATCH/mpn_al_runs/selection_24070_25/analysis' outputs/selection/24070_25
```

Cechy sprawdzono przed uruchomieniem na 300 losowych kaflach (skrajne wartości każdej cechy pokazują oczekiwane struktury: kość, tłuszcz,
erytrocyty, barwnik, komórkowość); `nuclear_size` i `large_nuclei` różnicują słabo. Próba generalna na prawdziwych kaflach z **losowymi**
embeddingami (nie są to wyniki): losowe przestrzenie wypadają jak selekcja losowa, a KMeans na cechach wyznacza pułap.

