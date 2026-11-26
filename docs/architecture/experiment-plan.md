# Eksperyment: porównanie modeli bazowych SSL do segmentacji komórek

Protokół i stan wdrożenia. Kontekst: [ssl-base-models.md](ssl-base-models.md), uruchamianie na klastrze:
[plgrid.md](plgrid.md).

## Pytanie i hipotezy (zapisane przed wynikami)

Który zamrożony enkoder SSL daje najlepszą segmentację komórek i reprezentacje przy małej liczbie anotacji i jaki
jest koszt. H1: modele pretrenowane na histologii biją ogólny DINOv2. H2: `h0-mini` jest konkurencyjny wobec
dużych modeli. H3: dla megakariocytów 0,5 µm/px wystarcza, dla małych komórek potrzeba 0,25.

## Konstrukcja

- **Jedna komórka siatki** = model × rozdzielczość głowicy × odsetek kafli treningowych × seed
  (`configs/experiments/*.yaml`, `scripts/run_experiment.py`). Indeks zadania tablicy Slurm wskazuje komórkę.
- **Stałe:** architektura i hiperparametry głowicy, kafle, podział po pacjentach, augmentacje.
- **Podział:** plik z kolumnami `tile_id, patient_id, split` (CSV albo `tiles.parquet`). Pacjent w dwóch zbiorach
  to błąd. Podzbiory kafli są zagnieżdżone dla danego seeda.
- **Test:** `--final` odblokowuje zbiór testowy, jeden raz po zamrożeniu konfiguracji. Bez flagi nie jest
  nawet liczony.
- **Metryki** (`mpn_al.evaluation`): PQ/DQ/SQ, mPQ i PQ per klasa, przedział 95% przez bootstrap **po
  pacjentach**, nakład korekty (obiekty do dodania, usunięcia, poprawy, piksele do poprawy), czas na kafel.
  Przy mniej niż 3 pacjentach przedział to NaN, bo nie ma uczciwego.
- **Wyniki:** `<out>/<grid>/<run_id>/result.json`, `tile_counts.jsonl` (zliczenia per kafel, do ponownych
  analiz), `provenance.json`, `head/`.

## Kolejność

| Krok | Stan |
|---|---|
| 0. `remote.sh smoke` na Atenie, czasy i pamięć | **wykonane 20.11.2026** (A100, wyniki w ssl-base-models.md, sekcja 10); runner siatki sprawdzony w tablicy 8 zadań na danych demo (plgrid.md) |
| 1. Decyzje: tło/ignore, obrys komórki, klasy, `mpp`, podział po pacjentach | do ustalenia z zespołem i lekarzem |
| 2. Metryki, runner, siatki, testy | **gotowe**, 111 testów (lokalnie, losowy enkoder) |
| 3. Pilotaż `pilot.yaml`: czas i dobór hiperparametrów na walidacji | po kroku 0 i 1 |
| 4. Zamrożenie konfiguracji, pełna siatka `encoders.yaml` | po pilotażu |
| 5. Jedno otwarcie testu (`--final`), baseline'y zewnętrzne, ablacje | po kroku 4 |

## Uruchomienie pilotażu

```bash
python scripts/run_experiment.py --grid configs/experiments/pilot.yaml --list        # ile komórek
bash scripts/plgrid/remote.sh put ./tiles '$SCRATCH/mpn_al_data/tiles'               # kafle, maski, split.csv
bash scripts/plgrid/remote.sh submit --array=0-17 scripts/run_experiment.py \
    --grid configs/experiments/pilot.yaml --images '$SCRATCH/mpn_al_data/tiles' \
    --masks '$SCRATCH/mpn_al_data/masks' --split-file '$SCRATCH/mpn_al_data/split.csv' \
    --mpp 0.25 --out '$SCRATCH/mpn_al_runs/experiments' --skip-existing
```

## Zbieranie wyników

```bash
python scripts/summarize_experiment.py <katalog> --split validation --reference dinov2-s14
```

Zapisuje `runs.csv` (jedna komórka = wiersz), `curves.csv` (średnia i odchylenie po seedach dla każdego modelu,
rozdzielczości i odsetka kafli, czyli krzywe uczenia), `comparison.csv` (każdy model wobec odniesienia, test
Wilcoxona sparowany po (rozdzielczość, odsetek, seed) z poprawką Holma) i `report.md`. Przy kilku parach
najmniejsze możliwe p jest duże (6 par: 0,031), więc brak istotności znaczy „za mało przebiegów”.

## Decyzja: cache cech enkodera

Nie zrobiony celowo. Trening stosuje przerzuty i zmiany koloru przed enkoderem, więc cache cech oznaczałby inną,
słabszą augmentację i zmienił wyniki. Wrócimy do tego po pomiarze czasu treningu na GPU (`smoke`): jeśli enkoder
dominuje koszt, można trenować z cache bez augmentacji fotometrycznej i porównać obie wersje na pilotażu.

## Wynik eksperymentu zastępczego na BaMBo (21.11.2026)

> **Uwaga (25.11.2026):** 28 z 204 obrazów ma zepsute pseudo-etykiety (patrz sekcja o wadzie pseudo-etykiet niżej); liczby w tej sekcji je zawierają.

**Co to jest:** `configs/experiments/bambo_proxy.yaml`, 204 obrazy BaMBo, 47 pacjentów (podział losowy po
pacjentach: 30 trening / 8 walidacja / 9 test, ziarno 0), etykiety to **maski jąder z baseline'u barwnikowego**,
nie anotacje lekarza. 4 modele otwarte × 4 odsetki obrazów (10–100%) × 3 seedy = 48 przebiegów na A100 (job
3567632, 4–5 min na przebieg), zbiór testowy policzony raz (`--final`), konfiguracja nie była stroiona (pilotaż
mierzył tylko czas). Mierzy: jak dobrze głowica na danym zamrożonym enkoderze odtwarza obrysy jąder
znalezione barwnikowo. **Nie mówi nic o megakariocytach, o zgodności z patologiem ani o komórkach z
cytoplazmą.** BaMBo nie zapisuje rozdzielczości (założone 0,5 µm/px).

PQ na zbiorze testowym (średnia z 3 seedów; odchylenie między seedami 0,003–0,024):

| model | 10% (13 obr.) | 25% | 50% | 100% (130 obr.) |
|---|---|---|---|---|
| `dinov2-s14` (ogólny) | 0,510 | 0,499 | 0,497 | 0,490 |
| `lunit-dino-s16` | 0,522 | 0,543 | 0,552 | 0,547 |
| `phikon-v2` | 0,517 | 0,524 | 0,538 | 0,539 |
| `midnight-12k` (ViT-g) | 0,499 | 0,518 | 0,525 | 0,521 |

Test sparowany (Wilcoxon po odsetku i seedzie, 12 par, poprawka Holma) względem `dinov2-s14`: `lunit-dino-s16`
+0,042 (p_holm 0,003), `phikon-v2` +0,031 (0,003), `midnight-12k` +0,017 (0,034).

**Jak to czytać:**
- Trzy modele pretrenowane na histologii lub dużych danych są nieznacznie lepsze od ogólnego DINOv2 (różnice
  0,02–0,04 PQ), co jest zgodne z H1, ale tylko w tym zastępczym zadaniu. Pary nie są niezależne: ten sam
  zbiór testowy (9 pacjentów) i zagnieżdżone podzbiory, więc p-wartości są **zbyt optymistyczne**. Prawdziwa
  liczba niezależnych jednostek to 9 pacjentów testowych.
- Lunit (ViT-S, 22 mln parametrów) jest tu nie gorszy niż ViT-g (1,1 mld) i ViT-L. Różnic między tymi trzema
  nie testowano. Duży model nie daje przewagi w tym zadaniu, a koszt treningu głowicy jest podobny (125–158 s),
  bo dominuje dekoder.
- Krzywe uczenia są niemal płaskie od 25% obrazów (dla `dinov2-s14` nawet lekko spadają). Zadaniem jest odtworzenie
  deterministycznej reguły barwnej, którą głowica uczy się z kilkudziesięciu obrazów, więc krzywe nie mówią nic o
  tym, ile anotacji lekarza potrzeba.
- Walidacja w pilotażu dała PQ ok. 0,60, test 0,50–0,55; różnica może wynikać z innego składu pacjentów
  (test to 9 pacjentów, w tym obrazy innej skali), nie badano jej.
- Nie sprawdzono: modeli bramkowanych (H-optimus, UNI2-h, Virchow2, H0-mini), rozdzielczości (H3), prawdziwych
  anotacji, innego podziału po pacjentach (jeden podział, ziarno 0).

Wnioski dla dalszych prac: potok od obrazu po `report.md` działa w skali 48 przebiegów i daje powtarzalne
wyniki (małe odchylenia między seedami). Ranking enkoderów trzeba powtórzyć na prawdziwych anotacjach komórek
przed jakąkolwiek decyzją.

### Powtórzenie eksperymentu zastępczego (22–23.11.2026)

Ta sama siatka (`configs/experiments/bambo_proxy.yaml`, bez zmian, bez `--skip-existing`, `--final`) uruchomiona ponownie na Atenie jako tablica 3573579 (indeksy 0–47) do nowego katalogu `exp_bambo_rerun`; wyniki lokalnie w `outputs/plgrid/exp_bambo_rerun/bambo_proxy` (runs.csv, curves.csv, comparison.csv, report.md). Wszystkie 48 komórek ma `result.json`, żadne zadanie nie padło ani nie było zgłaszane ponownie.

Porównanie z przebiegiem 3567632 (`outputs/plgrid/exp_bambo_final`), PQ na teście, parami po `run_id`: maksymalna bezwzględna różnica 0,0000 (wszystkie 48 wartości identyczne), więc żadna różnica nie przekracza progu 0,02. Średnie PQ per model (ułamek 0,10–1,00) i ułamek też się zgadzają w całości; porównanie z dinov2-s14 (comparison.csv) jest takie samo (lunit-dino-s16 +0,0418, phikon-v2 +0,0305, midnight-12k +0,0166, Holm p odpowiednio 0,0029, 0,0029, 0,0342). Różnią się tylko czasy (`train_seconds`, średnio o ok. 7 s), co potwierdza, że były to świeże wykonania, a nie ponowne wczytanie starych wyników.

Wniosek: pipeline (seedy, podział, trening głowy, ewaluacja) jest w tej konfiguracji w pełni deterministyczny, niedeterminizm GPU nie wpływa na PQ. Wartości z sekcji powyżej pozostają bez zmian.

## Zapis predykcji do wizualizacji (etapy 1–2, 23.11.2026)

Plan wizualizacji wyników każdego modelu na obrazach: najpierw przebieg zapisuje predykcje, potem figury rysuje się offline
(bez GPU i bez enkodera). Zrobione są etapy zapisu (1–2) i skrypt składający figury (3, `scripts/visualize_experiment.py`); nie uruchomiono go jeszcze na wynikach z klastra.

- **Konfiguracja:** sekcja `visualize: {per_split: N, seed: S}` w pliku siatki (brak sekcji = nic nie jest zapisywane).
  Nadpisanie z wiersza poleceń: `--viz-per-split N`, wyłączenie: `--no-viz`.
- **Wybór kafli zależy tylko od danych**, nigdy od wyniku modelu (`src/experiment/viz.py`), więc wszystkie przebiegi siatki pokazują te same kafle:
  `dense` (najwięcej komórek referencyjnych na megapiksel), `atypical` (rozmiar inny niż najczęstszy w zbiorze), `random` (ziarno z konfiguracji).
  Najpierw po jednym kaflu na pacjenta, dopiero potem powtórzenia. Grupa jest zapisana w `viz/index.csv`.
- **Bramka testu:** zbiór testowy jest wybierany i zapisywany tylko wtedy, gdy jest oceniany (`--final`), tak jak jego metryki.
- **Zawartość `<przebieg>/viz/<zbiór>/<tile_id>.npz`:** mapa instancji, prawdopodobieństwo (uint8), HV (int8), PCA tokenów w rozdzielczości siatki
  tokenów (uint8), klasy instancji oraz referencja i maska `ignore`, względem których liczono wynik. Test sprawdza, że TP, FP i FN policzone z tego pliku
  są równe zliczeniom w `tile_counts.jsonl`, więc figura nie może się rozjechać z tabelą.
- **Koszt (zmierzony 25.11.2026 na BaMBo, obrazy do 1920 × 1200):** ok. 3,4 MB na kafel, 3,0 GB na 48 przebiegów × 18 kafli; razem z głowicami (4,5 GB) katalog siatki ma ok. 7 GB.
  Wcześniejszy szacunek „kilkaset kB na kafel” był zaniżony. PCA liczone z macierzy kowariancji D × D (ok. 1 s dla ViT-g).
  `bambo_proxy.yaml` ma teraz `per_split: 9`; ta sekcja nie wpływa na żadną metrykę.

### Figury z zapisanych predykcji (`scripts/visualize_experiment.py`)

```bash
python scripts/visualize_experiment.py <katalog_eksperymentu> --images <obrazy> --out outputs/figures/<nazwa>     --split validation --fraction 1.0 --seed 0 --zoom 256
```

Na kafel (z grupy `dense`, `atypical` lub `random`, patrz wyżej) powstają: `_gallery.png` (wiersz na model: kontury | błędy | prawdopodobieństwo | PCA tokenów),
`_zoom.png` (najgęstsze okno powiększone bez interpolacji), `_curve.png` (kolumny to odsetki kafli treningowych, ten sam seed), `_seeds.png` (kontury
wszystkich seedów i miejsca niezgodności), `_disagreement.png` (ile modeli uznaje piksel za komórkę), a do tego `viz_scores.csv` (każdy przebieg × kafel) i `index.html`.

- **Błędy** są dopasowane tak jak w metryce PQ (IoU > 0,5): trafienie (TP), fałszywy alarm (FP), pominięcie (FN), w kolorach Okabe-Ito dla osób z zaburzeniami widzenia barw.
- **Spójność z tabelą:** `viz_scores.csv` ma kolumnę `matches_counts`; skrypt ostrzega, jeśli zapisana predykcja nie odtwarza `tile_counts.jsonl` przebiegu.
- **Zbiór testowy** istnieje tylko dla przebiegów z `--final`; bez niego skrypt kończy się błędem, zamiast cokolwiek rysować.
- Skrypt nie potrzebuje GPU ani enkodera; wystarczą zapisane `viz/` i katalog z obrazami (obraz jest skalowany do rozdzielczości roboczej predykcji).
- Teksty na figurach są po angielsku i bez polskich znaków (OpenCV); strona `index.html` jest po angielsku.
- **Uwaga do `_seeds.png` i `_curve.png`:** pojawiają się tylko wtedy, gdy siatka ma więcej niż jeden seed lub odsetek.

## Siatka BaMBo z zapisem predykcji i wada pseudo-etykiet (25.11.2026)

**Przebieg.** Tablica 3578584 (48 zadań, `--final`, `visualize: per_split: 9`) zakończona w całości: 48 × `result.json`, 48 × `viz/index.csv`, 864 zapisane predykcje
(`outputs/plgrid/exp_bambo_viz/`, ok. 7 GB). **PQ identyczne z `exp_bambo_final` w 48 z 48 przebiegów** (największa różnica 0,0), więc zapis predykcji nie zmienia metryk.
Figury: `outputs/figures/bambo_validation/` i `outputs/figures/bambo_test/` (po 45 obrazów i `index.html`); `matches_counts` = True we wszystkich 432 wierszach obu zbiorów.

**Znaleziona wada: pseudo-etykiety są zepsute na części obrazów.** Na obrazach z małą liczbą jąder (tłuszcz, kość) i na części obrazów 512 × 512 baseline barwnikowy
zaznacza jako „komórki” ogromne obszary (nawet 87% obrazu; na figurach referencja to proste, rozchodzące się linie). Przyczyna: `segment_nuclei_stain` dzieli piksele tkanki
progiem Otsu bez minimalnego poziomu hematoksyliny, więc gdy na obrazie prawie nie ma jąder, dzieli szum.

Kryterium „podejrzany” (dobrane **po** obejrzeniu figur): pierwszy plan > 30% obrazu albo > 5% obrazu w obiektach większych niż 500 µm². Spełnia je **28 z 204 obrazów**:
16 z 130 treningowych, 2 z 32 walidacyjnych, **10 z 41 testowych**.

PQ przy 100% kafli treningowych, średnia z 3 seedów (analiza wrażliwości, **nie zastępuje** liczb wyżej):

| model | test, wszystkie | test bez podejrzanych | tylko podejrzane (test) | walidacja bez podejrzanych |
|---|---|---|---|---|
| `lunit-dino-s16` | 0,547 | 0,594 | 0,291 | 0,630 |
| `phikon-v2` | 0,539 | 0,582 | 0,296 | 0,629 |
| `midnight-12k` | 0,521 | 0,573 | 0,233 | 0,613 |
| `dinov2-s14` | 0,490 | 0,551 | 0,186 | 0,606 |

- **Ranking się nie zmienia** (`lunit` ≥ `phikon` > `midnight` > `dinov2`), ale odstęp `dinov2` od reszty na teście maleje.
- **Różnica walidacja–test** (wcześniej niewyjaśniona, ok. 0,1) w dużej części wynika z częstszych zepsutych etykiet na teście: bez nich zostaje ok. 0,03–0,05.
- **Czego ta analiza nie naprawia:** 16 zepsutych obrazów było w treningu, więc głowice uczyły się na błędnych etykietach; wyłączenie ich z oceny tego nie cofa. Kryterium
  dobrano po fakcie i nie sprawdzono go ręcznie na wszystkich 28 obrazach (obejrzano kilka).
- **Decyzja do podjęcia:** dodać do baseline'u minimalny poziom hematoksyliny (albo odrzucać takie obrazy), wygenerować pseudo-etykiety od nowa i powtórzyć siatkę.
  Zmienia to dane eksperymentu, więc nie zrobiono tego bez zgody. Przy prawdziwych anotacjach problem znika.

**Inne obserwacje z figur** (9 kafli na zbiór, wybranych z danych; wnioski jakościowe z obejrzanych kilku kafli):
- Na gęstym kaflu walidacyjnym (`PGI_15__PGI_image_17`, powiększenie) błędy to głównie pominięte małe jądra; cztery modele wyglądają bardzo podobnie (PQ 0,67–0,71).
- Na obrazie z samym tłuszczem (`PGI_5__PGI_image_148`) wszystkie modele zawodzą, a prawdopodobieństwo komórki jest wysokie na pustym tle. To skutek (a przynajmniej zbieżność) treningu na etykietach tego typu.
- Na 9 kaflach walidacyjnych, średnio po seedach, więcej kafli treningowych daje więcej fałszywych alarmów i mniej pominięć; PQ jest prawie płaskie (`lunit`, `phikon`) lub lekko spada (`dinov2`).
  Na całym zbiorze walidacyjnym `lunit` i `phikon` lekko rosną, `dinov2` stoi w miejscu.

## Poprawka pseudo-etykiet i powtórka siatki (25.11.2026)

**Przyczyna.** Tło skanów BaMBo jest jasnoszare, a baseline barwnikowy liczył gęstość optyczną względem czystej bieli (255). Maska tkanki obejmowała więc prawie cały obraz
(także szkło) i na obrazach z małą liczbą jąder próg Otsu dzielił szum tła.

**Poprawka** (`segmenter/stain_baseline.py`): gęstość optyczna jest liczona względem tła danego obrazu (99. percentyl kanałów, nie niżej niż 180), ale **tylko gdy ten kolor
jest prawie neutralny** (rozrzut R/G/B ≤ 25), czyli wygląda jak szkło; w polu wypełnionym tkanką najjaśniejszy kolor to różowa cytoplazma i nie może zostać „bielą”.
`white_balance=False` przywraca dawne zachowanie. Odrzuciłem prostszy próg minimalny barwnika: na bladych obrazach kasował prawdziwe jądra.

**Skutek na etykietach** (204 obrazy, to samo kryterium podejrzenia co wyżej):

| | przed | po |
|---|---|---|
| obrazy podejrzane | 28 (trening 16, walidacja 2, test 10) | **11** (trening 8, walidacja 1, test 2) |
| obrazy wcześniej dobre, które stały się podejrzane | — | 0 |
| pierwszy plan na dobrych obrazach (mediana nowy/stary) | — | 0,96; 3 straciły ponad połowę (obejrzane: znikają fałszywe obrysy na szkle i kości) |

Pozostałe 11 to w większości obrazy 512 × 512 o różowym zabarwieniu tła, na których baseline zawodzi niezależnie od poprawki. **Wyłączono je z eksperymentu**
(`outputs/bambo_split_v2.csv`, lista w `outputs/bambo_excluded_v2.txt`); kryterium (pierwszy plan > 30% albo > 5% obrazu w obiektach > 500 µm²) dotyczy tylko etykiet,
nie wyników modeli, ale zostało dobrane po obejrzeniu figur. Podział po zmianie: trening 122, walidacja 32, test 39 obrazów; liczba pacjentów w każdym zbiorze bez zmian.
Tam, gdzie algorytm się nie zmienił, etykiety wygenerowane lokalnie są identyczne piksel w piksel ze starymi z klastra.

**Powtórka siatki:** job 3581464 (48 zadań, `--final`, maski `bambo_masks_wb/masks`, wyniki `exp_bambo_v2`). **Job anulowany 25.11.2026 przed startem** (na polecenie, razem z całą kolejką), więc wyników nie ma i trzeba go zgłosić od nowa; dane (maski, podział) są na klastrze. Liczb z tej powtórki
nie wolno porównywać wprost ze starymi: inne etykiety i inny skład zbiorów.

## Czego jeszcze brakuje

Baseline'y CellViT++ i HistoPLUS, sonda liniowa na embeddingach, test na prawdziwych anotacjach. Dopóki nie ma prawdziwych anotacji, żadna liczba z tej siatki nie jest wynikiem dla megakariocytów.
