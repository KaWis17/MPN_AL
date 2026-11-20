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
| 0. `remote.sh smoke` na Atenie, czasy i pamięć | **do wykonania przez Ciebie** (dostęp do klastra zablokowany w sesji) |
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

## Czego jeszcze brakuje

Baseline'y CellViT++ i HistoPLUS, sonda liniowa na embeddingach, test na prawdziwych anotacjach. Dopóki nie ma prawdziwych anotacji, żadna liczba z tej siatki nie jest wynikiem dla megakariocytów.
