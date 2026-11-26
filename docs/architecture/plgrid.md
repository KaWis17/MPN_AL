# Uruchamianie na PLGrid (Athena)

Runbook dla skryptów modeli bazowych SSL ([ssl-base-models.md](ssl-base-models.md)): embeddingi kafli,
segmentacja komórek, trening głowicy. Wszystko sterowane jednym skryptem `scripts/plgrid/remote.sh`,
uruchamianym **z Twojej maszyny**.

## Stan: co jest sprawdzone, a co nie

- **Przetestowane lokalnie:** logika skryptów (przesyłanie kodu, cytowanie argumentów, opcje tablic zadań,
  czekanie na job, pobieranie wyników) na atrapie klastra (`tests/test_plgrid_scripts.py`), narzędzia
  Pythona (shardy, scalanie, pobieranie wag z atrapą Hugging Face) i test dymny na CPU.
- **Wykonane na Atenie (20.11.2026):** `sync`, `setup`, `download phikon-v2 lunit-dino-s16 --verify` i
  `smoke` na A100-SXM4-40GB (job 3565487): test dymny PASS dla obu modeli, a 117 testów jednostkowych
  przechodzi na węźle. Zgodność torch 2.14.1 (CUDA 13.0) ze sterownikiem 595.91, ścieżka float16 i
  kolejka Slurm (partycja i grant z `.env`) działają. Nie sprawdzono: modeli bramkowanych, odczytu WSI
  z pliku NDPI i uruchomienia na prawdziwych danych.
- **Runner eksperymentu na A100 (20.11.2026):** `configs/experiments/demo.yaml` (2 modele × 2 odsetki × 2 seedy,
  dane syntetyczne, plik podziału po 4 pacjentach), `submit --array=0-7 scripts/run_experiment.py ... --final`
  (job 3565807): 8 zadań w 17–22 s. `summarize_experiment.py` zebrał je na klastrze dla walidacji i testu
  (PQ 0,84–0,89, bez znaczenia dla histologii). Ponowne zlecenie z `--skip-existing` (job 3565835) nie przeliczyło
  niczego: znaczniki czasu wszystkich `result.json` bez zmian. Zbiór testowy był liczony tylko z `--final`.
- **Tablice zadań (20.11.2026):** `put` danych demo (8 kafli), potem `submit --array=0-3` dla
  `extract_embeddings.py` (job 3565519) i `segment_cells.py --baseline stain` (job 3565521): wszystkie 8 zadań
  zakończyło się w 7–17 s. `merge_shards.py` scalił 4 shardy (8 wierszy). Sprawdzone względem przebiegu
  jednorazowego na CPU: każdy kafel występuje raz, shard i zawiera kafle i, i+4, embeddingi z GPU i CPU są
  zgodne (cosinus 1,000000), liczby komórek identyczne (8 × 12), 8 plików GeoJSON i 96 wierszy tabeli komórek.
- Pierwsze próby `smoke` wykryły błędy w samych testach na Linuksie (atrapy bez bitu wykonywalnego,
  dziedziczenie środowiska zadania przez atrapę klastra); poprawione, na klastrze 117 przechodzi.
- Fakty o Atenie (limity, moduły, internet) pochodzą z rozpoznania z 9.08.2026 opisanego w
  `AL_System/docs/phase5_plgrid_launcher.md` i nie były tu sprawdzane ponownie.

## 1. Przygotowanie (jednorazowo, na swojej maszynie)

1. **Ustawienia połączenia.** Skopiuj `scripts/plgrid/env.example` do `.env` w korzeniu repozytorium
   (plik jest w `.gitignore`) i uzupełnij `PLG_LOGIN` oraz, jeśli klucz nie jest w agencie,
   `PLG_IDENTITY_FILE`. Klucze są te same co w AL_System, więc można wskazać jego plik:
   `PLG_ENV_FILE=../AL_System/.env bash scripts/plgrid/remote.sh ...`. Czytane są tylko klucze `PLG_*`
   i `MPN_AL_*`, a plik nie jest wykonywany.
2. **SSH.** Skrypty łączą się w trybie nieinteraktywnym (`BatchMode`), więc klucz z hasłem musi być w
   agencie: w PowerShellu `ssh-add -t 8h <ścieżka do klucza>`. Na Windowsie uruchamiaj skrypty z **Git
   Basha**, bo wołają natywny `ssh.exe`, który widzi agenta Windows (Git Bash ma własny `ssh`, który go nie
   widzi). Szczegóły i pułapki: `AL_System/docs/plgrid_connection_setup_windows.md`.
3. **Końce linii.** Skrypty `*.sh` i `*.sbatch` muszą mieć LF (CRLF psuje je na klastrze). `.gitattributes`
   pilnuje tego w świeżym checkoucie, a `remote.sh sync` odmawia wysłania, gdy znajdzie CRLF.

## 2. Pierwsze uruchomienie

```bash
bash scripts/plgrid/remote.sh sync
bash scripts/plgrid/remote.sh setup
bash scripts/plgrid/remote.sh download phikon-v2 lunit-dino-s16 --verify
bash scripts/plgrid/remote.sh smoke --models lunit-dino-s16 phikon-v2
```

| Polecenie | Co robi | Gdzie |
|---|---|---|
| `sync` | wysyła kod do `~/mpn_al` (zastępuje tylko `src`, `scripts`, `configs`, `schemas`, `tests`, `docs`) i zapisuje grant z partycją w `~/mpn_al/.plgrid.env` | węzeł logowania |
| `setup` | tworzy venv w `$SCRATCH/mpn_al-venv` (moduł `Python/3.10.4`), instaluje torch i pakiet (`-e ".[hf,wsi,dev]"`) | węzeł logowania, potrzebny internet |
| `download` | pobiera wagi do `$SCRATCH/hf-cache`; `--verify` ładuje każdy model offline | węzeł logowania, potrzebny internet |
| `smoke` | wysyła kod, zleca job GPU z `smoke_gpu.sbatch`, czeka, pobiera raport do `outputs/plgrid/smoke-<jobid>/` | węzeł A100 |

**Test dymny** nie potrzebuje danych (generuje syntetyczne kafle). Dla każdego modelu sprawdza ładowanie
z cache offline, przebiegi enkodera dla 224, 448 i 1024 px w float32 i w float16 z autocastem (czas,
pamięć, zgodność embeddingu), embeddingi kafli, kilka kroków treningu głowicy (z float16 na GPU) i
segmentację całych kafli ze sprawdzeniem, że maska, tabela komórek i GeoJSON mają tyle samo obiektów. Potem
uruchamia testy jednostkowe na węźle. Wynik to `smoke_report.md` i `smoke_report.json`.

**Kryterium zaliczenia:** `SMOKE TEST PASSED`. Ostrzeżenia (`warn`) nie przerywają testu, ale trzeba je
przeczytać: na przykład `fp16_cosine` poniżej 0,98 oznacza, że float16 zmienia embeddingi tego modelu.

Raport z pierwszego przebiegu (lub jego `smoke_report.md`) jest tym, czego brakuje, żeby uznać wdrożenie
za zweryfikowane. Zawiera wersje, kartę GPU, czasy i pamięć, a więc dane do planowania rund.

### Gdy `smoke` kończy się błędem

| Objaw | Przyczyna i działanie |
|---|---|
| `Permission denied (publickey)` / `cannot log in` | klucz nie jest w agencie (`ssh-add -l`), albo klucz publiczny nie dotarł jeszcze do klastra (do kilkudziesięciu minut po dodaniu w portalu) |
| środowisko: `PyTorch sees no CUDA device` | sterownik węzła jest za stary dla domyślnej wersji CUDA w torch. `MPN_AL_TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/plgrid/remote.sh setup`, potem znów `smoke`. AL_System blokuje torch 2.13.0 z bibliotekami CUDA 13; czy to pasuje do sterowników Atheny, pokaże ten test |
| `<model> / load` kończy się `LocalEntryNotFoundError` lub `GatedRepoError` | wagi nie są w cache (brak internetu na węźle obliczeniowym): `download`, a dla modeli bramkowanych sekcja 3 |
| `out of memory` | zmniejsz `--max-size` (przebieg segmentacji) i `--embed-batch`; ViT-g w float32 przy 1024 px potrzebuje wielu GB |
| zadanie długo `PENDING` | kolejka; `remote.sh status`, a `smoke --no-wait` zwraca od razu |
| `sbatch: error: Invalid account` | konto lub partycja z `.env` nie pasują; AL_System używał `plgaktywneuczenie-gpu-a100` i `plgrid-gpu-a100` (komenda `hpc-grants` na klastrze pokazuje Twoje granty) |

## 3. Modele bramkowane (UNI2-h, H-optimus, Virchow2, H0-mini, ...)

Licencje są **na osobę**: każdy, kto uruchamia taki model, sam składa wniosek na Hugging Face (adres
instytucji jako główny) i zalogowuje się na klastrze własnym tokenem. Tokenu nie przekazuje się skryptom
(ani przez `.env`):

```bash
bash scripts/plgrid/remote.sh shell
# na klastrze:
source scripts/plgrid/setup_env.sh
hf auth login
exit
bash scripts/plgrid/remote.sh download uni2-h h-optimus-0 --verify
```

Pamięć podręczna z takimi wagami (`$SCRATCH/hf-cache`) wolno współdzielić tylko z osobami, które same
zaakceptowały warunki modelu. Szczegóły licencji: [ssl-base-models.md](ssl-base-models.md), sekcja 3.

## 4. Zwykła praca

Najpierw dane na klaster. Kafle i anotacje (rzędu setek MB) wysyła `put`; skany WSI (GB) lepiej `scp` na
storage grupy, zgodnie z `AL_System/scripts/plgrid/sync_data.sh`.

```bash
bash scripts/plgrid/remote.sh put ./tiles_png '$SCRATCH/mpn_al_data/tiles'
```

Zmienne `$NAZWA` w apostrofach są rozwijane **na klastrze**; podstawienia poleceń (`$(...)`) nigdy się nie
wykonują. Skrypty z `scripts/` zleca się przez `submit` (opcje Slurm przed nazwą skryptu, reszta trafia do
skryptu):

```bash
# embeddingi kafli: 8 zadań, każde z własnym plikiem
bash scripts/plgrid/remote.sh submit --array=0-7 scripts/extract_embeddings.py \
    --base-model h-optimus-0 --images '$SCRATCH/mpn_al_data/tiles' --mpp 0.25 \
    --out '$SCRATCH/mpn_al_runs/round_01/embeddings.parquet' --skip-existing
# po zakończeniu (kontrola: remote.sh status); scalanie jest lekkie, więc może iść na węźle logowania:
bash scripts/plgrid/remote.sh run scripts/merge_shards.py embeddings '$MPN_AL_RUNS/round_01/embeddings.parquet'

# trening głowicy (jedno zadanie)
bash scripts/plgrid/remote.sh submit --time=02:00:00 scripts/train_segmentation_head.py \
    --base-model h-optimus-0 --images '$SCRATCH/mpn_al_data/tiles' \
    --annotations '$SCRATCH/mpn_al_data/annotations' --labels configs/labels/megakaryocyte_binary.yaml \
    --mpp 0.25 --val-ids '$SCRATCH/mpn_al_data/val_ids.txt' --out '$SCRATCH/mpn_al_runs/round_01/run_001'

# segmentacja: 8 zadań, wznawialna
bash scripts/plgrid/remote.sh submit --array=0-7 scripts/segment_cells.py \
    --head '$SCRATCH/mpn_al_runs/round_01/run_001/head.pt' --images '$SCRATCH/mpn_al_data/tiles' --mpp 0.25 \
    --out '$SCRATCH/mpn_al_runs/round_01/predictions' --skip-existing
```

Z `tiles.parquet` zamiast katalogu obrazów: `--tiles-parquet ... --slides-parquet ...` (WSI muszą być na
klastrze, a pakiet z dodatkiem `wsi`, co robi `setup`).

- **Tablice zadań** numeruj `0-(N-1)`: zadanie dostaje `--shard-index` i `--num-shards` z Slurma, a
  `extract_embeddings.py` zapisuje `embeddings.shard-III-of-NNN.parquet`. Zadań nie powinno być więcej niż
  kafli (puste zadania kończą się bez błędu i bez pliku, a `merge_shards.py --allow-missing` je pomija).
- **Wznawianie:** zlecenie tego samego polecenia z `--skip-existing` powtarza tylko brakujące kafle i
  shardy. Pliki pojawiają się pod docelową nazwą dopiero, gdy są kompletne.
- **Statystyki niepewności** z wielu zadań: `remote.sh run scripts/merge_shards.py tile-stats '<katalog_przebiegu>'`
  (albo `read_tile_stats(<katalog>)` czyta je bez scalania). Per-kaflowe `by_tile/*.geojson` i `cells/*.parquet`
  nie wymagają scalania: `pd.read_parquet("<katalog>/cells")` wczytuje wszystko.
- **Podgląd:** `remote.sh status`, `remote.sh logs [JOBID]`. Logi zadań leżą w `~/mpn_al/logs`.
- **Wyniki:** `remote.sh fetch '$SCRATCH/mpn_al_runs/round_01/predictions' ./outputs/round_01`.

### Trwałość

`$HOME` ma 10 GB (tylko kod), `$SCRATCH` jest okresowo czyszczony (cache i wyniki robocze). Wszystko, co ma
zostać (głowice, predykcje rundy, embeddingi), skopiuj na storage grupy, np.
`cp -r "$MPN_AL_RUNS/round_01" /net/pr2/projects/plgrid/plggaktywneuczenie/mpn_al/`
(do uzgodnienia w zespole, gdzie ma leżeć drzewo `data/` z [data-layout.md](data-layout.md)). Jeśli ma ono
leżeć na storage grupy, dowiąż je do kodu (`ln -s <storage>/data ~/mpn_al/data`): `sync` nie rusza niczego
poza katalogami z kodem.

## 5. Skrypty

| Plik | Rola |
|---|---|
| `scripts/plgrid/remote.sh` | sterowanie z Twojej maszyny: `sync`, `setup`, `download`, `smoke`, `submit`, `run` (lekki skrypt na węźle logowania), `status`, `logs`, `put`, `fetch`, `shell` |
| `scripts/plgrid/setup_env.sh` | środowisko na klastrze (moduł Pythona, cache na `$SCRATCH`, aktywacja venv); `source` na początku każdego zadania |
| `scripts/plgrid/remote_setup.sh` | tworzy venv i instaluje pakiet; `MPN_AL_TORCH_SPEC`, `MPN_AL_TORCH_INDEX`, `MPN_AL_EXTRAS` |
| `scripts/plgrid/run_gpu.sbatch` | zadanie GPU dla dowolnego skryptu z `scripts/`; dokłada `--num-workers` i opcje shardów tylko tam, gdzie skrypt je zna; `HF_HUB_OFFLINE=1` |
| `scripts/plgrid/smoke_gpu.sbatch` | zadanie testu dymnego (`smoke_test.py` i testy jednostkowe) |
| `scripts/plgrid/smoke_test.py` | test dymny; działa też na CPU (`--device cpu`), tak używają go testy |
| `scripts/download_base_models.py` | pobieranie wag do cache Hugging Face (węzeł logowania) |
| `scripts/merge_shards.py` | scalanie wyników zadań tablicy |
| `scripts/make_demo_dataset.py` | syntetyczne kafle z anotacjami, do prób bez danych |

## 6. Czego ten runbook nie obejmuje

- Preprocessing i `tiles.parquet` (Krzysztof), selektor, paczki QuPath i menedżer zadań: osobne komponenty.
- Rozproszone uczenie na wielu GPU: głowica jest mała i trenuje się na jednym.
- Kontener (Apptainer): nie jest potrzebny, dopóki wystarcza venv na `$SCRATCH`.
