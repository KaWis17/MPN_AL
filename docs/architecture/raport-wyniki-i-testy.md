# Raport: wyniki eksperymentów i testów (23.11.2026)

## 1. Eksperyment zastępczy BaMBo, 48 przebiegów

Siatka `configs/experiments/bambo_proxy.yaml`: 4 zamrożone enkodery SSL × 4 ułamki danych treningowych (0,10 / 0,25 / 0,50 / 1,00) × 3 seedy, segmentacja w natywnej skali, metryka PQ na podziale testowym (pseudo-etykiety jąder, nie ręczne adnotacje).

| Model | PQ @10% | @25% | @50% | @100% | średnia |
|---|---|---|---|---|---|
| lunit-dino-s16 | 0,522 | 0,543 | 0,552 | 0,547 | **0,541** |
| phikon-v2 | 0,517 | 0,524 | 0,538 | 0,539 | 0,530 |
| midnight-12k | 0,499 | 0,518 | 0,525 | 0,521 | 0,516 |
| dinov2-s14 (referencja) | 0,510 | 0,500 | 0,497 | 0,490 | 0,499 |

Odchylenie standardowe PQ między seedami: 0,003–0,024.

Parowe porównanie z dinov2-s14 (12 par, poprawka Holma):

| Model | średnia różnica PQ | p (Holm) |
|---|---|---|
| lunit-dino-s16 | +0,042 | 0,0029 |
| phikon-v2 | +0,031 | 0,0029 |
| midnight-12k | +0,017 | 0,034 |

Obserwacje:
- Wszystkie trzy modele histopatologiczne są istotnie lepsze od ogólnego DINOv2, ale różnice są małe (0,02–0,04 PQ).
- Modele histopatologiczne zyskują do ok. 50% danych, potem plateau. DINOv2 nie zyskuje wcale, jego PQ spada z 0,510 do 0,490 (rozrzut między seedami do 0,019, więc trend jest słaby).
- DQ (0,60–0,67) jest wąskim gardłem, SQ jest stabilne (0,80–0,83).
- Zastrzeżenie: etykiety są pseudo-etykietami jąder, więc wynik mówi o zgodności z nimi, nie o jakości względem ręcznej adnotacji.

## 2. Powtórzenie (job 3573579, nowy katalog `exp_bambo_rerun`)

- Ta sama siatka bez zmian, bez `--skip-existing`; 48/48 komórek ma `result.json`, żadne zadanie nie padło. Wyniki: `outputs/plgrid/exp_bambo_rerun/bambo_proxy`.
- Porównanie ze starym przebiegiem (job 3567632, `outputs/plgrid/exp_bambo_final`): różnica PQ dla każdej z 48 par wynosi dokładnie 0,0000 (próg zgłaszania 0,02 nieprzekroczony). Różnią się tylko czasy treningu (średnio ok. 7 s), więc to były świeże wykonania.
- Wniosek: pipeline jest deterministyczny w tej konfiguracji (seedy, podział, głowa, ewaluacja); niedeterminizm GPU nie wpływa na PQ. Szczegóły w `docs/architecture/experiment-plan.md`.
- Uwaga operacyjna: kolejka GPU była blokowana przez ok. dobę (ReqNodeNotAvail); stary job 3568570 (REQUEUE_HELD, wizualizacja) pozostawiony nietknięty.

## 3. Testy lokalne (pytest, Python 3.10.11, bez GPU)

**Z `timm` (venv na systemowym Pythonie, timm 1.0.28 z lokalnego cache uv, `HF_HUB_OFFLINE=1`): 132 zaliczone, 0 niezaliczonych (3 min 26 s).** Czyli wszystkie porażki z poniższego przebiegu wynikały z braku `timm`, nie z kodu. Testy nie obejmują GPU ani prawdziwych wag modeli (używają małych losowych ViT).

Poprzedni przebieg, bez `timm` w środowisku systemowym:

Wynik: **94 zaliczone, 18 niezaliczonych, 20 błędów** (2 min).

- Przyczyna: w lokalnym środowisku nie ma pakietu `timm` (35 z błędów to `ModuleNotFoundError: No module named 'timm'`). Dotyczy to testów ładujących model (`test_base_models`, `test_training`, `test_cli`, `test_smoke`, `test_experiment`, część `test_cluster_tools` i `test_configs`).
- Pozostałe asercje (`assert 1 == 0`, brak etapu `segment` w raporcie smoke) wynikają z tego samego: etap `load` modelu timm się nie powiódł. Nie badałem każdego z osobna, więc nie wykluczam, że część ma inną przyczynę.
- Nie jest to dowód błędu w kodzie: ten sam kod wykonał 48 przebiegów na klastrze, gdzie `timm` jest zainstalowany. Natomiast **testy zależne od modeli nie zostały lokalnie zweryfikowane**; pełny zestaw trzeba uruchomić w środowisku z `timm` (klaster lub lokalne venv z timm), żeby uznać je za zielone.
- Zaliczone 94 testy obejmują m.in. agregację, postprocessing, konfiguracje, okna i kafle, skrypty PLGrid.

## 4. Co nie przeszło / ograniczenia

- Testy zależne od `timm`: nie uruchomione sensownie lokalnie (patrz wyżej).
- Wynik eksperymentu dotyczy pseudo-etykiet BaMBo, nie docelowych danych projektu.
- Powtórzenie potwierdza odtwarzalność na tym samym sprzęcie i kodzie, nie niezależność od seedów (seedy te same).
