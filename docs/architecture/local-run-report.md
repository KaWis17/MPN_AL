# Raport: które modele da się uruchomić lokalnie i ile to potrwa

Stan na 22.11.2026. Dotyczy laptopa projektu (AMD Ryzen AI 9 365, 10 rdzeni / 20 wątków, 31 GB RAM, RTX 5060 Laptop 8 GB,
Windows 11). Kontekst: [ssl-base-models.md](ssl-base-models.md), [experiment-plan.md](experiment-plan.md),
[model-input-report.md](model-input-report.md), czasy na klastrze w ssl-base-models.md, sekcja 10.

## Warunki i skąd są liczby

- **Lokalnie jest tylko CPU.** Zainstalowany torch jest buildem CPU, więc GPU laptopa nie jest używane (wariant z GPU niżej).
- **Zmierzone** 22.11.2026 na tym CPU (10 wątków torch, fp32, losowe wagi: szybkość nie zależy od wag; składnia głowicy
  `base_width=32` jak w eksperymencie BaMBo): `lunit-dino-s16`, `dinov2-s14`, `phikon-v2`. Skrypty pomiarowe w katalogu
  roboczym sesji, nie w repozytorium.
- **Oszacowane** (oznaczone `~`): `h0-mini` (ViT-B/14) i `midnight-12k` (ViT-g/14) nie zbudowały się offline
  (pierwszy jest bramkowany, drugi potrzebuje konfiguracji z huba), więc liczby wzięto z interpolacji i ekstrapolacji po liczbie
  parametrów między zmierzonymi punktami (ViT-S 22 mln, ViT-L 303 mln) i z proporcji z A100 (ViT-g jest 4,4 razy wolniejszy od ViT-L).
  Błąd takiego szacunku może sięgać ±50%, dla ViT-g bywa większy. Modele bramkowane i tak nie ruszą, dopóki nie ma dostępu na HF.
- Czasy dotyczą samej sieci. **Nie zawierają** wczytywania obrazów, dekodowania instancji, wielokątów i zapisu
  (wcześniejszy pomiar całości dla Lunit: 14 s na obraz BaMBo 1600×1200 na CPU, czyli kilka przebiegów po 1024 px plus dekodowanie).
- Zwykłe wahania: laptop przy długim obciążeniu spowalnia się termicznie; przy kilkugodzinnych zadaniach zakładaj +20–30%.

## Zmierzone czasy jednostkowe (CPU, ten laptop)

| | `lunit-dino-s16` (ViT-S/16, 22 mln) | `dinov2-s14` (ViT-S/14, 22 mln) | `phikon-v2` (ViT-L/16, 303 mln) |
|---|---|---|---|
| segmentacja kafla 1024 px przy 0,25 µm/px (jedno przejście) | 4,4 s | 4,0 s | 15,3 s |
| to samo przy 0,5 µm/px (kafel 512 px) | 1,1 s | 0,7 s | 3,4 s |
| krok treningu głowicy (batch 8, przycięcie 256 px) | 3,4 s | 3,4 s | 8,5 s |
| embedding kafla, tryb `grid` (9 okien 224 px) | 0,38 s | 0,37 s | 3,8 s |

Dla porównania A100 (fp16, ssl-base-models.md): segmentacja kafla 1024 px 0,40–0,47 s, trening 60 kroków 6,2–9,6 s, czyli
laptop na CPU jest około 10–30 razy wolniejszy.

## Oszacowania dla pozostałych architektur (`~`, patrz ostrzeżenie wyżej)

| architektura (przykład) | kafel 1024 px, 0,25 µm/px | kafel 0,5 µm/px | krok treningu | embedding kafla (9 okien) |
|---|---|---|---|---|
| ViT-B/14 (`h0-mini`, `phikon`) | ~7 s | ~1,7 s | ~4,5 s | ~1,2 s |
| ViT-H/14 (`uni2-h`, `virchow`, `virchow2`) | ~28 s | ~6 s | ~14 s | ~8 s |
| ViT-g/14 (`midnight-12k`, `h-optimus-0/1`, `prov-gigapath`) | ~55 s | ~12 s | ~26 s | ~17 s |

ViT-L/14 (`hibou-l`) i ViT-L/16 (`uni`) mają koszt `phikon-v2`. Pamięć RAM nie jest ograniczeniem: ViT-g w fp32 to około 4,6 GB.

## Ile godzin zajmują typowe zadania

Liczone z powyższych czasów na jednostkę; bez wczytywania obrazów i dekodowania.

**1. Embeddingi kafli, 1000 kafli** (tryb `grid`):

| ViT-S | ViT-B (~) | ViT-L | ViT-H (~) | ViT-g (~) |
|---|---|---|---|---|
| 6 min | 20 min | 1,0 h | 2,2 h | 4,7 h |

**2. Segmentacja gotową głowicą, 1000 kafli 1024 px:**

| rozdzielczość robocza | ViT-S | ViT-B (~) | ViT-L | ViT-H (~) | ViT-g (~) |
|---|---|---|---|---|---|
| 0,5 µm/px | 0,2–0,3 h | 0,5 h | 0,95 h | 1,7 h | 3,2 h |
| 0,25 µm/px | 1,1–1,2 h | 1,9 h | 4,3 h | 7,8 h | 15 h |
| 0,25 µm/px z marginesem 128 px (×1,56 pikseli) | 1,7–1,9 h | 3 h | 6,6 h | 12 h | 24 h |

Dekodowanie instancji i zapis dodają kilka sekund na kafel, jeśli kafel ma tysiące komórek (nie mierzono na prawdziwych kaflach).

**3. Trening jednej głowicy** (konfiguracja eksperymentu BaMBo: 10 epok × 50 kroków = 500 kroków, batch 8, przycięcie 256 px,
bez walidacji w trakcie):

| ViT-S | ViT-B (~) | ViT-L | ViT-H (~) | ViT-g (~) |
|---|---|---|---|---|
| 0,5 h | 0,6 h | 1,2 h | 1,9 h | 3,6 h |

Do tego walidacja po każdej epoce i ewaluacja końcowa: na obrazach BaMBo ok. 14 s na obraz dla ViT-S (zmierzone wcześniej), więc
ewaluacja kilkudziesięciu obrazów to dodatkowe 10 min do godziny na przebieg.

**4. Cała siatka eksperymentu zastępczego BaMBo (48 przebiegów: 4 modele × 4 odsetki × 3 seedy):**

| | czas lokalnie na CPU | na A100 (zmierzone) |
|---|---|---|
| ViT-S ×2 modele (24 przebiegi) | około 12–20 h | około 2–3 h razem |
| `phikon-v2` (12 przebiegów) | około 15–25 h | około 1,2 h |
| `midnight-12k` (12 przebiegów) | około 45–60 h (szacunek) | około 1,4 h |
| razem | **około 70–100 h (3–4 doby ciągłej pracy)** | około 5 h (6–7 min na przebieg) |

Lokalna siatka tego rozmiaru nie ma sensu, jeśli klaster działa.

## Rekomendacje, co uruchamiać lokalnie

| model | lokalnie (CPU) | uwagi |
|---|---|---|
| `lunit-dino-s16`, `dinov2-s14` (ViT-S) | **tak, wszystko**: embeddingi, segmentacja, trening, pilotaż kilku przebiegów | wagi lokalnie tylko dla Lunit (87 MB); dinov2 trzeba pobrać (bez bramki) |
| `phikon-v2` (ViT-L/16) | **tak, w ograniczonym zakresie**: embeddingi, segmentacja przy 0,5 µm/px, pojedynczy trening (ok. 1,2 h) | wagi są lokalnie (1,2 GB); segmentacja przy 0,25 µm/px to ok. 4 h na 1000 kafli |
| `h0-mini`, `phikon`, ViT-B | tak, jeśli będzie dostęp (`h0-mini` jest bramkowany) | koszt między ViT-S a ViT-L |
| `uni`, `hibou-l` (ViT-L) | tak jak `phikon-v2` | bramkowane (hibou-l bramkowany wg rejestru) |
| `uni2-h`, `virchow`, `virchow2` (ViT-H) | tylko embeddingi małych zbiorów i segmentacja przy 0,5 µm/px; bez siatek | bramkowane |
| `midnight-12k`, `h-optimus-0/1`, `prov-gigapath` (ViT-g) | **nie do treningu ani siatek**; embeddingi lub segmentacja małej liczby kafli przy 0,5 µm/px (do ok. 100 kafli: ~20 min) | `midnight-12k` wymaga pobrania 4,2 GB; reszta bramkowana |

Sensowny podział pracy: lokalnie (CPU) **debug, wizualizacje, pilotaż na 1–2 modelach ViT-S i pojedyncze przebiegi ViT-L**; siatki, ViT-g
i wszystko, co ma wiele seedów, na Atenie (A100).

## Wariant z GPU laptopa (RTX 5060, 8 GB): niezmierzony

Obecny torch jest wyłącznie CPU. Instalacja torcha z CUDA (nowszy build z obsługą kart Blackwell; paczka kilka GB, wymaga pobrania) mogłaby
przyspieszyć sieć o rząd wielkości względem CPU, ale **nie sprawdzano** ani zgodności, ani szybkości: A100 daje ~10–30× względem
CPU, a karta laptopowa jest znacznie słabsza, więc przyspieszenie 3–10× jest tylko rozsądną spodziewaną wartością, nie wynikiem. Pamięć 8 GB
wystarcza na ViT-S/B/L (na A100: do 3,5 GB przy treningu ViT-L) i prawdopodobnie na inferencję ViT-g w fp16 (wagi ok. 2,3 GB; na A100 4,6 GB
przy fp32), a trening ViT-g (6,7 GB na A100) byłby na granicy. Warto to sprawdzić tylko, jeśli kolejka na Atenie nadal będzie blokować pracę; wymaga
Twojej zgody na pobranie paczki.

## Wniosek

Bez GPU lokalnie sensownie robi się **ViT-S (`lunit-dino-s16`, `dinov2-s14`) w całości i `phikon-v2` punktowo**; model ViT-g i siatki
z wieloma seedami należą na klaster. Orientacyjnie: pilotaż jednego modelu ViT-S na prawdziwych anotacjach to kilka godzin, pełna siatka
z czterema modelami to 3–4 doby ciągłego liczenia (wobec ok. 5 godzin na A100).
