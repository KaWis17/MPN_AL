# Raport: jak podawać wejście do porównywanych modeli SSL

Dotyczy: `scripts/segment_cells.py`, `scripts/extract_embeddings.py`, `scripts/train_segmentation_head.py` i
porównania wizualnego `scripts/compare_models_visual.py`. Stan na 21.11.2026. Kontekst: [ssl-base-models.md](ssl-base-models.md),
[data-layout.md](data-layout.md), [experiment-plan.md](experiment-plan.md).

## Odpowiedź w skrócie

1. **Do modelu trafia sam kafel 1024 × 1024 px (anotowane FOV), bez sąsiadów i bez dopełnienia.** Tak działa
   cały obecny kod: trening głowicy, segmentacja i embeddingi. Nic nie trzeba uzupełniać, żeby to uruchomić.
2. **Wszystkie porównywane modele dostają to samo wejście** (ten sam kafel, ta sama rozdzielczość robocza);
   różni je tylko to, co zrobi z nim adapter: rozmiar wejścia enkodera zależy od rozmiaru patcha (14 albo 16) i
   rozdzielczości roboczej (tabela niżej). To jedyny dopuszczalny sposób porównania „enkoder przeciw enkoderowi”.
3. **Sąsiednie kafle są potrzebne tylko do jednej rzeczy: jakości przy krawędzi kafla** (komórki przecięte
   krawędzią, brak kontekstu dla attention). Zmierzony efekt (ok. 0,1 PQ w pasie 50 px) jest w sekcji „Pomiar wpływu kontekstu”.
   Do treningu sąsiedzi nie są potrzebni, bo nie mają anotacji.
4. Niezależnie od wyniku: komórki dotykające krawędzi kafla są oznaczane (`touches_border`), a przy eksporcie
   do QuPath sąsiednich kafli komórkę z krawędzi trzeba zdeduplikować.

## Co dokładnie dostaje model

Kafel anotacji jest zawsze 1024 × 1024 px na poziomie 0 (około 0,25 µm/px przy 40×, czyli 256 × 256 µm).
Enkodery SSL były trenowane na 224 px przy około 0,5 µm/px (patrz kolumna `input` w `list_base_models.py`),
czyli na polu widzenia około 112 × 112 µm. Kafel jest więc większy niż wejście, na którym enkoder się uczył.

### Segmentacja (głowica CellViT-podobna)

| Etap | Co się dzieje | Gdzie w kodzie |
|---|---|---|
| 1 | Kafel jest przeskalowany do rozdzielczości roboczej głowicy `seg_mpp` (bez wartości = własna rozdzielczość kafla, ok. 0,25 µm/px) i zaokrąglony do wielokrotności 16 | `segmenter/model.py: working_size` |
| 2 | Cały obraz idzie do sieci **jednym przebiegiem**, jeśli jego bok ≤ `max_size` (1024). Większe obrazy: nakładające się przycięcia 1024 px (zakładka 128 px) łączone oknem ważonym | `predict_working_maps` |
| 3 | Dla patcha 16 enkoder dostaje obraz bez zmian; dla patcha 14 obraz jest skalowany do `H/16·14`, tak żeby siatka tokenów była taka sama (`H/16 × W/16`) | `CellSegmenter.forward` |
| 4 | Enkoder zwraca 4 warstwy ukryte, dekoder wzbogaca je o obraz i zwraca mapy `np`, `hv`, `nt`; HV → instancje → wielokąty GeoJSON | `decoder.py`, `postprocess.py` |

Wejście enkodera dla kafla 1024 × 1024 px przy 0,25 µm/px:

| `seg_mpp` | rozmiar roboczy | patch 16 (Lunit, Phikon, Phikon-v2, UNI, Prov-GigaPath) | patch 14 (DINOv2, H-optimus-0/1, Midnight, UNI2-h, Virchow/2, H0-mini, Hibou-L) | siatka tokenów | przebiegi |
|---|---|---|---|---|---|
| brak (0,25 µm/px) | 1024 | 1024 × 1024 | 896 × 896 | 64 × 64 | 1 |
| 0,5 µm/px | 512 | 512 × 512 | 448 × 448 | 32 × 32 | 1 |

(Poprawna lista modeli po rozmiarze patcha: `python scripts/list_base_models.py`, kolumna `architecture`.)

Konsekwencje dla porównania modeli:

- Przy `seg_mpp` bez wartości enkoder widzi komórki w **dwa razy większym powiększeniu** niż w pretreningu i
  przy sekwencji 4096 tokenów. Embeddingi pozycyjne są interpolowane przez adapter (timm: dynamiczny rozmiar,
  transformers: `interpolate_pos_encoding`). To działa, ale jest odejściem od rozkładu pretreningu, więc
  rozdzielczość robocza jest **parametrem porównania** (hipoteza H3), a nie szczegółem implementacji.
- Wszystkie modele w jednym porównaniu muszą mieć tę samą wartość `seg_mpp` i ten sam kafel. Różny rozmiar
  patcha jest kompensowany w adapterze (punkt 3), więc dekoder zawsze widzi siatkę tej samej wielkości.
- Normalizacja kolorów (średnia/odchylenie) jest własna dla każdego modelu i stosowana wewnątrz adaptera
  (`base.normalize`), więc użytkownik zawsze podaje obraz RGB w `[0, 1]`.

### Embeddingi kafla i komórek

| Tryb okna | Wejście enkodera | Zastosowanie |
|---|---|---|
| `grid` (domyślny) | 3 × 3 okna po 448 px poziomu 0, każde zmniejszone do 224 px (0,5 µm/px), wektor kafla = średnia | embedding kafla zgodny z pretreningiem (rozkład wejścia jak przy treningu enkodera) |
| `center` | jedno okno 448 px w środku | najtańszy |
| `full` | cały kafel w jednym przebiegu przy 0,5 µm/px: 518 px (patch 14) albo 512 px (patch 16) | spójny kontekst kafla; inny rozkład niż pretrening |
| `thumbnail` | cały kafel zgnieciony do 224 px | najgrubszy |

Wszystkie okna leżą **wewnątrz kafla**. Wektor komórki to tokeny z segmentacji uśrednione pod maską komórki
(`cell_features.py`), czyli ma ten sam kontekst co segmentacja.

### Trening głowicy

Kafel jest raz skalowany do rozdzielczości roboczej i trzymany w pamięci; w treningu losowane są przycięcia
(domyślnie 256 px) z przerzutami i zmianą koloru. Przycięcia są **mniejsze niż kafel**, więc w treningu enkoder
widzi mniej kontekstu niż w inferencji na całym kaflu. To zamierzone (pamięć, więcej wariantów), ale znaczy, że
głowica jest już przyzwyczajona do „braku kontekstu” i nie wymaga sąsiadów w treningu.

## Czy trzeba dodać sąsiednie kafle?

| Pytanie | Odpowiedź |
|---|---|
| Czy sąsiedzi są potrzebni do uruchomienia? | Nie. |
| Czy są potrzebni do **treningu**? | Nie. Sąsiedzi nie mają anotacji; dodanie ich wymagałoby maski `ignore` poza kaflem, bez korzyści dla etykiet. |
| Czy są potrzebni do **porównania modeli**? | Nie, jeśli wszystkie dostają ten sam kafel. Kontekst daje podobny efekt wszystkim czterem zmierzonym modelom; wpływu na ranking nie testowano. |
| Czy poprawiają **predykcje przy krawędzi**? | Tak, w pasie około 50 px od krawędzi (PQ 0,78 wobec 0,88 w głębi); patrz sekcja z pomiarem. |
| Jak je dodać? | `segment_cells.py --context-margin M` (np. 128 px na bok, kafle z WSI): czyta region `(x−M, y−M, w+2M, h+2M)`, segmentuje całość i zostawia komórki ze środkiem w kaflu. Sam pomiar efektu: `compare_models_visual.py --context-margin`. |

## Pomiar wpływu kontekstu (21–22.11.2026, A100, job 3573478)

**Co zmierzono.** Cztery głowice z eksperymentu BaMBo (`dinov2-s14`, `lunit-dino-s16`, `phikon-v2`, `midnight-12k`,
100% obrazów, seed 0) na 9 obrazach testowych, po jednym na pacjenta (wyniki w `outputs/plgrid/visual_bambo/`,
obrazki `<tile_id>_compare.png` i `_context.png`). Z każdego obrazu wycięto środek (margines 128 px = 64 µm z każdej
strony) i segmentowano go dwa razy: samego i jako część całego obrazu. Porównanie: PQ jednego wyniku względem
drugiego, osobno w pasie 48 px przy krawędzi środka i głębiej.

**Zastrzeżenia (ważne przy czytaniu liczb).**
- To nie są prawdziwe sąsiednie kafle, tylko przycięcie obrazu: dolne oszacowanie efektu, bo obrazy BaMBo nie mają kontekstu poza kadrem.
- Obrazy BaMBo mają różne rozmiary (89 × 1600×1200, 70 × 1920×1200, 44 × 512×512, 1 × 1461×913), więc „sam środek” i „cały obraz” są też
  różnie dzielone na przebiegi po 1024 px. Część różnicy w głębi to efekt tego podziału, nie kontekstu. Liczby są więc górnym oszacowaniem
  efektu samego kontekstu w głębi i mieszanką w pasie przy krawędzi.
- Etykiety odniesienia to pseudo-etykiety (jądra z baseline'u barwnikowego), głowice są niestrojone, jeden podział, 9 obrazów.
  Nie ma tu przedziałów ufności.

**Wynik, 7 typowych obrazów** (bez dwóch opisanych niżej; średnia po obrazach):

| model | PQ cały środek | PQ w pasie 48 px od krawędzi | PQ głębiej |
|---|---|---|---|
| dinov2-s14 | 0,87 | 0,76 | 0,88 |
| lunit-dino-s16 | 0,90 | 0,80 | 0,90 |
| phikon-v2 | 0,89 | 0,80 | 0,89 |
| midnight-12k | 0,87 | 0,78 | 0,87 |

- Przy krawędzi wynik z kontekstem i bez zgadza się o około 0,1 PQ gorzej niż w głębi (0,78 wobec 0,88 średnio), podobnie dla wszystkich modeli.
  Efekt dotyczy więc pasa rzędu 50 px (25 µm przy 0,5 µm/px), a nie całego kafla. Megakariocyty (kilkadziesiąt µm) mogą być przy brzegu cięte, tego tu nie mierzono.
- Liczba komórek stykających się z krawędzią jest po segmentacji z kontekstem średnio 1,26 razy większa, więc samodzielny kafel gubi część komórek
  przy brzegu albo je skraca.
- Różnica prawdopodobieństwa komórki spada z około 0,015 w pasie 0–16 px do około 0,006 po 256 px; zasięg wpływu jest rzędu 100 px.
- Rozmiar efektu jest podobny u wszystkich czterech modeli, więc **kontekst nie zmienia rankingu enkoderów** w tym teście. To nie jest wynik o rankingu
  jako takim: nie porównywano go z kontekstem i bez niego.

**Dwa obrazy odstające.** `PGI_11__PGI_image_162` (1920×1200) i `PGI_40_TEST__2` (512×512): wynik zmienia się mocno także w głębi
(PQ 0,27–0,52 oraz 0,04–0,74 zależnie od modelu; liczba komórek `lunit-dino-s16` na drugim: 207 samodzielnie, 97 z kontekstem). Tu segmentacja jest
niestabilna, a nie tylko pozbawiona kontekstu na brzegu. Prawdopodobnie obrazy spoza rozkładu (inna skala lub barwienie; 512×512 to inny typ obrazów BaMBo),
ale przyczyny nie zbadano. Wniosek praktyczny: wynik głowicy na obrazie o nietypowym wyglądzie trzeba oglądać, a nie ufać średniej.

**Zgodność modeli między sobą** (średnie PQ para-do-pary): 0,59–0,68, czyli modele często różnią się obrysami nawet na typowym obrazie; dla obrazów
1600×1200 średnio 0,76, dla 1920×1200 0,55 (zaniżone przez obraz odstający), dla 512×512 0,44. Jakość tych różnic (który model ma rację) widać dopiero
na prawdziwych anotacjach.

**Zalecenie (decyzja do podjęcia z zespołem).**
1. Do **porównania modeli i treningu**: sam kafel 1024 × 1024 px, bez sąsiadów, jednakowy dla wszystkich (to, co jest teraz).
2. Do **produkcyjnej predykcji dla pętli AL** (QuPath): segmentować kafel z marginesem 64 px, a lepiej 128 px, i zachować komórki, których środek leży w kaflu.
   Koszt: kafel 1024 + 2 × 128 = 1280 px, czyli 1,56 raza więcej pikseli. Powinno to zmniejszyć różnicę przy krawędzi (nie sprawdzono) i unika duplikatów między sąsiednimi kaflami.
   **Zaimplementowane**: `segment_cells.py --tiles-parquet ... --context-margin 128` (czyta kafel z marginesem z WSI, segmentuje całość i zostawia komórki
   ze środkiem w kaflu; `touches_border` znaczy wtedy „przekracza krawędź kafla”, wielokąty nie są przycinane). Działa tylko dla kafli z WSI, nie dla plików PNG.
   Przetestowane na sztucznym slajdzie (kafle stykające się nie raportują wspólnych komórek); **nie sprawdzone** na prawdziwym NDPI ani to, o ile margines poprawia wynik na brzegu.
3. Jeśli kafle są sąsiadujące i ich wyniki scalane, margines i tak jest potrzebny do deduplikacji komórek na styku.

### Porównanie na 9 obrazach z powiększeniem (24.11.2026, job 3575305, A100)

Te same cztery głowice (`f100__s0`, z `exp_bambo_rerun`) i te same 9 obrazów co wyżej, z nowymi `_zoom.png` i `reference_pq.csv`
(`outputs/plgrid/visual_bambo_zoom/`). PQ względem pseudo-etykiet, cały obraz, średnia z 9 obrazów:

| model | PQ | DQ | SQ |
|---|---|---|---|
| phikon-v2 | 0,485 | 0,607 | 0,767 |
| lunit-dino-s16 | 0,482 | 0,597 | 0,770 |
| midnight-12k | 0,440 | 0,552 | 0,757 |
| dinov2-s14 | 0,420 | 0,534 | 0,744 |

- Kolejność modeli zgadza się z eksperymentem na 48 przebiegach (histopatologiczne przed DINOv2), ale na 9 obrazach i jednym seedzie
  różnice między phikon-v2 i lunit-dino-s16 (0,003) oraz między modelami na pojedynczym obrazie (np. 0,47–0,51 na `PGI_2`) są w granicach szumu.
- Na 7 typowych obrazach PQ to 0,26–0,78, a na dwóch odstających (`PGI_11__PGI_image_162`, `PGI_40_TEST__2`) 0,01–0,09 u wszystkich modeli;
  liczba komórek tam waha się 2–3 razy między modelami (np. 836 vs 2639 na `PGI_11`). Potwierdza to wcześniejszy wniosek, że na obrazach spoza rozkładu
  trzeba oglądać wynik, a nie ufać średniej; przyczyny nie zbadano. Średnie powyżej są przez te dwa obrazy zaniżone.
- Na wycinku z `PGI_10` (jeden obraz, więc to obserwacja, nie wniosek) wszystkie modele znajdują podobny zestaw komórek i różnią się głównie
  rozdzieleniem sąsiednich jąder i obrysem największych, ciemnych komórek; różnica PQ (0,59–0,70) wynika głównie z DQ.
- Ograniczenia: pseudo-etykiety (jądra z baseline'u barwnikowego), jeden seed, głowice niestrojone, brak przedziałów ufności.

## Jak to zobaczyć: porównanie wizualne

```bash
python scripts/compare_models_visual.py \
    --head dinov2-s14=<run>/head/head.pt --head lunit-dino-s16=<run>/head/head.pt \
    --images tiles/ --mpp 0.25 --limit 6 --masks masks/ --context-margin 128 --out outputs/visual
```

| Plik | Zawartość |
|---|---|
| `<tile_id>_compare.png` | wiersz na model: kontury komórek na obrazie, prawdopodobieństwo komórki, mapy HV, PCA tokenów enkodera; pierwszy wiersz to obraz i maska odniesienia |
| `<tile_id>_zoom.png` | (dodane 23.11.2026) ten sam, najgęstszy fragment obrazu (`--zoom 256` px, `0` wyłącza) powiększony bez interpolacji: obraz \| referencja \| jeden panel z obrysami na model, z PQ modelu względem referencji w podpisie (przy `--masks`); miniatury 360 px z `_compare.png` nie pozwalają ocenić pojedynczych obrysów |
| `reference_pq.csv` | (przy `--masks`) PQ, DQ, SQ i liczba komórek każdego modelu względem referencji, per obraz; na konsoli średnie po obrazach |
| `<tile_id>_context.png` | środek obrazu segmentowany samodzielnie i jako część większego obrazu, plus mapa różnic |
| `geometry.csv` | co enkoder faktycznie dostaje (rozmiar roboczy, wejście enkodera, siatka tokenów, liczba przebiegów) |
| `agreement.csv` | PQ między parami modeli na tym samym kaflu (zgodność modeli, nie jakość) |
| `context_effect.csv`, `context_summary.csv` | efekt kontekstu: PQ przy krawędzi i wewnątrz, różnica prawdopodobieństwa wg odległości od krawędzi |

Jak czytać panele: **PCA tokenów** pokazuje, co enkoder rozróżnia, zanim dekoder cokolwiek zrobi (komponenty są
dopasowane do jednego obrazu, więc kolory wolno porównywać tylko wewnątrz panelu); **HV** powinny zmieniać
znak między stykającymi się komórkami; rozjazd konturów dwóch modeli na tym samym kaflu to miejsca, gdzie wybór
enkodera zmienia wynik.
