# Porównanie embeddingów z różnych modeli SSL (klastrowanie)

Plan z 24.11.2026 i stan wdrożenia. Kontekst: [ssl-base-models.md](ssl-base-models.md), sonda liniowa w `src/evaluation/probe.py`.

## Pytania i podejście

1. **Modele:** otwarte `lunit-dino-s16` (ViT-S/16), `phikon-v2` (ViT-L/16), `midnight-12k` (ViT-g/14), `dinov2-s14` (ogólny); po uzyskaniu dostępu
   `h0-mini`, `h-optimus-0/1`, `uni2-h`, `virchow2`. Kontrole: losowy enkoder tej samej architektury i cechy barwy H&E.
2. **Wejście i wymiary:** ten sam kafel 1024 px w trybie okien `grid`. Wymiary porównujemy dwutorowo: w przestrzeni modelu po normalizacji L2
   (miary oparte na sąsiedztwie nie wymagają równych wymiarów) oraz po PCA do wspólnego d = 50 (wariant 128); centrowanie po preparacie
   jako wariant (z nim i bez niego), bez wybielania.
3. **Klastrowanie:** KMeans (k-means++), k ∈ {2, 3, 4, 5, 6, 8, 10, 12, 16, 20}, 10 seedów (każdy `n_init=1`, żeby mierzyć wrażliwość
   na inicjalizację) oraz 20 powtórzeń z pominięciem 20% pacjentów (wrażliwość na dane).
4. **Metryki:** silhouette (kosinusowa), Davies-Bouldin, Calinski-Harabasz, inercja; stabilność (ARI/AMI między seedami, ARI pominiętych pacjentów);
   zgodność z etykietami (NMI, ARI, czystość oraz poziom przypadku dla przetasowanych etykiet); zależność od pacjenta lub preparatu (NMI, AMI);
   między modelami: ARI/AMI partycji, nakładanie sąsiedztw kNN, liniowe CKA.
5. **Wizualizacja:** UMAP kolorowany klastrem, etykietą i pacjentem; krzywe metryk po k; macierze zgodności modeli; galerie kafli na klaster;
   eksport klastrów do QuPath.

## Stan wdrożenia

| Element | Stan |
|---|---|
| `src/evaluation/clustering.py`: `preprocess`, `run_grid`, stabilność po seedach i po pacjentach, zgodność z etykietami, zależność od pacjenta | **gotowe**, 9 testów na danych z zaplanowaną strukturą (odzyskanie k, stabilność spada z szumem, efekt partii wykrywany i usuwany przez centrowanie) |
| `src/evaluation/embedding_compare.py`: ARI/AMI partycji, nakładanie sąsiedztw, liniowe CKA | **gotowe**, testy niezmienniczości (obrót, skala) |
| `scripts/cluster_embeddings.py` (wczytanie `embeddings.parquet` modeli, zgodne próbki, warianty centrowania i PCA, kontrola szumem, `metrics.csv`, `best_k.csv`, `clusters.parquet`, porównania modeli, `report.md`) | **gotowe**, testy na danych syntetycznych dwóch „modeli” o różnych wymiarach; uruchomione na prawdziwych embeddingach `phikon-v2` z BaMBo (jeden model) |
| `scripts/visualize_clusters.py` (krzywe metryk po k, mapy model × k, macierze ARI/CKA/kNN między modelami, mapa 2-D, skład klastrów, galerie kafli na klaster, GeoJSON dla QuPath, `index.html`) | **gotowe**, 7 testów; uruchomione na prawdziwych embeddingach `phikon-v2` z BaMBo (jeden model); wymaga `pip install -e ".[viz]"` |
| Uruchomienie na prawdziwych embeddingach (BaMBo, 4 modele otwarte) | do zrobienia |

## Pierwszy wynik na prawdziwych danych (24.11.2026, jeden model)

`phikon-v2`, 204 obrazy BaMBo (`outputs/plgrid/bambo/embeddings.parquet`), 47 pacjentów, `--nuisance-col image_size`, 5 seedów, 10 powtórzeń z pominięciem pacjentów:

- **Najbardziej stabilny podział (k = 2, ARI między seedami 1,0, silhouette 0,46) to rozmiar obrazu**: wszystkie 44 obrazy 512 × 512 w jednym klastrze,
  obrazy 1600 × 1200 i 1920 × 1200 w drugim (poza 4 wyjątkami). NMI klastrów z rozmiarem obrazu wynosi 0,5–0,58 przy każdym k (szum: 0,005).
  Stabilność sama nie dowodzi, że klastry mają sens biologiczny; BaMBo miesza obrazy o różnej skali (przyjęto 0,5 µm/px dla wszystkich).
- Kontrola szumem: stabilność i silhouette bliskie 0, więc metryki rozróżniają strukturę od przypadku.
- NMI z pacjentem rośnie z k (mierzy też samą liczbę klastrów); do porównań między k służy AMI (`ami_nuisance`).
- Galeria dla k = 4: klaster 1 to prawie wyłącznie obrazy 512 × 512 (widać na nich większe struktury, jak beleczki kostne, czyli inne powiększenie), pozostałe trzy różnią się na oko zawartością komórkową, ilością tłuszczu i kości.
  To pierwsze wrażenie z jednego przebiegu, nieweryfikowane.
- Tylko jeden model, więc nie ma jeszcze porównania między modelami. Wniosek dotyczy `phikon-v2` na BaMBo, nie innych modeli ani danych szpiku.

## Wizualizacje (`scripts/visualize_clusters.py`)

```bash
python scripts/visualize_clusters.py --cluster-dir outputs/clustering/bambo --embeddings phikon-v2=<tabela> --meta outputs/bambo_meta.csv \
    --images outputs/bambo_png --nuisance-col image_size --k 4 --variant l2
```

Tabele, `--meta` i kolumny muszą być takie jak w przebiegu klastrowania. Wynik w `<cluster-dir>/figures`:
`curves_<wariant>.png` (silhouette, ARI między seedami, ARI pominiętych pacjentów, AMI z pacjentem i ze zmiennymi zakłócającymi, ARI z etykietami; kontrola szumem przerywaną szarą linią),
`heatmap_*.png` (model × k), `matrix_ari.png`, `matrix_cka.png`, `matrix_knn*_overlap.png` (modele między sobą), `map_<model>.png` (UMAP, a gdy jest niezgodny z wersją scikit-learn lub go brak, t-SNE;
kolor: klaster, etykieta, zmienne zakłócające, pacjent), `composition_<model>.png` (z czego składa się klaster), `gallery_<model>.png` (kafle najbliższe środkowi klastra i losowi członkowie),
`qupath/<model>_k<k>/<slajd>.geojson` (kafle jako prostokąty z klasyfikacją `<model>_cluster_<n>`; tylko dla kafli, których id niesie położenie `slajd_x…_y…_s…`) i `index.html`.

- **Krzywe używają miar skorygowanych o przypadek** (AMI, ARI). NMI z pacjentem rośnie z k także dla szumu (przy k = 20 wynosi 0,47), więc samo NMI byłoby mylące.
- Kolory na mapach 2-D nie są porównywalne między modelami, a mapa pokazuje strukturę, nie odległości.
- Na BaMBo kafle nie mają położenia w slajdzie w identyfikatorze, więc plików QuPath nie powstaje; będą dla kafli z prawdziwego potoku (`tl_<slajd>_x…_y…_s1024`).

## Pierwsze porównanie dwóch modeli na BaMBo (25.11.2026)

`phikon-v2` (embeddingi z Atheny) i `lunit-dino-s16` (embeddingi policzone lokalnie na CPU, te same ustawienia: `--mpp 0.5`, tryb `grid`), 204 obrazy, 47 pacjentów (3–5 obrazów na pacjenta),
10 seedów, 20 powtórzeń z pominięciem 20% pacjentów, kontrola szumem. Wyniki w `outputs/clustering/bambo2/` (`metrics.csv`, `best_k.csv`, `compare_*.csv`, `figures/`).
Pozostałe dwa modele (`dinov2-s14`, `midnight-12k`) nie mają jeszcze embeddingów: joby ekstrakcji (3580205–3580207) anulowano 25.11.2026 przed startem.

- **Oba modele najsilniej dzielą dane według rozmiaru obrazu.** Przy k = 2 klaster to obrazy 512 × 512 kontra reszta (ARI między seedami 1,0 dla obu, silhouette 0,50 `lunit` i 0,46 `phikon`).
  AMI klastrów z rozmiarem obrazu wynosi 0,39–0,47 dla `lunit` i 0,46–0,57 dla `phikon` (zależnie od k; szum: 0,0). Na mapie `lunit` osobno leżą też obrazy 1920 × 1200 i 1600 × 1200.
  Różnica między modelami (`lunit` mniej zależny od rozmiaru) to jeden przebieg na 204 obrazach, bez przedziału ufności; nie rozstrzyga, który model jest lepszy.
- **Po centrowaniu po pacjencie** AMI z rozmiarem obrazu spada do 0,03–0,44 (`phikon`) i 0,04–0,34 (`lunit`) zależnie od k, ale tym samym znika też część prawdziwej różnicy między pacjentami
  (pacjenci mają po 3–5 obrazów, więc średnia pacjenta jest zaszumiona). Wariant jest pokazywany obok surowego, nie zamiast niego.
- **Stabilność przy k ≥ 4** jest podobna u obu modeli (ARI między seedami 0,62–0,75 (`phikon`) i 0,55–0,67 (`lunit`) w surowym wariancie, 0,46–0,54 po centrowaniu; szum 0,02–0,07), z szerokimi przedziałami: różnice między modelami nie są rozstrzygające.
- **Przestrzenie są podobne:** CKA 0,91 (surowe) i 0,83 (po centrowaniu) wobec 0,08–0,27 dla szumu; nakładanie sąsiedztw kNN (k = 10) 0,62 i 0,51 wobec 0,02. Klastrowania zgadzają się umiarkowanie
  (ARI 0,18–0,74 zależnie od k i wariantu, AMI 0,17–0,79), więc ten sam podział na kafle nie daje tego samego podziału na klastry.
- **Zastrzeżenia:** dwa modele, jeden zbiór BaMBo (bez klas komórek, więc bez zgodności z etykietami), embeddingi `lunit` z CPU w fp32, a `phikon` z innego przebiegu (nie porównywano
  numerycznie), skala przyjęta 0,5 µm/px dla wszystkich obrazów. Wniosek: na BaMBo struktura embeddingów jest w dużej mierze strukturą źródła obrazu; ocena modeli na prawdziwych anotacjach i skali dopiero przed nami.

## Ograniczenia, które trzeba pamiętać

- Wartości k i miary stabilności to konwencja: pokazujemy zakres, nie jedną liczbę.
- Davies-Bouldin i Calinski-Harabasz zależą od wymiaru i skali przestrzeni, więc służą do wyboru k w obrębie jednego modelu, nie do porównań między modelami.
- Silhouette w przestrzeniach o różnym wymiarze nie jest w pełni porównywalna; stąd dwa tory (pełna przestrzeń i wspólne PCA).
- BaMBo nie ma klas komórek, więc zgodność z etykietami ocenimy dopiero po anotacjach; do tego czasu tylko stabilność i zależność od pacjenta.
- Testy sprawdzają, czy kod odzyskuje zaplanowaną strukturę; nie mówią nic o tym, jakie klastry dadzą prawdziwe embeddingi.
