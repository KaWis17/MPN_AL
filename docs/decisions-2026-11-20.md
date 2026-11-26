# Decyzje do podjęcia: opcje, konsekwencje, rekomendacje (20.11.2026)

Uzupełnienie [raportu stanu](report-2026-11-20.md). Dla każdej decyzji: kontekst, wszystkie rozważane opcje z
zaletami i wadami, rekomendacja, wpływ na kod i koszt, kto decyduje, co odblokowuje oraz **co przyjmę domyślnie,
jeśli decyzji nie będzie** (żeby praca nie stała). Rekomendacje są moje i nie są uzgodnione z zespołem ani z
lekarzem. Fakty o zbiorach i spotkaniach pochodzą z notatek `meeting_*.txt`.

## Kolejność i zależności

```text
D1 tło/ignore ─┐
D2 obrys ──────┼─> D3 klasy ──> pilotaż ──> D6 skład siatki ──> pełna siatka ──> test (D5) ──> D14 kryterium wyboru
D9 krawędzie ──┘                  ^
D5 podział ───────────────────────┤        D4 rozdzielczość (mierzona w pilotażu)
D17 źródło danych ────────────────┘        D16 wariancja barwienia (ablacja)
Niezależnie: D7 licencje/wagi, D8 LoRA, D10 miara, D11 baseline'y, D12 miejsce danych, D13 drugi anotator
```

Minimalny zestaw na najbliższe spotkanie, bez którego eksperyment nie ruszy: **D1, D2, D3, D5, D17**. Reszta może
poczekać na wyniki pilotażu.

---

## D1. Nieoznaczone piksele w kaflu: tło czy `ignore`

**Kontekst.** `data-layout.md` mówi, że nieoznaczone piksele to `ignore`, nie tło. Głowica uczy się odróżniać komórkę od
tła; jeśli żaden piksel nie jest tłem, nie ma czego się uczyć (kod odmawia treningu bez tła, flagi `--unlabeled`
i klasa ROI to obsługują).

| Opcja | Opis | Za | Przeciw |
|---|---|---|---|
| A. Wszystko poza obrysami to tło | lekarz obrysowuje każdą komórkę w całym kaflu 1024×1024 | prosto, pełna informacja o tle | drogie, a pominięta komórka staje się błędnym „tłem” (karze trafne predykcje) |
| B. Wszystko poza obrysami to `ignore` | lekarz obrysowuje wybrane komórki | tanie | głowica nie widzi tła, więc nie nauczy się niczego użytecznego |
| C. ROI: tło tylko wewnątrz ROI | lekarz zaznacza obszar opisany w całości (klasa `ROI`), komórki w nim muszą być wszystkie | uczciwe, tańsze niż A, kod gotowy | ROI trzeba zaznaczać; małe ROI to mało danych o tle |
| D. Rundy ≥1: kafel to poprawiona propozycja modelu | lekarz akceptuje lub poprawia WSZYSTKIE propozycje na kaflu, więc po poprawie kafel jest kompletny; runda 0 wg C | najtańsze w pętli AL, tło wiarygodne | wymaga dyscypliny: „zaakceptowany” musi znaczyć „sprawdziłem cały kafel”; propozycja kotwiczy lekarza (pominie to, czego model nie zaproponował) |

**Rekomendacja:** D (runda 0 wg C). Dla rundy 0 wystarczy ROI na części kafli.
**Kod:** gotowy dla A, B, C (`--unlabeled`, klasa `ROI`). D wymaga oznaczania kafla jako „kompletny” w manifeście
anotacji (drobna zmiana).
**Kto:** zespół i dr Zduniak. **Odblokowuje:** każdy trening na prawdziwych danych.
**Domyślnie:** C, z założeniem że lekarz zaznacza ROI.

## D2. Co jest obrysem: jądro czy cała komórka

**Kontekst.** Zespół chce masek całych komórek, nie jąder (PathoSAM daje jądra). Lista klas QuPath zawiera `bare_nucleus`
i `cytoplasm`, co sugeruje, że cytoplazma bywa obrysowywana osobno od jądra.

| Opcja | Opis | Za | Przeciw |
|---|---|---|---|
| A. Obrys = cała komórka (jądro + cytoplazma) jednym wielokątem | jeden obiekt na komórkę | najprostsze, pasuje do instancji | megakariocyty są wielkie, a małe komórki mają mało cytoplazmy; wymaga zgody, jak rysować |
| B. Obrys = jądro | tylko jądra | łatwe, zgodne z gotowymi modelami (CellViT++, HistoPLUS) | nie spełnia celu (maski komórek); megakariocyt ma jądro wielopłatowe |
| C. Dwa poziomy: jądro i komórka jako osobne klasy lub osobne mapy | jądro (`bare_nucleus`) i cytoplazma/komórka | najwięcej informacji, można liczyć stosunek jądro/cytoplazma | podwaja pracę i zmienia głowicę (dodatkowa maska) |
| D. Jądro rysuje lekarz, komórkę wyprowadzamy | maska komórki z jądra (rozrost, ststain) | tanie | zadanie badawcze, jakość nieznana (wspominane 27.10: „co zrobić, żeby z maski jądra zrobić maskę komórki”) |

**Rekomendacja:** A dla megakariocytów (to one są celem), z jądrem tylko jako klasą pomocniczą, jeśli lekarz i tak je
rysuje.
**Kod:** przy A i C trzeba obsłużyć nakładające się wielokąty (dziś późniejszy zamalowuje wcześniejszy, więc
zagnieżdżone jądro w komórce zniszczyłoby maskę komórki); to ok. pół dnia pracy.
**Kto:** dr Zduniak (jak rysuje), zespół (co jest celem). **Odblokowuje:** definicję etykiet i metryk.
**Domyślnie:** A; nakładające się wielokąty łączę (unia), klasę bierze większy obiekt.

## D3. Zestaw klas i architektura klasyfikacji

**Kontekst.** Ustalenie z 10.11: najpierw „megakariocyt czy nie”, potem podtyp (`meg_norm`, `meg_atypical`,
`meg_proliferative`, `meg_dysplastic`). Do tego `bare_nucleus`, `dwarf`, `cytoplasm`, `uncertain`. Klasy są
niezrównoważone (małe komórki liczniejsze, megakariocyty wielkie i rzadkie). Diagnozy (PV, ET, PMF) zależą od
podtypów.

| Opcja | Opis | Za | Przeciw |
|---|---|---|---|
| A. Binarnie: megakariocyt / nie | jedna klasa w głowicy | start najszybszy, kod gotowy | nie daje podtypów |
| B. Dwa etapy | głowica znajduje megakariocyty; podtyp daje lekki klasyfikator na embeddingach komórek | pasuje do ustalenia 10.11, tanie dodawanie klas, mała liczba etykiet wystarcza | klasyfikator jeszcze niezaimplementowany; embeddingi zależą od preparatu |
| C. Wieloklasowa głowica | głowica przewiduje wszystkie klasy naraz | jedna sieć | przy rzadkich klasach niestabilna, potrzebuje wag klas |
| D. Hierarchia + C | binarna głowica + wieloklasowa na wyciętych komórkach | najlepsza jakość potencjalnie | najwięcej pracy |

Pytania w tle: czy `bare_nucleus` i `dwarf` to osobne klasy komórek, czy części komórki (D2); czy `uncertain` ma być
`ignore` (obecny szkic) czy klasą do rozpoznania.

**Rekomendacja:** A w pilotażu, B jako docelowe.
**Kod:** schemat YAML obsługuje dowolne klasy; B wymaga klasyfikatora na embeddingach (krok 17).
**Kto:** dr Zduniak i zespół. **Odblokowuje:** `megakaryocyte_binary.yaml` w wersji ostatecznej.
**Domyślnie:** A; `bare_nucleus`, `dwarf`, `cytoplasm`, `uncertain` pozostają `ignore`.

## D4. Rozdzielczość pracy głowicy

**Kontekst.** Kafel lekarza to 1024 px przy ok. 0,25 µm/px (40×); większość modeli czytała 224 px przy 0,5 µm/px;
Virchow2 widział także 0,25. Praca przy 0,5 jest ok. 4× tańsza.

| Opcja | Za | Przeciw |
|---|---|---|
| A. 0,25 µm/px (natywna) | najlepsze dla małych komórek i brzegów | 4× koszt; większość enkoderów nie widziała tej skali |
| B. 0,5 µm/px | tanio, zgodne z pretreningiem | małe komórki tracą szczegóły |
| C. Obie jako oś eksperymentu | odpowiedź z danych (H3) | podwaja siatkę |
| D. Wielorozdzielczość w jednym modelu | potencjalnie najlepsze | zmiana architektury, poza zakresem |

**Rekomendacja:** C w pilotażu (jest w `pilot.yaml`), w pełnej siatce zostawić zwycięską lub obie dla 2–3 modeli.
**Kto:** użytkownik, na podstawie pilotażu. **Wymaga:** `mpp` dla każdego kafla (BaMBo go nie zapisuje).
**Domyślnie:** C.

## D5. Podział po pacjentach i protokół oceny

**Kontekst.** Pacjent nie może występować w dwóch zbiorach (ustalenie 26.10 i 27.10). Liczba pacjentów będzie
mała, więc przedziały ufności będą szerokie. Zbiór testowy ma być otwierany raz.

| Opcja | Opis | Za | Przeciw |
|---|---|---|---|
| A. Jeden stały podział train/val/test losowy po pacjentach | proste | tanie, zgodne z `patient_split_fixed.csv` | przy małej liczbie pacjentów wynik zależy od losu |
| B. Stały podział stratyfikowany (po diagnozie PV/ET/PMF i częstości klas) | jak A, ale zbalansowane | porównywalność, ważne przy rzadkich klasach | wymaga danych klinicznych na etapie podziału |
| C. Walidacja krzyżowa po pacjentach (k-fold) do wyboru modelu, oddzielny stały test | lepsze wykorzystanie małych danych | k razy koszt | komplikuje siatkę |
| D. Test zewnętrzny (inny zbiór, np. publiczny z Zenodo/Mendeley lub BoMBR) | najuczciwsza ocena uogólnienia | inne barwienie i skala, rozbieżne etykiety | trudno porównać |

Decyzje towarzyszące: proporcje (np. 60/20/20), kto zamraża plik (Krzysztof, bo robi preprocessing i tiling), od kiedy
obowiązuje, czy kafle jednego pacjenta mogą mieć nakładanie (jeśli tak, podział po pacjencie to zabezpiecza).

**Rekomendacja:** B do pilotażu i wyboru modelu, C jeśli pacjentów jest poniżej ok. 15, plus D jako dodatkowy test.
**Kod:** runner czyta CSV lub `tiles.parquet`, odrzuca pacjentów w kilku zbiorach; C wymaga dopisania pętli po foldach.
**Kto:** zespół. **Domyślnie:** A z losowaniem z ustalonym ziarnem, aż pojawi się dane kliniczne.

## D6. Skład pełnej siatki enkoderów

**Kontekst.** Siatka rośnie iloczynem: modele × rozdzielczości × odsetki × seedy. Koszt poznamy z `smoke`.
Propozycja w `encoders.yaml`: 8 modeli × 2 × 4 × 3 = 192 komórki.

| Opcja | Za | Przeciw |
|---|---|---|
| A. Mała: 3–4 modele (`lunit`, `phikon-v2`, `h-optimus-0`, `uni2-h`) | tanio, szybko | może ominąć najlepszy |
| B. Pełna 8 modeli | rzetelne porównanie | koszt, dostęp do bramkowanych |
| C. Sukcesywne odcinanie | wszystkie modele na małym odsetku (10%) i 1 seedzie, najlepsze 3–4 na pełnym | oszczędza 70–80% obliczeń | wymaga stałej reguły odcięcia ustalonej z góry |
| D. Rozszerzona o `h-optimus-1`, `virchow`, `hibou-l`, `prov-gigapath`, `uni`, `phikon` | najpełniej | wiele dostępów i koszt |

**Rekomendacja:** C, z regułą odcięcia zapisaną przed startem (np. zostają modele w granicach jednego odchylenia od
najlepszego). Ostateczna lista zależy też od tego, kto uzyska dostęp do jakich modeli.
**Kto:** użytkownik. **Domyślnie:** C na liście z `encoders.yaml` zawężonej do modeli, które się pobiorą.

## D7. Publikacja wag i licencje

**Kontekst.** Licencja CC-BY-NC-ND (`uni2-h`, `virchow2`, `h-optimus-1`, `h0-mini`, `uni`) zabrania udostępniania wag
pochodnych (po dostrojeniu, LoRA, destylacji). Zamrożony enkoder i własna głowica są dozwolone, bo głowica nie
zawiera wag modelu. Permisywne: `h-optimus-0`, `virchow`, `hibou-l` (Apache-2.0), `midnight-12k` (MIT). Karta
Prov-GigaPath ogranicza użycie do badań mimo tagu Apache.

| Opcja | Konsekwencja |
|---|---|
| A. Publikujemy tylko głowice i wyniki, bez wag enkodera | każdy model dozwolony; czytelnik pobiera enkoder sam |
| B. Publikujemy też dostrojone enkodery | wolno tylko modele permisywne |
| C. Nic nie publikujemy (praca wewnętrzna) | dowolne modele, najmniej ograniczeń |
| D. Wymagamy zgody właściciela (kontakt dla zastosowań komercyjnych) | przy planach komercyjnych lub klinicznych |

Dodatkowo: zakaz użycia klinicznego w licencjach Virchow2 (diagnoza, IUO, RUO); zespół pracuje nad systemem wspierającym
diagnostykę, więc to trzeba przeczytać przed jakimkolwiek zastosowaniem poza badaniami.

**Rekomendacja:** A, a enkoder produkcyjny wybrać z permisywnych, jeśli różnica jakości jest mała.
**Kto:** użytkownik, promotor. **Domyślnie:** A.

## D8. Dostrajanie enkodera

| Opcja | Za | Przeciw |
|---|---|---|
| A. Tylko zamrożony enkoder | tanio, licencyjnie bezpiecznie, szybko powtarzalne po rundach | może brakować dopasowania do szpiku |
| B. LoRA na modelach permisywnych | zwykle zauważalny zysk przy małej liczbie wag | koszt, kod niezaimplementowany, wagi do publikacji tylko permisywne |
| C. Pełne dostrajanie | największa elastyczność | duży koszt, ryzyko przeuczenia przy małej liczbie danych |
| D. Dalszy pretrening SSL na nieoznaczonych kaflach szpiku (LeJEPA/DINO) | adaptacja bez etykiet, wątek badawczy | najdroższe, osobny projekt |

**Rekomendacja:** A w eksperymencie głównym, B jako ablacja dla `h-optimus-0` po wyniku siatki, D jako odrębne zadanie.
**Kto:** użytkownik. **Domyślnie:** A.

## D9. Komórki częściowo widoczne i ramka kontekstu

**Kontekst.** Pytanie 4 ze spotkania 27.10. Według notatek lekarz zaznacza komórkę od ok. 80% widoczności; przy 60%
w kaflu bez ramki nie zaznaczy, a z ramką tak, co w etykietach wyglądałoby jak błąd. Model dostaje „okno”, lekarz „kafel”.

| Opcja | Opis | Za | Przeciw |
|---|---|---|---|
| A. Zawsze pełna komórka | etykieta jak narysował lekarz | prosto | komórki ucięte brzegiem dają niejednoznaczne etykiety |
| B. Próg widoczności (np. 80%) | poniżej progu obszar komórki to `ignore` | spójne z praktyką lekarza | trzeba policzyć widoczność z obrysu |
| C. Pas `ignore` przy brzegu | w pasie szerokości w komórki nie liczą się do straty ani metryk | prosto i odporne | tracimy dane przy brzegu |
| D. Ramka kontekstu przy wysyłaniu do lekarza i przy predykcji | model i lekarz widzą szerszy wycinek, liczymy komórki ze środkiem w kaflu | najbliżej rzeczywistości | zmiana protokołu QuPath i kodu predykcji |

**Rekomendacja:** D do predykcji i wysyłki, w metrykach B (próg 80%). Komórki na brzegu są oznaczane flagą
`touches_border` już dziś.
**Kto:** dr Zduniak, zespół. **Domyślnie:** C (pas 16 px) w metrykach, D po implementacji.

## D10. Główna miara jakości

| Opcja | Za | Przeciw |
|---|---|---|
| A. PQ/mPQ (per klasa) | standard dla instancji, rozdziela detekcję (DQ) i dokładność (SQ) | nie mówi o pracy lekarza |
| B. Czas lekarza | bezpośredni cel AL | niewiarygodny (lekarz pamięta kafle), uwaga Igora |
| C. Nakład korekty: piksele i obiekty do poprawy | mierzalny, powtarzalny | wymaga ustalenia wag dodania, usunięcia, poprawy |
| D. Zestaw A+C, czas tylko pomocniczo | wiarygodne i bliskie celowi | więcej liczb do czytania |

**Rekomendacja:** D, z mPQ jako miarą rozstrzygającą wybór modelu i nakładem korekty do porównania strategii AL.
Zaimplementowane: PQ/DQ/SQ, mPQ, nakład korekty. **Kto:** zespół. **Domyślnie:** D.

## D11. Baseline'y zewnętrzne

| Opcja | Za | Przeciw |
|---|---|---|
| A. Tylko baseline barwnikowy | bez kosztów | słaby punkt odniesienia |
| B. CellViT++ (SAM-H, HIPT) | popularny, eksport GeoJSON | min. 24 GB VRAM, licencja z Commons Clause, jądra |
| C. HistoPLUS (H0-mini) | nowy, 13 typów komórek | CC-BY-NC-ND, bramkowany, tylko dla instytucji akademickich |
| D. B + C (+ CellViT-Hibou-L, PathoSAM) | najpełniej | czas wdrożenia, wiele uprawnień |

Uwaga: żaden z nich nie jest uczony na szpiku według opisów, więc porównanie pokazuje „ile daje własna głowica”, a nie
„który model segmentuje szpik lepiej”. **Rekomendacja:** B i C bez dostrajania, jako jedyny punkt odniesienia z
zewnątrz. **Kto:** użytkownik. **Domyślnie:** A, aż do wdrożenia B i C.

## D12. Miejsce danych i wyników na PLGrid

**Kontekst.** `$HOME` 10 GB, `$SCRATCH` czyszczony, storage grupy trwały (`/net/pr2/projects/plgrid/plggaktywneuczenie`).
Zasoby są do 07.2027.

| Opcja | Opis |
|---|---|
| A. Wszystko w `$SCRATCH` | szybko, ale ryzyko utraty |
| B. Dane i głowice na storage grupy, `$SCRATCH` jako cache i miejsce przebiegów | trwałe, zalecane |
| C. Osobny grant i katalog dla projektu MPN | porządek, wymaga wniosku |

Pytania do zespołu: czy projekt MPN używa grantu `plgaktywneuczenie-gpu-a100`, kto jest właścicielem katalogu grupy,
gdzie ma leżeć drzewo `data/`. **Rekomendacja:** B, `data/` dowiązane do storage grupy. **Kto:** zespół, Cyprian.
**Domyślnie:** B z katalogiem `mpn_al/` w storage grupy.

## D13. Drugi anotator i zgodność międzyobserwatorska

Wymóg z notatek: do publikacji dwóch niezależnych anotatorów (koleżanka dr. Zduniaka).

| Opcja | Za | Przeciw |
|---|---|---|
| A. Jeden anotator | tanio | publikacja wątpliwa |
| B. Dwóch na zbiorze testowym | rzetelny test i miara zgodności, tańsze niż całość | część kafli podwójnie |
| C. Dwóch na wszystkim | najlepsze | najdroższe |
| D. Dwóch na podzbiorze losowym | oszacowanie zgodności | test nadal jednego anotatora |

**Rekomendacja:** B. Daje też „sufit” dla PQ (zgodność lekarzy), do którego porównujemy model. **Kto:** promotor,
dr Zduniak. **Domyślnie:** A do czasu ustaleń.

## D14. Kryterium wyboru enkodera

Ustalić **przed** wynikami, żeby nie dobierać reguły do rezultatu.

| Opcja | Reguła |
|---|---|
| A. Najwyższe mPQ na walidacji | proste, ignoruje koszt |
| B. Najwyższe mPQ, remisy (różnica poniżej progu, np. 0,01 lub CI zachodzą na siebie) rozstrzyga koszt i licencja | praktyczne |
| C. Fronta Pareto: jakość kontra czas na kafel i pamięć | pokazuje kompromis |

**Rekomendacja:** B z progiem remisu zapisanym z góry i C jako wykres. Test dopiero po wyborze. **Kto:** użytkownik i
promotor. **Domyślnie:** B, próg 0,01 mPQ.

## D15. Parametry głowicy i post-processingu (techniczne, zaproponuję sam)

Szerokość dekodera (32 lub 64), wagi strat, wagi klas przy niezrównoważeniu, progi dekodowania instancji (prawdopodobieństwo,
próg markerów, minimalna powierzchnia w µm²). **Reguła:** strojone wyłącznie na walidacji w pilotażu, potem zamrożone
dla wszystkich modeli (te same ustawienia, żeby porównanie było uczciwe). Wpływ: małe, ale wyniki zależą od progów
dekodowania, więc muszą być wspólne. **Domyślnie:** wartości z pilotażu na jednym enkoderze, bez strojenia per model.

## D16. Wariancja barwienia i skanera

**Kontekst.** Embeddingi komórek grupują się głównie według obrazu, a skala i barwienie różnią się między obrazami.

| Opcja | Za | Przeciw |
|---|---|---|
| A. Nic (tylko augmentacja koloru w treningu) | proste | embeddingi zależne od preparatu |
| B. Normalizacja barwienia (Macenko/Reinhard) przed enkoderem | typowe w histologii | może zmienić to, co model widział w pretreningu |
| C. Centrowanie embeddingów w obrębie preparatu | tanie, tylko dla klasyfikatora | wymaga wielu komórek na preparat |
| D. Silniejsza augmentacja HED | bez zmiany wejścia | potrzebuje implementacji |

**Rekomendacja:** A w siatce głównej, B i D jako ablacje, C dla klasyfikatora komórek. **Kto:** użytkownik.
**Domyślnie:** A.

## D17. Źródło danych na początek

**Kontekst.** BaMBo ma maski klas (0–3), nie obrysy komórek, więc nie nadaje się do uczenia segmentacji komórek.
Dostępne: 10 obrazów od dr. Zduniaka (Google Drive), jego publiczny zbiór z anotacjami, inny publiczny zbiór
(Mendeley), BoMBR, a docelowo skany z DCO/Łukasiewicz.

| Opcja | Za | Przeciw |
|---|---|---|
| A. Publiczny zbiór z anotacjami od dr. Zduniaka | gotowe anotacje (klasy `meg_*`) | rozmiar i zgodność protokołu nieznane |
| B. 10 obrazów dr. Zduniaka + własne anotacje | kontrola nad protokołem | mało danych |
| C. Czekać na skany DCO/Łukasiewicz | docelowe dane | opóźnienie |
| D. A dla pilotażu, C dla wyniku końcowego | równoległość | dwie fazy |

**Rekomendacja:** D. Pilotaż mierzy czas i dobiera ustawienia, wynik końcowy dopiero na docelowych danych.
**Kto:** zespół. **Domyślnie:** D.

---

## Propozycja porządku obrad (jedno spotkanie, ok. 45 minut)

1. D17: skąd bierzemy anotacje do pilotażu (5 min).
2. D2 i D1 z lekarzem: co jest obrysem i jak wygląda kompletny kafel (15 min).
3. D3 i D9: klasy, `uncertain`, komórki na brzegu (10 min).
4. D5: podział po pacjentach i właściciel pliku (10 min).
5. D12 i D13 pro forma: gdzie dane, ilu anotatorów (5 min).

Decyzje D4, D6, D8, D14, D15, D16 zapadają po pilotażu na podstawie liczb; D7, D10, D11 wystarczy potwierdzić.

## Co się zmienia w kodzie po decyzjach

| Decyzja | Zmiana |
|---|---|
| D1 (D), D2 (A/C), D9 | obsługa nakładających się wielokątów, znacznik „kafel kompletny”, próg widoczności w rasteryzacji i metrykach |
| D3 (B) | klasyfikator komórek na embeddingach |
| D5 (C) | pętla walidacji krzyżowej po pacjentach w runnerze |
| D8 (B) | LoRA dla `timm` i `transformers` |
| D11 | skrypty eksportu CellViT++ i HistoPLUS do wspólnego formatu GeoJSON i metryk |
| D14, D6 (C) | skrypt wyboru według zapisanej reguły (odcinanie) |
