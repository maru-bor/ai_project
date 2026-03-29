# ML/AI projekt - Predikce žánrů filmů
Tento projekt umožňuje predikovat žánry filmů na základě titulku a popisu. Používá strojové učení s **TF-IDF** vektorizací a **OneVsRestClassifier** s Logistic Regression, aby zvládla multi-label klasifikaci (tj. filmy mohou patřit do více žánrů).

---

## Jak model funguje
1. **TF-IDF vektorizace**
   - Text (titulek + popis filmu) se převede na číselný vektor
   - Každé slovo nebo kombinace slov (n-gram) má hodnotu podle frekvence ve filmu a vzácnosti v celém datasetu
2. **MultiLabelBinarizer**
   - Převede seznam žánrů na číselný vektor
3. **OneVsRest Classifier + Logistic Regression**
   - Pro každý žánr se trénuje samostatný binární klasifikátor
   - Model vrací pravděpodobnost pro každý žánr.
   - Používá se threshold 0.5, tj. žánr je predikován, pokud pravděpodobnost ≥ 0.5
  
---

## Distribuce žánrů v datasetu
| Žánr    | Počet filmů |
| -------- | ------- |
| Drama  | 6340    |
| Komedie | 4325    |
| Thriller    | 2210    |
| Akční  | 1931    |
| Krimi | 1846     |
| Dokumentární    | 1687    |
| Romatický    | 1416    |
| Dobrodružný  | 1354    |
| Sci-Fi | 895     |
| Životopisný    | 819    |
| Fantasy  | 773    |
| Animovaný| 709     |
| Horor    | 692    |
| Historický| 581    |
| Válečný    | 462    |


## Návod pro instalaci
1. Naklonujte repozitář:
```
git clone https://github.com/maru-bor/ai_project.git
cd ai_project
```

2. Vytvořte virtuální prostředí:
- Unix / macOS:
```
python3 -m venv .venv
source .venv/bin/activate
```
- Windows:
```
python -m venv .venv
.venv\Scripts\activate
```

3. Instalujte závislostí:
```
pip install -r requirements.txt
```
4. Spusťte aplikaci:
```
python app.py
```
5. Přejděte na adresu:
```
http://127.0.0.1:5000/
```
