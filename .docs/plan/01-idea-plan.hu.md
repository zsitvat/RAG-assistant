# Ötletterv – Agentic RAG asszisztens (high level)

Angol változat: [01-idea-plan.en.md](01-idea-plan.en.md).

## 1. A választott téma

**Céges költségtérítési és munkavállalói támogatási asszisztens.**

Egy agentic RAG chatbot, amely egy fiktív vállalat belső szabályzatai alapján megválaszolja
a munkavállalók költségtérítéssel és juttatásokkal kapcsolatos kérdéseit: elszámolható-e egy
kiadás, mekkora összeg téríthető, milyen bizonylat kell hozzá, szükséges-e vezetői jóváhagyás,
és mi a következő lépés.

A prototípus nem kapcsolódik valódi pénzügyi rendszerhez, és nem ad adózási vagy jogi
tanácsot – a szabályzatok egy fiktív vállalatot írnak le, és kizárólag demonstrációs célt
szolgálnak.

## 2. Miért ezt a témát választottam

A témaválasztás szándékosan közelít a feladatot kiadó cég tevékenységéhez: a pénzügyi
folyamatok, belső kontrollok, szabályozottság és megfelelőség (compliance) vizsgálata egy
tanácsadó / könyvvizsgáló környezetben központi téma. Egy olyan use case-t akartam, amely
ehhez tartalmilag is kapcsolódik, nem csak technikai demó.

Emellett a probléma:

- valós és általános – szinte minden nagyobb vállalatnál létező adminisztratív folyamat;
- dokumentum-alapú – a válaszok belső szabályzatokból származnak, ezért természetesen RAG-ot igényel;
- döntési logikát tartalmaz – limitek, jogosultság, határidők, jóváhagyási szintek;
- számítást igényel – így determinisztikus toolokra van szükség, nem elég az LLM szöveggenerálása.

## 3. Milyen igényt elégít ki

A munkavállaló természetes nyelven kérdez, és egyetlen válaszban megkapja:

1. elszámolható-e / jogosult-e;
2. mekkora összeg téríthető;
3. milyen dokumentum szükséges;
4. kell-e jóváhagyás;
5. mi a következő lépés és melyik szabálypont alapján.

Ezzel kiváltja a szabályzatok manuális átolvasását, és csökkenti a hibás igénylések számát.

## 4. Miért agentic RAG, és nem sima RAG

Egy sima RAG megkeresi a releváns szövegrészt és válaszol. A tipikus kérdések viszont több
lépésesek, és a válasz nem áll a dokumentumban készen. Példa:

> „Saját autóval járok dolgozni, 32 km-re lakom az irodától, hetente háromszor megyek be.
> Mennyi támogatást kaphatok ebben a hónapban?”

A rendszernek fel kell ismernie a kategóriát, ki kell nyernie az adatokat, vissza kell kérdeznie
(egyirányú vagy teljes távolság?), meg kell keresnie a szabályt, ellenőriznie a jogosultságot,
majd kiszámolnia az összeget. Ehhez kell: szándékfelismerés, állapotkezelés, feltételes
útválasztás, tool-használat és visszakérdezés – vagyis agentic működés.

Kulcsviselkedés: hiányos kérdés esetén a chatbot **nem tippel**, hanem rákérdez a hiányzó adatra.

## 5. Lefedett kategóriák (szűkítve, kezelhető prototípusra)

1. **Üzleti költségek** – étkezés, szállás, taxi, parkolás.
2. **Munkába járás** – bérlet, saját autó, kilométer-térítés.
3. **Munkaeszközök** – monitor, fejhallgató, irodai eszközök.
4. **Munkavállalói támogatások** – nyaralási / rekreációs / képzési támogatás, home office.

## 6. Megoldás nagy vonalakban

- **Fő agentic workflow (LangGraph):** intent felismerés → adatkinyerés → hiányzó adat ellenőrzés
  (visszakérdezés) → autonóm tool-loop → forrásokra épülő válaszgenerálás.
- **Dedikált RAG algráf (subgraph):** szándékosan egyszerű – kérdés embeddelése → similarity search
  költségkategória-szűrővel → kontextusépítés idézetekkel. Önálló,
  újrahasznosítható modul; a projekt komplexitása az agentic workflow-ban és a toolokban van, nem a
  keresőben.
- **Toolok (determinisztikus Python, nem LLM):** költség- és térítéskalkulátor, szabályellenőrző
  és határidő-számító.
- **Adatforrás:** fiktív vállalati szabályzatok `.docx` formátumban a
  `.docs/sources/<lang>/` alatt, ingest során Markdownra konvertálva, hogy a címsor-struktúra megmaradjon;
  kézzel átnézve a belső konzisztencia miatt, kis mennyiségű – a hangsúly a minőségi feldolgozáson van. A
  strukturált szabálykatalógus (`rules.yaml`) a prototípusban kézzel írott; éles megoldásban magukból a
  dokumentumokból lenne kinyerve.
- **Nyelv:** egy angol és egy magyar tudásanyag, mindkettőhöz külön Redis index, és egy config beállítás
  (`ACTIVE_LANG`) dönti el, melyiket használja az asszisztens. Mindkét index ugyanazzal a multilingual
  embedding modellel készül: magyarra nincs erős nyelvspecifikus retrieval embedder, és a közös modell
  miatt a két nyelv eval-számai összehasonlíthatók maradnak.
- **Tároló:** Redis (Redis Stack) mint egyetlen adatbázis – vektorindexek a szabályzat-chunkoknak
  és LangGraph beszélgetés-checkpointok.
- **Modell:** helyben futtatható, nyílt forráskódú LLM (fizetős API nélkül); szükség esetén dummy LLM
  fallback.
- **Szolgáltatás-szétválasztás:** az agent FastAPI service-ként fut, a Streamlit egy vékony HTTP-kliens –
  így az agentet az értékelő szkript vagy curl is közvetlenül hívhatja.
- **UI:** fókuszált Streamlit chat, amely a választ, a dátumot és a válaszidőt, valamint egy
  streamelt, lenyitható forrás- és lépésösszefoglalót mutat. A részletes tool argumentumok,
  időzítések és retrieval diagnosztika Langfuse-ban láthatók.
- **Futtatás:** konténerizált – Dockerfile és docker-compose (`api`, `ui`, `redis`, `ollama`) a repo
  gyökerében. Az alkalmazás stdout-ra és hét napig megőrzött, naponta rotált fájlokba logol.

## 7. Értékelés nagy vonalakban

- **Funkcionális:** 10–20 kérdésből álló tesztkészlet a fenti kategóriákra (benne hiányos kérdés,
  kerettúllépés, tiltott költségtípus, összetett eset). Mért dimenziók: intent-pontosság,
  adatkinyerés, dokumentumtalálat, tool-választás, kalkuláció, végső döntés, forrásmegjelölés.
  A verziózott JSON dataset a repóban marad, a futtatás és score-ok Langfuse-ba kerülnek.
- **Teljesítmény:** 50–200 lekérdezés, latency metrikák (átlag, medián, p95), a bottleneck
  azonosítása (várhatóan az LLM-generálás), és 1–2 konkrét optimalizálási javaslat.

## 8. Várható eredmény

Működő prototípus, amely egy életszerű vállalati adminisztrációs problémán demonstrálja a
LangGraph-alapú agentic működést, a moduláris RAG-ot, a determinisztikus tool-használatot, az
állapotkezelést és az autonóm döntéshozatalt – forrásokkal alátámasztott, ellenőrizhető válaszokkal.

Megjegyzés a terjedelemről: a graph szándékosan kicsi marad, miközben megtartja a külön intent
classification node-ot, az autonóm tool-választást, a moduláris RAG-ot, a determinisztikus
kalkulációt és a checkpointelt többkörös állapotot.
