# Darukaa.Earth — AI Biodiversity Intelligence Chatbot

An AI-powered environmental scientist assistant that answers biodiversity and environment questions using a grounded knowledge base, multi-metric reasoning, and the Gemini API.

## Architecture

```
┌─────────────────┐     HTTP     ┌──────────────────────────────────────────────────┐
│  Streamlit UI   │◄────────────►│              FastAPI Backend                     │
│  (ui/)          │              │  (app/api.py)                                    │
└─────────────────┘              │                                                  │
                                 │  ┌──────────────────────────────────────────┐    │
                                 │  │         ConversationManager              │    │
                                 │  │  - Chat history management               │    │
                                 │  │  - Pipeline orchestration                │    │
                                 │  │                                          │    │
                                 │  │  ┌──────────────┐  ┌─────────────────┐  │    │
                                 │  │  │  Reasoning    │  │  RAG Retriever  │  │    │
                                 │  │  │  Engine       │  │  (ChromaDB)     │  │    │
                                 │  │  │  - Input      │  │  - Embeddings   │  │    │
                                 │  │  │    parsing    │  │  - Similarity   │  │    │
                                 │  │  │  - Threshold  │  │    search       │  │    │
                                 │  │  │    checks     │  │                 │  │    │
                                 │  │  │  - Cross-var  │  │                 │  │    │
                                 │  │  │    analysis   │  │                 │  │    │
                                 │  │  └──────────────┘  └─────────────────┘  │    │
                                 │  │                                          │    │
                                 │  │  ┌──────────────────────────────────┐    │    │
                                 │  │  │         Gemini API               │    │    │
                                 │  │  │  - System prompt with rules      │    │    │
                                 │  │  │  - Context-enriched generation   │    │    │
                                 │  │  └──────────────────────────────────┘    │    │
                                 │  └──────────────────────────────────────────┘    │
                                 └──────────────────────────────────────────────────┘
```

### Data Flow

```
User Question + (optional) Structured Env Data
        │
        ▼
  Structured Input Parsing (reasoning.py)
  ── Parse environmental variables
  ── Flag threshold violations
  ── Detect cross-variable interactions
  ── Generate clarifying questions
        │
        ▼
  RAG Retrieval (retriever.py + ChromaDB)
  ── Embed query with sentence-transformers
  ── Cosine similarity search in ChromaDB
  ── Return top-k relevant knowledge entries
        │
        ▼
  Prompt Construction (conversation.py)
  ── System prompt with scientific rules
  ── Chat history for context
  ── Environmental data assessment
  ── Retrieved knowledge context
        │
        ▼
  Gemini API Generation
  ── Grounded response with recommendations
  ── Structured format: action, basis, metrics, timeline, source
        │
        ▼
  Response with metadata
  ── AI response text
  ── Environmental concerns
  ── Source references
  ── Cross-variable interactions
  ── Clarifying questions
```

## Project Structure

```
darukaa-biodiversity-ai/
├── .env                              # GOOGLE_API_KEY (not committed)
├── .gitignore
├── README.md
├── app/
│   └── api.py                        # FastAPI backend (endpoints)
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration & env variables
│   ├── embeddings.py                 # Remote Gemini embedding API (zero local model in RAM)
│   ├── knowledge_base.py             # JSON knowledge loader & document prep
│   ├── retriever.py                  # ChromaDB vector store & RAG retrieval
│   ├── reasoning.py                  # Multi-metric reasoning engine
│   └── conversation.py               # Conversation manager & Gemini integration
├── ui/
│   └── streamlit_app.py              # Streamlit chat frontend
├── data/
│   ├── documents/                    # (extensible) additional documents
│   └── environmental_data/
│       └── knowledge.json            # Structured environmental knowledge base
├── chroma_db/                        # ChromaDB persistent storage (auto-created)
└── .venv/                            # Python virtual environment
```

## Database & Schema

The platform employs a hybrid data architecture consisting of a persistent vector database (ChromaDB), structured JSON seed knowledge, and in-memory conversation state.

### 1. Vector Database (ChromaDB)
- **Engine**: ChromaDB (`chromadb.PersistentClient`)
- **Collection Name**: `environmental_knowledge`
- **Persistence Directory**: `chroma_db/` (persisted locally on disk)
- **Distance Metric**: Cosine distance (`hnsw:space: "cosine"`)
- **Embedding Model**: `gemini-embedding-001` via Google GenAI remote API (precomputed offline into `data/environmental_data/precomputed_embeddings.json`, zero local ML model in RAM)
- **Vector Dimension**: 768 dimensions (dense float32 vectors)

#### Vector Document Types
During document ingestion (`src/knowledge_base.py`), each knowledge topic is partitioned into three specialized document chunk types:

| Document Type (`doc_type`) | ID Format | Content Embedded |
|---|---|---|
| `knowledge` | `{id}_knowledge` | Core scientific overview, threshold definitions, and general principles |
| `recommendation` | `{id}_rec_{index}` | Specific actionable interventions, scientific basis, metrics affected, timeline, and confidence |
| `interaction` | `{id}_interactions` | Cross-variable relationships and compound environmental effects |

#### Vector Metadata Schema
Every chunk stored in ChromaDB contains the following metadata attributes for filtering and attribution:
```python
{
    "id": str,          # Base topic identifier (e.g., "soc_001", "moist_001")
    "topic": str,       # Environmental topic name (e.g., "soil organic carbon")
    "category": str,    # Domain category (e.g., "soil_health", "water", "biodiversity")
    "doc_type": str,    # "knowledge" | "recommendation" | "interaction"
    "variables": str,   # Comma-separated variables (e.g., "soil_organic_carbon, water_retention")
    "source": str       # Authoritative source/literature citation
}
```

### 2. Knowledge Base Seed Schema (`knowledge.json`)
The primary scientific knowledge resides in `data/environmental_data/knowledge.json`. Each entry adheres to the following JSON schema:

```json
{
  "id": "soc_001",
  "topic": "soil organic carbon",
  "category": "soil_health",
  "environmental_variables": ["soil_organic_carbon", "soil_health", "biodiversity", "water_retention"],
  "knowledge": "Scientific description of the metric, health baselines, and environmental role...",
  "thresholds": {
    "critical_low": 0.5,
    "low": 1.0,
    "moderate": 2.0,
    "good": 3.0,
    "excellent": 5.0,
    "unit": "percent"
  },
  "interactions": {
    "soil_ph": "Interaction description with pH...",
    "soil_moisture": "Interaction description with moisture...",
    "temperature": "Interaction description with temperature...",
    "rainfall": "Interaction description with rainfall..."
  },
  "recommended_actions": [
    {
      "action": "Clear, actionable intervention",
      "scientific_basis": "Biogeochemical or ecological mechanism",
      "metrics_affected": ["metric_1", "metric_2"],
      "time_horizon": "Expected turnaround for measurable results",
      "confidence": "high | moderate | low"
    }
  ],
  "source": "Primary literature citation or institutional source (e.g., FAO, IPCC, USDA-NRCS)"
}
```

**Topics Covered**: Soil organic carbon, soil pH & nutrient availability, soil moisture & water management, rainfall patterns & water resources, temperature effects, biodiversity indicators, pollution & contamination, deforestation & land degradation, crop monoculture, agroforestry systems, wetland conservation, and soil erosion.

### 3. Conversation & Session State Schema
- **Storage**: In-memory dictionary managed by `ConversationManager` (`src/conversation.py`).
- **Key**: `conversation_id` (UUID4 string).
- **Value**: Chronological list of turns, each structured as:
  ```json
  [
    {"role": "user", "content": "User input text"},
    {"role": "assistant", "content": "Scientifically grounded assistant response"}
  ]
  ```
- **Context Window**: Rolling buffer capped at `MAX_CONVERSATION_HISTORY = 20` messages to optimize prompt context without exceeding token limits.
- **Lifecycle**: Maintained across multi-turn interactions via API/UI sessions; reset via `/chat/{id}/clear` or server restart.

---

## Local Setup

### Prerequisites

- Python 3.12+
- A Google Gemini API key

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd darukaa-biodiversity-ai
   ```

2. **Create/activate virtual environment** (if not already):
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your API key** in `.env`:
   ```
   GOOGLE_API_KEY=your-api-key-here
   ```

## Running the Application

### 1. Start the FastAPI Backend

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

On first startup, the knowledge base will be automatically embedded and indexed in ChromaDB.

### 2. Start the Streamlit Frontend

In a **separate terminal**:

```bash
streamlit run ui/streamlit_app.py
```

The frontend will open at `http://localhost:8501`.

## Usage

### Natural Language Queries

Simply type questions like:
- "How does deforestation affect local rainfall patterns?"
- "What are the best practices for improving biodiversity in a rice monoculture?"
- "Explain the relationship between soil organic carbon and water retention."

### Structured Environmental Data

Toggle the "Provide structured environmental data" panel in the sidebar to enter specific measurements:
- Soil organic carbon (%)
- Soil pH
- Soil moisture (% field capacity)
- Rainfall (mm/year)
- Temperature (°C)
- Land use type
- Biodiversity index
- Pollution level
- Deforestation rate

The system will:
1. Assess each variable against scientific thresholds
2. Detect cross-variable interactions
3. Retrieve relevant knowledge from the vector database
4. Generate specific, grounded recommendations

### Sample Test Scenarios

**Scenario 1: Degraded Farmland**
- Toggle environmental data panel
- Soil Organic Carbon: 0.8%
- Soil pH: 4.5
- Land Use: "rice monoculture"
- Question: "What should I do to restore this farmland?"

**Scenario 2: Deforestation Impact**
- Deforestation Rate: 3.0%/year
- Biodiversity Index: 0.8
- Rainfall: 600 mm/year
- Question: "How is deforestation affecting this ecosystem and what can be done?"

**Scenario 3: Simple Query (no data)**
- Question: "What is the relationship between soil organic carbon and biodiversity?"

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + knowledge base info |
| `POST` | `/chat` | Main chat endpoint |
| `POST` | `/chat/stream` | Streaming chat endpoint (SSE stream with progressive tokens + final structured payload) |
| `POST` | `/chat/{id}/clear` | Clear conversation history |
| `POST` | `/admin/rebuild-index` | Rebuild ChromaDB index |

### Example API Call

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "My soil has 0.8% organic carbon and pH 4.5. What should I do?",
    "environmental_data": {
      "soil_organic_carbon": 0.8,
      "soil_ph": 4.5,
      "land_use": "rice monoculture"
    }
  }'
```

### Example Structured Response

```json
{
  "conversation_id": "8f3b2c1a-5b21-4a16-92f7-7b64a13d9c82",
  "response": "### Assessment\nYour soil exhibits severe compound degradation: low soil organic carbon (0.8%) combined with strong acidity (pH 4.5) in a rice monoculture...",
  "concerns": [
    "CRITICAL: Soil organic carbon (0.8%) is critically low (<1.0%). Soil is degraded.",
    "WARNING: Soil pH (4.5) is strongly acidic. Nutrient availability is compromised and aluminum toxicity risk is elevated.",
    "NOTE: Monoculture land use (rice monoculture) is associated with reduced biodiversity and increased pest pressure."
  ],
  "clarifying_questions": [
    "What is your annual rainfall (mm) or current soil moisture level (% FC)? (Determines whether biological amendments can establish without moisture stress).",
    "What is your soil texture (e.g., clay, loam, sandy loam)? (Assists in calculating precise organic matter retention and buffer capacity)."
  ],
  "sources": [
    "FAO Global Soil Organic Carbon Map (GSOCmap) — fao.org/soils-portal",
    "Brady, N.C. & Weil, R.R. (2016) The Nature and Properties of Soils, 15th Edition, Pearson"
  ],
  "assessment": {
    "provided_variables": [
      "soil_organic_carbon",
      "soil_ph",
      "land_use"
    ],
    "missing_variables": [
      "soil_moisture",
      "rainfall",
      "temperature",
      "biodiversity_index",
      "pollution_level",
      "deforestation_rate"
    ],
    "interactions": [
      "Low SOC combined with extreme pH suggests severely compromised soil biology. Address pH first, as microbial activity needed for SOC buildup is pH-dependent."
    ]
  },
  "recommendations": [
    {
      "action": "Apply compost or well-decomposed organic manure annually at 5-10 tonnes/ha alongside agricultural lime.",
      "scientific_basis": "External organic matter additions supply stable carbon pools, enhance cation exchange capacity to buffer acidic pH, and stimulate microbial biomass to kickstart biological nutrient cycling.",
      "metrics_affected": [
        "Soil organic carbon",
        "soil pH stability",
        "nutrient availability",
        "soil biological activity"
      ],
      "time_horizon": "1-3 years for measurable SOC elevation; ongoing annual maintenance required",
      "measurable_improvement": "Can raise SOC by 0.1-0.2% per year under sustained annual application",
      "evidence": "FAO Global Soil Organic Carbon Map (GSOCmap); Brady, N.C. & Weil, R.R. (2016) The Nature and Properties of Soils"
    }
  ],
  "confidence": "high"
}
```

## Limitations

- **Knowledge base scope**: Currently covers 12 environmental topics. Additional topics can be added to `knowledge.json`.
- **No real-time data**: Does not connect to live environmental monitoring systems.
- **Single-user sessions**: Conversation state is stored in memory; restarting the backend clears all conversations.
- **No authentication**: The API is open. Add authentication for production use.
- **Embedding model**: Uses Google Gemini's `gemini-embedding-001` (768-dim) remote API, with precomputed offline embeddings ensuring zero local model memory overhead.
- **Gemini rate limits**: Subject to Google Gemini API rate limits and quotas.

## Extending the Knowledge Base

To add new topics, add entries to `data/environmental_data/knowledge.json` following the existing structure, then rebuild the index:

```bash
curl -X POST http://localhost:8000/admin/rebuild-index
```

Or restart the backend after deleting the `chroma_db/` directory.

---

## CI/CD

### Current Status
Automated CI/CD pipelines (e.g., GitHub Actions workflows) are **not currently configured or committed** in this repository. All builds, dependency installations, vector index generation, and end-to-end testing are currently performed and verified in local environments.

### Practical CI/CD & Deployment Architecture

For production readiness and submission deployment, the recommended CI/CD lifecycle is structured as follows:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ GitHub Actions  │──────►│ Container Build │──────►│ Cloud Hosting   │
│ (CI Tests/Lint) │       │ (Docker Images) │       │ (Backend & UI)  │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

#### 1. Continuous Integration (CI Pipeline)
Triggered on every `push` and `pull_request` to the `main` branch:
- **Linting & Code Quality**: Run `ruff check .` or `flake8` to enforce PEP 8 standards.
- **Type Checking**: Run `mypy src/ app/` for strict type adherence on reasoning schemas.
- **Unit & Property Tests**: Run `pytest tests/` covering:
  - Threshold evaluations and single-variable alert bounds in `src/reasoning.py`.
  - Multi-variable interaction triggers (e.g. high rainfall + low soil moisture).
  - Schema validation for `data/environmental_data/knowledge.json`.
  - ChromaDB ingestion and cosine similarity retrieval in `src/retriever.py`.
- **Integration Test**: Launch FastAPI in test mode and execute a mock `/chat` call verifying non-empty response and structured metadata.

#### 2. Continuous Deployment (CD Pipeline & Containerization)
- **Containerization via Docker**:
  - **Backend Container (`Dockerfile.api`)**: Packages Python 3.12, installs runtime dependencies, pre-downloads the embedding model `all-MiniLM-L6-v2` during image build, and exposes port 8000 running Uvicorn.
  - **Frontend Container (`Dockerfile.ui`)**: Packages Streamlit, targets port 8501, configured to route API traffic to the backend service.
  - **Multi-Container Orchestration (`docker-compose.yml`)**:
    ```yaml
    services:
      backend:
        build:
          context: .
          dockerfile: Dockerfile.api
        ports:
          - "8000:8000"
        volumes:
          - chroma_data:/app/chroma_db
        environment:
          - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      frontend:
        build:
          context: .
          dockerfile: Dockerfile.ui
        ports:
          - "8501:8501"
        environment:
          - BACKEND_URL=http://backend:8000
    volumes:
      chroma_data:
    ```
- **Deployment Targets**:
  - **FastAPI Backend**: Deploy to managed container platforms such as Google Cloud Run or AWS ECS with automated secret injection for `GOOGLE_API_KEY`.
  - **Streamlit Frontend**: Deploy via Streamlit Community Cloud or alongside the backend on container services.
  - **Vector Store Persistence**: Mount persistent volume storage to preserve `chroma_db/` across container restarts.
