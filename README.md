# Darukaa.Earth — AI Biodiversity Intelligence Chatbot

An AI-powered environmental scientist assistant that answers biodiversity and environment questions using a grounded knowledge base, multi-metric reasoning, and the Google Gemini API.

[![Live App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit)](https://darukaa-biodiversity-ai-7.streamlit.app/)
[![Backend API](https://img.shields.io/badge/Render-FastAPI%20Backend-46E3B7?logo=render)](https://darukaa-biodiversity-ai-atjv.onrender.com)
[![API Docs](https://img.shields.io/badge/Swagger-API%20Docs-85EA2D?logo=swagger)](https://darukaa-biodiversity-ai-atjv.onrender.com/docs)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/Shivaram7-29/darukaa-biodiversity-ai)

- **Live Frontend**: [https://darukaa-biodiversity-ai-7.streamlit.app/](https://darukaa-biodiversity-ai-7.streamlit.app/)
- **Backend API**: [https://darukaa-biodiversity-ai-atjv.onrender.com](https://darukaa-biodiversity-ai-atjv.onrender.com)
- **API Health Check**: [https://darukaa-biodiversity-ai-atjv.onrender.com/health](https://darukaa-biodiversity-ai-atjv.onrender.com/health)
- **GitHub Repository**: [https://github.com/Shivaram7-29/darukaa-biodiversity-ai](https://github.com/Shivaram7-29/darukaa-biodiversity-ai)
- **Final Deployed Commit**: `678ea98`

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
  ── Embed query with remote Gemini API (gemini-embedding-001)
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
    "What is your annual rainfall (mm) or current soil moisture level (%)? (Determines whether biological amendments can establish without moisture stress).",
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
      "time_horizon": "1-3 years for gradual pH buffering stabilization; ongoing annual maintenance required",
      "measurable_improvement": "Gradual SOC increase under continuous annual application in degraded soils, with improved cation exchange capacity and microbial biomass (Lal 2004; FAO GSOCmap)",
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

## CI/CD & Deployment

### Live Production Deployment

The platform is deployed and fully operational across two decoupled cloud hosting platforms:

```
┌───────────────────────────┐         HTTP / Streaming         ┌───────────────────────────┐
│ Streamlit Community Cloud │ ───────────────────────────────► │   Render Web Service      │
│  (Interactive Web UI)     │   (JSON / NDJSON Stream)         │  (FastAPI + ChromaDB)     │
│  streamlit_app.py         │ ◄─────────────────────────────── │  app/api.py               │
└───────────────────────────┘                                  └─────────────┬─────────────┘
                                                                             │
                                                                             ▼
                                                               ┌───────────────────────────┐
                                                               │ Google Gemini Cloud APIs  │
                                                               │ - gemini-3.5-flash-lite   │
                                                               │ - gemini-embedding-001    │
                                                               └───────────────────────────┘
```

1. **Frontend**: [https://darukaa-biodiversity-ai-7.streamlit.app/](https://darukaa-biodiversity-ai-7.streamlit.app/)
   - Hosted on Streamlit Community Cloud.
   - Connected directly to GitHub repository (`origin/main`).
   - Auto-deploys instantaneously upon commit merges.
   - Communicates with the Render backend via secure HTTPS (`BACKEND_URL`).

2. **Backend API**: [https://darukaa-biodiversity-ai-atjv.onrender.com](https://darukaa-biodiversity-ai-atjv.onrender.com)
   - Hosted on Render Web Services as a managed Python 3.12 service.
   - Connected directly to GitHub repository (`origin/main`) with automated deployment on push.
   - Memory-optimized: Uses Google's remote `gemini-embedding-001` API and precomputed embeddings (`precomputed_embeddings.json`), requiring **under 140 MB of RAM** and operating reliably within Render's 512 MB free tier.
   - Exposes interactive Swagger documentation at `/docs` and health monitoring at `/health`.

3. **Final Deployed Commit**: `678ea98` ("Refine soil moisture and SOC interpretations for scientific grounding").

### Automated Verification Pipeline
Before deployment, the system is validated locally via two comprehensive test suites:
```bash
# 1. Comprehensive Hackathon Audit Suite (6/6 sections pass)
python scratch/test_final_audit.py

# 2. End-to-End Scientific Compliance Suite (8/8 compliance suites pass)
python scratch/test_compliance_suite.py
```
