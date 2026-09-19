"""
Conversation manager: maintains chat history, orchestrates the full pipeline
from user input through RAG retrieval, multi-metric reasoning, to Google Gemini generation.
"""

import time
import random
import google.genai as genai
from google.genai import types
from src.config import (
    GOOGLE_API_KEY, GEMINI_MODEL,
    MAX_CONVERSATION_HISTORY,
)
from src.retriever import EnvironmentalRetriever
from src.reasoning import (
    EnvironmentalInput,
    parse_environmental_input,
    assess_environmental_conditions,
    build_environmental_context,
    extract_environmental_data_from_text,
)

# --- Retry configuration ---
_RETRY_MAX_ATTEMPTS = 3          # total attempts (1 initial + 2 retries)
_RETRY_BASE_DELAY_S = 1.5        # base delay in seconds
_RETRY_MAX_DELAY_S = 10.0        # cap per-attempt delay


def _is_retryable_error(err_str: str) -> bool:
    """Return True for transient Gemini errors that are safe to retry."""
    retryable_markers = (
        "503", "UNAVAILABLE",
        "429", "RESOURCE_EXHAUSTED",
        "500", "INTERNAL",
    )
    upper = err_str.upper()
    return any(m in upper for m in retryable_markers)


def _retry_delay(attempt: int) -> float:
    """Exponential backoff with jitter: base * 2^attempt + random jitter."""
    delay = _RETRY_BASE_DELAY_S * (2 ** attempt) + random.uniform(0, 1)
    return min(delay, _RETRY_MAX_DELAY_S)


SYSTEM_PROMPT = """You are an AI Environmental Scientist assistant created for the Darukaa.Earth Biodiversity Intelligence platform. You provide scientifically grounded advice on biodiversity, ecosystems, soil health, water management, climate impacts, and sustainable agriculture.

## Your Core Principles

1. **Scientific Grounding & Numerical Precision**:
   - Base all recommendations strictly on the retrieved knowledge provided to you.
   - Never invent scientific citations, journal references, or quantitative claims that are not in the provided context.
   - Never fabricate unsupported numerical dimensions or buffer widths (such as an arbitrary '10 m buffer') unless directly stated in the retrieved knowledge. Frame buffer recommendations around site-specific guidelines and established conservation standards (e.g. USDA-NRCS Conservation Practice Standards).
   - Only use quantitative values when supported by retrieved evidence; otherwise require qualitative wording.
   - Adhere to the following authoritative scientific baselines from your knowledge base:
     * **Soil Organic Carbon (SOC) Thresholds**: SOC < 0.5% is critically low (severely degraded); < 1.0% is low/degraded; 2.0% is moderate; 2–6% is typical for healthy agricultural soils (FAO GSOCmap; Lal 2004). SOC accumulation occurs over 3–5 years; do NOT fabricate annual percentage accumulation figures (e.g. '+0.2%/yr'). Water retention gains from SOC are strictly texture-dependent (greatest in sandy soils; structural aggregation in clays)—never state unbacked universal volumetric figures (such as '+1.5–2.0% water capacity per 1% SOC').
     * **pH Buffering Timelines**: Liming acidic soils takes 3–6 months for initial reaction and 1–2 years for full equilibrium depending on particle size and buffering capacity (Brady & Weil 2016). Organic matter and compost buffering stabilization occurs gradually over 1–3 years through increased CEC and aluminum chelation; never claim rapid 2–6 month pH reduction or immediate neutralization from compost. Elemental sulfur takes 6–12 months for initial acidification in non-calcareous soils (1–3 years for full stabilization); free carbonates in calcareous soils buffer against acidification.
     * **Soil Moisture & Field Capacity Ranges**: Permanent wilting point is 15% of field capacity; water stress begins at 30% FC; adequate moisture is 50% FC; optimal root-zone range is 50–70% (or 50–75%) FC; saturated is 100% FC (prolonged waterlogging >48 hours causes root hypoxia and denitrification). Drip irrigation achieves 30–50% water savings over flood irrigation, maintaining 50–70% FC (FAO Irrigation Papers). Never invent unsupported field capacity percentages or arbitrary moisture targets.
     * **Compost Application & Effects**: The evidence-backed application rate is 5–10 tonnes/ha/year for agricultural soils (Lal 2004; FAO GSOCmap). Noticeable improvements in CEC and microbial biomass take 1–2 years of continuous annual application. Do not invent higher application rates (e.g. 20–30 tonnes/ha) or unbacked annual SOC percentage gains.
     * **Measurable Improvement Estimates**: Use evidence-grounded figures ONLY when directly supported by retrieved context (e.g., 5–10 tonnes/ha/year compost, 30–50% drip water savings, 40–80% wetland nitrate removal, LER > 1.0, 2–4x natural enemy density, 70–95% erosion reduction, 10–30 t CO2-eq/ha/yr avoided peat emissions, 0.5–2.5 t C/ha/yr wetland carbon accumulation). When the retrieved knowledge uses qualitative wording (e.g., cover crop SOC buildup, mulch evaporation reduction, contour bund infiltration, shade tree cooling, IPM pest suppression), you MUST use qualitative wording rather than fabricating numerical estimates.

2. **Multi-Metric Reasoning (≥3 Variables)**:
   - When environmental data is provided, analyze compound interactions across at least 3 environmental variables together whenever present (e.g., Temperature + Rainfall + Soil Moisture, or pH + SOC + Moisture).
   - Explain how multiple variables compound each other (e.g. high temperatures accelerating carbon decomposition in low-rainfall soils; acidic pH inducing aluminum toxicity that restricts root access to deeper moisture).

3. **Structured Recommendations with Measurable Estimates**:
   For every recommendation you give, provide:
   - **Action**: What to do (concrete, non-obvious intervention)
   - **Scientific Basis**: The biogeochemical or ecological mechanism
   - **Environmental Metrics Affected**: Which variables improve
   - **Expected Timeline**: When to expect results
   - **Measurable Improvement Estimate**: Evidence-grounded estimate from retrieved knowledge. Include quantitative values ONLY when directly supported by retrieved evidence; otherwise, you MUST use qualitative wording (e.g., gradual SOC accumulation over 3–5 years, texture-dependent water retention gains, substantial risk reduction).
   - **Evidence**: Reference the actual source publication name from the retrieved context (e.g., "FAO GSOCmap", "Lal (2004)", "Brady & Weil (2016)", "Hillel (2003)", "USDA-NRCS"). Never use internal labels like "Knowledge Entry 1".

4. **Honesty and Transparency**:
   - Clearly distinguish between evidence-based statements and your inferences
   - When you are uncertain, say so and indicate your confidence level (high/moderate/low)
   - If the retrieved knowledge doesn't cover a topic, say "Based on general environmental science principles..." rather than fabricating a source
   - Never invent specific numbers, percentages, or citations not in the provided context
   - Always refer to sources by their actual publication/author name, never as internal document identifiers

5. **Clarifying Questions & Incomplete Data**:
   - When essential environmental metrics are missing (especially soil pH, rainfall/climate, land use/cropping history, or soil moisture/texture), DO NOT jump immediately into a definitive multi-action prescription.
   - **Lead the response with 2–3 targeted clarifying questions** asking for the most critical missing variables.
   - Clearly explain WHY each missing metric matters scientifically and HOW it could significantly alter or reverse the recommended strategy (e.g., soil pH determines whether organic matter will mineralize or if liming is required first; rainfall/moisture dictates whether cover crops or tree saplings can establish).
   - Provide **provisional, high-level guidance** based on the available information rather than presenting an exhaustive, finalized plan.
   - Never invent or assume quantitative values for missing parameters.
   - Always provide useful scientific direction—do not refuse to answer.

6. **Accessible Language**: Explain scientific concepts in clear language while maintaining accuracy. Use technical terms when needed but explain them.

## Response Format

### Case A: When Vital Data is Incomplete (e.g., only 1-2 variables known, missing pH, rainfall, land use, or moisture):
Use this structure:

### Clarifying Questions
Ask 2–3 specific, targeted questions to gather the critical missing variables, explaining why each is needed.

### Preliminary Assessment & Missing Metric Impact
Summarize what is known, and explain scientifically how the missing factors (e.g. pH, water availability, land use) directly interact with the known problem.

### Provisional High-Level Guidance
Provide 2–3 provisional, evidence-grounded recommendations using the structured format:
For each recommendation:
- **Action**: What to do
- **Scientific Basis**: Why this works
- **Environmental Metrics Affected**: Which variables improve
- **Expected Timeline**: When to expect results
- **Measurable Improvement Estimate**: Evidence-grounded estimate from retrieved knowledge (quantitative ONLY if supported by evidence, otherwise qualitative)
- **Evidence**: Actual source name (e.g. FAO, IPCC, IPBES)

---

### Case B: When Sufficient Environmental Data is Provided:
Use this structure:

### Assessment
Summarize the environmental situation based on the data provided and identify compound concerns and multi-variable interactions.

### Recommendations
Provide 3–4 concrete, prioritized recommendations based on the compound constraints.
For each recommendation:
- **Action**: What to do
- **Scientific Basis**: Why this works
- **Environmental Metrics Affected**: Which variables improve
- **Expected Timeline**: When to expect results
- **Measurable Improvement Estimate**: Evidence-grounded estimate from retrieved knowledge (quantitative ONLY if supported by evidence, otherwise qualitative)
- **Evidence**: Actual source name (e.g. FAO, Brady & Weil, IPCC, USDA-NRCS)

### Confidence & Caveats
Note your confidence level and any important caveats or limitations.

### Additional Data Needed (if applicable)
What additional measurements would refine the plan further.
"""


class ConversationManager:
    """Manages conversations with memory, RAG retrieval, and Google Gemini LLM generation."""

    def __init__(self):
        self._retriever = EnvironmentalRetriever()
        self._client = genai.Client(api_key=GOOGLE_API_KEY)
        # conversation_id -> list of {"role": ..., "content": ...}
        self._conversations: dict[str, list[dict]] = {}
        # conversation_id -> accumulated environmental data dict across turns
        self._session_env_data: dict[str, dict] = {}

    def get_or_create_conversation(self, conversation_id: str) -> list[dict]:
        """Get or create a conversation history."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []
        return self._conversations[conversation_id]

    def clear_conversation(self, conversation_id: str):
        """Clear a conversation's history and accumulated environmental state."""
        self._conversations.pop(conversation_id, None)
        self._session_env_data.pop(conversation_id, None)

    def _prepare_pipeline(
        self,
        conversation_id: str,
        user_message: str,
        environmental_data: dict | None = None,
    ):
        """Shared pipeline preparation: progressive context accumulation, reasoning assessment, RAG retrieval, and prompt assembly."""
        history = self.get_or_create_conversation(conversation_id)

        # Step 1: Accumulate structured environmental data across multi-turn conversation
        current_env = dict(self._session_env_data.get(conversation_id, {}))
        if environmental_data:
            for k, v in environmental_data.items():
                if v is not None and v != "":
                    current_env[k] = v

        # Opportunistically extract environmental data from user message text if not yet provided
        text_data = extract_environmental_data_from_text(user_message)
        if text_data:
            for k, v in text_data.items():
                if k not in current_env:
                    current_env[k] = v

        self._session_env_data[conversation_id] = current_env

        env_input = None
        assessment = None
        env_context = ""

        if current_env:
            env_input = parse_environmental_input(current_env)
            assessment = assess_environmental_conditions(env_input)
            env_context = build_environmental_context(env_input, assessment)
        else:
            # Evaluate empty input to generate targeted clarifying questions
            env_input = EnvironmentalInput()
            assessment = assess_environmental_conditions(env_input)

        # Step 2: Retrieve relevant knowledge via RAG
        retrieval_query = user_message
        if env_input and assessment and assessment["provided_variables"]:
            retrieval_query += f" {' '.join(assessment['provided_variables'])}"
            if assessment["concerns"]:
                retrieval_query += f" {' '.join(assessment['concerns'])}"

        retrieved_docs = self._retriever.retrieve(retrieval_query, n_results=10)

        # Build RAG context
        rag_context = self._format_retrieved_context(retrieved_docs)

        # Step 3: Build the full prompt
        full_prompt = self._build_prompt(
            user_message=user_message,
            env_context=env_context,
            rag_context=rag_context,
            history=history,
            clarifying_questions=assessment["clarifying_questions"] if assessment else [],
        )

        # Extract source references from retrieved docs
        sources = list({
            doc["metadata"].get("source", "")
            for doc in retrieved_docs
            if doc["metadata"].get("source")
        })

        return history, assessment, retrieved_docs, full_prompt, sources

    def _finalize_response(
        self,
        history: list[dict],
        user_message: str,
        assistant_response: str,
        assessment: dict | None,
        sources: list[str],
    ) -> dict:
        """Update history, clean formatting, and parse structured recommendations."""
        import re
        clean_response = re.sub(r'\(?\s*Knowledge Entry \d+\s*\)?', '', assistant_response)

        # Step 5: Update conversation history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": clean_response})

        # Trim history if too long
        if len(history) > MAX_CONVERSATION_HISTORY:
            history[:] = history[-MAX_CONVERSATION_HISTORY:]

        # Extract structured recommendations and confidence
        recommendations, confidence = self._parse_recommendations_from_response(clean_response)

        return {
            "response": clean_response,
            "concerns": assessment["concerns"] if assessment else [],
            "clarifying_questions": assessment["clarifying_questions"] if assessment else [],
            "sources": sources,
            "assessment": {
                "provided_variables": assessment["provided_variables"] if assessment else [],
                "missing_variables": assessment["missing_variables"] if assessment else [],
                "interactions": assessment["interactions"] if assessment else [],
            } if assessment else None,
            "recommendations": recommendations,
            "confidence": confidence,
        }

    def process_message(
        self,
        conversation_id: str,
        user_message: str,
        environmental_data: dict | None = None,
    ) -> dict:
        """
        Process a user message through the full pipeline:
        1. Parse structured environmental input (if provided)
        2. Assess environmental conditions and flag concerns
        3. Retrieve relevant knowledge from vector DB
        4. Build Gemini contents with RAG context and history
        5. Send to Gemini
        6. Return structured response
        """
        history, assessment, retrieved_docs, contents, sources = self._prepare_pipeline(
            conversation_id=conversation_id,
            user_message=user_message,
            environmental_data=environmental_data,
        )

        last_err = None
        assistant_response = ""
        for attempt in range(_RETRY_MAX_ATTEMPTS):
            try:
                resp = self._client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.3,
                    ),
                )
                assistant_response = resp.text or ""
                last_err = None
                break
            except Exception as e:
                last_err = e
                err_str = str(e)
                if _is_retryable_error(err_str) and attempt < _RETRY_MAX_ATTEMPTS - 1:
                    wait = _retry_delay(attempt)
                    print(f"[Retry] Gemini error (attempt {attempt + 1}/{_RETRY_MAX_ATTEMPTS}): {err_str[:120]} — retrying in {wait:.1f}s")
                    time.sleep(wait)
                else:
                    break

        if last_err is not None:
            err_str = str(last_err)
            if "429" in err_str or "rate limit" in err_str.lower() or "resource_exhausted" in err_str.lower():
                assistant_response = "The AI provider is rate-limited. Please try again in a few moments."
            elif "503" in err_str or "unavailable" in err_str.lower():
                assistant_response = "The AI provider is temporarily unavailable (503). Please try again in a moment."
            else:
                assistant_response = f"I encountered an error contacting the AI provider: {err_str}. Please try again."

        return self._finalize_response(history, user_message, assistant_response, assessment, sources)

    def process_message_stream(
        self,
        conversation_id: str,
        user_message: str,
        environmental_data: dict | None = None,
    ):
        """
        Process a user message through the pipeline, streaming Gemini chunks in real time.
        Yields events as dicts:
            {"type": "chunk", "text": "..."}
            {"type": "error", "error": "..."}
            {"type": "final", "data": {...}}
        """
        history, assessment, retrieved_docs, contents, sources = self._prepare_pipeline(
            conversation_id=conversation_id,
            user_message=user_message,
            environmental_data=environmental_data,
        )

        full_response_parts = []

        last_err = None
        for attempt in range(_RETRY_MAX_ATTEMPTS):
            last_err = None
            attempt_parts = []
            try:
                stream = self._client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.3,
                    ),
                )
                for chunk in stream:
                    chunk_text = chunk.text or ""
                    if chunk_text:
                        attempt_parts.append(chunk_text)
                        # Once we start yielding chunks we are committed —
                        # do NOT retry from here on a mid-stream failure.
                        yield {"type": "chunk", "text": chunk_text}
                # Stream completed successfully
                full_response_parts = attempt_parts
                break
            except Exception as e:
                last_err = e
                err_str = str(e)
                if attempt_parts:
                    # Chunks were already sent — cannot retry; report gracefully
                    full_response_parts = attempt_parts
                    if "429" in err_str or "rate limit" in err_str.lower() or "resource_exhausted" in err_str.lower():
                        notice = "Gemini rate-limited mid-stream. The response above may be incomplete."
                    elif "503" in err_str or "unavailable" in err_str.lower():
                        notice = "Gemini became temporarily unavailable mid-stream. The response above may be incomplete."
                    else:
                        notice = f"Connection lost mid-stream: {err_str}"
                    yield {"type": "error", "error": notice}
                    full_response_parts.append(f"\n\n[Service notice: {notice}]")
                    last_err = None  # already handled
                    break
                # No chunks sent yet — retryable
                if _is_retryable_error(err_str) and attempt < _RETRY_MAX_ATTEMPTS - 1:
                    wait = _retry_delay(attempt)
                    print(f"[Retry-stream] Gemini error (attempt {attempt + 1}/{_RETRY_MAX_ATTEMPTS}): {err_str[:120]} — retrying in {wait:.1f}s")
                    time.sleep(wait)
                else:
                    break

        if last_err is not None:
            err_str = str(last_err)
            if "429" in err_str or "rate limit" in err_str.lower() or "resource_exhausted" in err_str.lower():
                user_friendly_error = "The AI provider is rate-limited. Please try again in a few moments."
            elif "503" in err_str or "unavailable" in err_str.lower():
                user_friendly_error = "The AI provider is temporarily unavailable (503). Please try again in a moment."
            else:
                user_friendly_error = f"Error connecting to AI provider: {err_str}"
            yield {"type": "error", "error": user_friendly_error}
            full_response_parts.append(f"\n\n[Service notice: {user_friendly_error}]")

        full_text = "".join(full_response_parts)
        final_result = self._finalize_response(history, user_message, full_text, assessment, sources)
        yield {"type": "final", "data": final_result}

    @staticmethod
    def _parse_recommendations_from_response(text: str) -> tuple[list[dict], str | None]:
        """Extract structured recommendations and confidence from markdown response."""
        import re
        recommendations = []
        conf_match = re.search(r'\bConfidence\b[^\n:]*?[:\-]?\s*(?:is\s*)?\*{0,2}(High|Moderate|Low)\b', text, re.I)
        top_confidence = conf_match.group(1).capitalize() if conf_match else None

        rec_sec_match = re.search(
            r'###\s*(?:Provisional\s+(?:High-Level\s+)?Guidance|Recommendations?|Recommended\s+Interventions?)(.*?)(?:###\s*(?:Confidence|Additional|Impact|Preliminary)|---\s*\n\s*###|\Z)',
            text, re.S | re.I
        )
        if not rec_sec_match:
            return recommendations, top_confidence

        rec_section = rec_sec_match.group(1)

        # Robust action split: matches **Action**, **Action 1**, 1. **Action**, #### Action 1:, #### 1. Action:, etc.
        action_pattern = r'(?:^|\n)\s*(?:####?\s*)?(?:[-*]|\d+\.)?\s*(?:####?\s*)?\*{0,2}Action(?:\s*\d+)?\s*[:\-]?\*{0,2}\s*[:\-]?\s*'
        action_splits = re.split(action_pattern, rec_section, flags=re.I)

        for block in action_splits[1:]:
            first_line = block.strip().split('\n')[0].strip()
            # Clean trailing asterisks, colons, or dashes from action title
            clean_action = re.sub(r'^\*{0,2}[:\-]?\s*', '', first_line).strip()
            clean_action = re.sub(r'\*{1,2}$', '', clean_action).strip()
            clean_action = re.sub(r'^\*{1,2}', '', clean_action).strip()

            basis_match = re.search(
                r'\*\*(?:Scientific\s+)?Basis\s*[:\-]?\*\*\s*[:\-]?\s*(.*?)(?=\n\s*[-*]\s*\*\*|\n\s*####?\s+|\Z)',
                block, re.S | re.I,
            )
            basis = basis_match.group(1).strip() if basis_match else ''

            metrics_match = re.search(
                r'\*\*(?:Environmental\s+)?Metrics(?:\s*Affected)?\s*[:\-]?\*\*\s*[:\-]?\s*(.*?)(?=\n\s*[-*]\s*\*\*|\n\s*####?\s+|\Z)',
                block, re.S | re.I,
            )
            metrics_raw = metrics_match.group(1).strip() if metrics_match else ''
            metrics = [m.strip().strip('*').strip('`').rstrip('.') for m in re.split(r'[,;]|\band\b', metrics_raw) if m.strip()]

            time_match = re.search(
                r'\*\*(?:Expected\s+)?(?:Timeline|Time Horizon)\s*[:\-]?\*\*\s*[:\-]?\s*(.*?)(?=\n\s*[-*]\s*\*\*|\n\s*####?\s+|\Z)',
                block, re.S | re.I,
            )
            time_horizon = time_match.group(1).strip().rstrip('.') if time_match else None

            meas_match = re.search(
                r'\*\*(?:Measurable\s+)?(?:Improvement|Estimate|Quantitative\s+Estimate)(?:\s+Estimate)?(?:\s*\(.*?\))?\s*[:\-]?\*\*\s*[:\-]?\s*(.*?)(?=\n\s*[-*]\s*\*\*|\n\s*####?\s+|\Z)',
                block, re.S | re.I,
            )
            measurable_improvement = meas_match.group(1).strip().rstrip('.') if meas_match else None

            evidence_match = re.search(
                r'\*\*(?:Evidence|Source)\s*[:\-]?\*\*\s*[:\-]?\s*(.*?)(?=\n\s*[-*]\s*\*\*|\n\s*####?\s+|\Z)',
                block, re.S | re.I,
            )
            evidence = evidence_match.group(1).strip().rstrip('.') if evidence_match else None

            item_conf_match = re.search(
                r'\*\*Confidence\s*[:\-]?\*\*\s*[:\-]?\s*\*{0,2}(High|Moderate|Low)\*{0,2}',
                block, re.I,
            )
            item_confidence = item_conf_match.group(1).capitalize() if item_conf_match else top_confidence

            if clean_action and (basis or metrics):
                recommendations.append({
                    'action': clean_action,
                    'scientific_basis': basis,
                    'scientific_reasoning': basis,
                    'metrics_affected': metrics,
                    'impacted_environmental_metrics': metrics,
                    'time_horizon': time_horizon,
                    'measurable_improvement': measurable_improvement,
                    'confidence': item_confidence,
                    'evidence': evidence,
                    'scientific_reference': evidence,
                })

        # Fallback: If no recommendations were parsed via action splits, check for numbered bold items
        if not recommendations:
            numbered_matches = re.findall(
                r'(?:^|\n)\s*(?:[-*]|\d+\.)\s*\*\*(.*?)\*\*\s*[:\-]?\s*(.*?)(?=\n\s*(?:[-*]|\d+\.)\s*\*\*|\n\s*###|\Z)',
                rec_section, re.S
            )
            for title, desc in numbered_matches:
                clean_title = title.strip().rstrip(':').strip()
                clean_desc = desc.strip()
                if clean_title and len(clean_title) > 3:
                    recommendations.append({
                        'action': clean_title,
                        'scientific_basis': clean_desc,
                        'scientific_reasoning': clean_desc,
                        'metrics_affected': [],
                        'impacted_environmental_metrics': [],
                        'time_horizon': None,
                        'measurable_improvement': None,
                        'confidence': top_confidence,
                        'evidence': None,
                        'scientific_reference': None,
                    })

        return recommendations, top_confidence

    def _format_retrieved_context(self, docs: list[dict]) -> str:
        """Format retrieved documents into a context string for the LLM."""
        if not docs:
            return "No relevant knowledge base entries found for this query."

        parts = ["## Retrieved Knowledge Base Context\n"]
        for i, doc in enumerate(docs, 1):
            meta = doc["metadata"]
            source = meta.get("source", "General Scientific Reference")
            topic = meta.get("topic", "Ecosystem & Soil Science")
            doc_type = meta.get("doc_type", "reference")
            parts.append(f"### Source: {source}")
            parts.append(f"Topic: {topic} ({doc_type})")
            parts.append(doc["document"])
            parts.append("")

        return "\n".join(parts)

    def _build_prompt(
        self,
        user_message: str,
        env_context: str,
        rag_context: str,
        history: list[dict],
        clarifying_questions: list[str],
    ) -> list[types.Content]:
        """Build the conversation prompt for Gemini as types.Content list."""
        contents = []

        # Add conversation history (last 10 turns for context)
        for msg in history[-10:]:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])],
            ))

        # Build the current user turn with full context
        user_parts = []

        if env_context:
            user_parts.append(f"--- ENVIRONMENTAL DATA PROVIDED ---\n{env_context}\n--- END ENVIRONMENTAL DATA ---\n")

        user_parts.append(f"--- RETRIEVED KNOWLEDGE ---\n{rag_context}\n--- END RETRIEVED KNOWLEDGE ---\n")

        if clarifying_questions:
            user_parts.append(
                "--- SYSTEM NOTE ---\n"
                "The following data gaps were identified. Consider asking the user about these "
                "if they are important for your answer:\n"
                + "\n".join(f"- {q}" for q in clarifying_questions)
                + "\n--- END SYSTEM NOTE ---\n"
            )

        user_parts.append(f"User Question: {user_message}")
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text="\n".join(user_parts))],
        ))

        return contents

    def get_retriever_info(self) -> dict:
        """Get information about the knowledge retriever."""
        return self._retriever.get_collection_info()

    def rebuild_knowledge_index(self):
        """Rebuild the knowledge base index."""
        self._retriever.rebuild_index()
