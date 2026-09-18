"""
Streamlit frontend for the Darukaa.Earth Biodiversity Intelligence Chatbot.
"""

import os
import json
import streamlit as st
import requests
import uuid

# --- Configuration ---
API_URL = os.getenv("BACKEND_URL", os.getenv("API_URL", "http://localhost:8000"))

st.set_page_config(
    page_title="Darukaa.Earth — Biodiversity Intelligence",
    page_icon="🌍",
    layout="wide",
)

# --- Session State Initialization ---
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_env_panel" not in st.session_state:
    st.session_state.show_env_panel = False


def check_api_health() -> bool:
    """Check if the FastAPI backend is running."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def send_message(message: str, env_data: dict | None = None) -> dict | None:
    """Send a message to the FastAPI backend."""
    payload = {
        "message": message,
        "conversation_id": st.session_state.conversation_id,
    }
    if env_data:
        # Filter out empty values
        filtered = {k: v for k, v in env_data.items() if v is not None and v != "" and v != 0.0}
        if filtered:
            payload["environmental_data"] = filtered

    try:
        resp = requests.post(f"{API_URL}/chat", json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API Error: {resp.status_code} — {resp.text}")
            return None
    except requests.ConnectionError:
        st.error("Cannot connect to the backend API. Make sure it is running on http://localhost:8000")
        return None
    except requests.Timeout:
        st.error("Request timed out. The AI model may be taking too long to respond.")
        return None


def clear_conversation():
    """Clear the current conversation."""
    try:
        requests.post(f"{API_URL}/chat/{st.session_state.conversation_id}/clear", timeout=5)
    except Exception:
        pass
    st.session_state.messages = []
    st.session_state.conversation_id = str(uuid.uuid4())


# --- Layout ---
st.title("🌍 Darukaa.Earth — Biodiversity Intelligence")
st.caption("AI Environmental Scientist Assistant | Powered by Gemini + RAG")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🔄 New Conversation", use_container_width=True):
        clear_conversation()
        st.rerun()

    st.divider()

    # Toggle environmental data panel
    st.header("📊 Environmental Data Input")
    st.session_state.show_env_panel = st.toggle(
        "Provide structured environmental data",
        value=st.session_state.show_env_panel,
    )

    env_data = {}
    if st.session_state.show_env_panel:
        st.markdown("*Enter values for any parameters you have. Leave others blank.*")

        with st.expander("🌱 Soil Parameters", expanded=True):
            env_data["soil_organic_carbon"] = st.number_input(
                "Soil Organic Carbon (%)", min_value=0.0, max_value=30.0,
                value=None, step=0.1, format="%.1f",
                placeholder="e.g., 2.5",
            )
            env_data["soil_ph"] = st.number_input(
                "Soil pH", min_value=0.0, max_value=14.0,
                value=None, step=0.1, format="%.1f",
                placeholder="e.g., 6.5",
            )
            env_data["soil_moisture"] = st.number_input(
                "Soil Moisture (% field capacity)", min_value=0.0, max_value=100.0,
                value=None, step=1.0, format="%.0f",
                placeholder="e.g., 60",
            )

        with st.expander("🌧️ Climate Parameters"):
            env_data["rainfall"] = st.number_input(
                "Annual Rainfall (mm)", min_value=0.0, max_value=10000.0,
                value=None, step=10.0, format="%.0f",
                placeholder="e.g., 800",
            )
            env_data["temperature"] = st.number_input(
                "Average Temperature (°C)", min_value=-50.0, max_value=60.0,
                value=None, step=0.5, format="%.1f",
                placeholder="e.g., 28",
            )

        with st.expander("🌾 Land Use"):
            env_data["land_use"] = st.text_input(
                "Land Use Type",
                placeholder="e.g., rice monoculture, mixed farming",
            )
            env_data["crop_type"] = st.text_input(
                "Crop Type",
                placeholder="e.g., wheat, rice, maize",
            )
            env_data["region"] = st.text_input(
                "Region",
                placeholder="e.g., Western Ghats, Indo-Gangetic Plain",
            )

        with st.expander("🦋 Biodiversity & Pollution"):
            env_data["biodiversity_index"] = st.number_input(
                "Biodiversity Index (Shannon)", min_value=0.0, max_value=10.0,
                value=None, step=0.1, format="%.1f",
                placeholder="e.g., 2.5",
            )
            env_data["pollution_level"] = st.selectbox(
                "Pollution Level",
                options=["", "low", "moderate", "high", "severe"],
                index=0,
            )
            env_data["deforestation_rate"] = st.number_input(
                "Deforestation Rate (%/year)", min_value=0.0, max_value=100.0,
                value=None, step=0.1, format="%.1f",
                placeholder="e.g., 1.5",
            )

        env_data["additional_notes"] = st.text_area(
            "Additional Notes",
            placeholder="Any extra context about the environment...",
        )

    st.divider()

    # API Status
    api_ok = check_api_health()
    if api_ok:
        st.success("✅ Backend API connected")
    else:
        st.error("❌ Backend API not reachable. Start it with:\n\n`uvicorn app.api:app --reload`")

    st.divider()

    # Sample queries
    st.header("💡 Sample Queries")
    sample_queries = [
        "My farm soil has 0.8% organic carbon and pH 4.5. What should I do?",
        "How does deforestation affect local rainfall patterns?",
        "What are the best practices for improving biodiversity in a rice monoculture?",
        "Explain the relationship between soil organic carbon and water retention.",
        "I have a region with high pollution and low biodiversity. What interventions help?",
    ]
    for q in sample_queries:
        if st.button(q, key=f"sample_{hash(q)}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

def render_metadata_expanders(meta: dict):
    """Render structured metadata sections."""
    if not meta:
        return

    if meta.get("concerns"):
        with st.expander("⚠️ Environmental Concerns Identified"):
            for concern in meta["concerns"]:
                st.warning(concern)

    if meta.get("sources"):
        with st.expander("📚 Sources & References"):
            for source in meta["sources"]:
                st.markdown(f"- {source}")

    if meta.get("assessment") and meta["assessment"].get("interactions"):
        with st.expander("🔗 Cross-Variable Interactions"):
            for interaction in meta["assessment"]["interactions"]:
                st.info(interaction)

    if meta.get("clarifying_questions"):
        with st.expander("❓ Suggested Clarifying Questions"):
            for q in meta["clarifying_questions"]:
                st.markdown(f"- {q}")


# --- Chat Display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show metadata for assistant messages
        if msg["role"] == "assistant" and "metadata" in msg:
            render_metadata_expanders(msg["metadata"])


# --- Handle pending sample query ---
pending = st.session_state.pop("pending_query", None)

# --- Chat Input ---
user_input = st.chat_input("Ask an environmental or biodiversity question...")

# Use pending query if set
if pending and not user_input:
    user_input = pending

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare environmental data if panel is open
    send_env_data = None
    if st.session_state.show_env_panel:
        send_env_data = env_data

    # Send to streaming API
    with st.chat_message("assistant"):
        stream_state = {"final_data": None, "error": None}

        def stream_chat_generator():
            payload = {
                "message": user_input,
                "conversation_id": st.session_state.conversation_id,
            }
            if send_env_data:
                filtered = {k: v for k, v in send_env_data.items() if v is not None and v != "" and v != 0.0}
                if filtered:
                    payload["environmental_data"] = filtered

            try:
                resp = requests.post(f"{API_URL}/chat/stream", json=payload, stream=True, timeout=120)
                if resp.status_code != 200:
                    err_msg = f"API Error ({resp.status_code}): {resp.text}"
                    stream_state["error"] = err_msg
                    yield f"⚠️ {err_msg}"
                    return

                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue

                    msg_type = item.get("type")
                    if msg_type == "chunk":
                        yield item.get("text", "")
                    elif msg_type == "final":
                        stream_state["final_data"] = item.get("data", {})
                    elif msg_type == "error":
                        stream_state["error"] = item.get("error", "Unknown error")
                        yield f"\n\n⚠️ {item.get('error')}"

            except requests.ConnectionError:
                err = "Cannot connect to the backend API. Make sure it is running on http://localhost:8000"
                stream_state["error"] = err
                yield f"⚠️ {err}"
            except requests.Timeout:
                err = "Request timed out. The AI model may be taking too long to respond."
                stream_state["error"] = err
                yield f"⚠️ {err}"
            except Exception as e:
                err = f"An unexpected error occurred: {str(e)}"
                stream_state["error"] = err
                yield f"⚠️ {err}"

        # Stream text progressively using Streamlit's native write_stream
        full_text = st.write_stream(stream_chat_generator())

        # Collect metadata from backend final payload
        final_meta = {}
        if stream_state["final_data"]:
            data = stream_state["final_data"]
            final_meta = {
                "concerns": data.get("concerns", []),
                "sources": data.get("sources", []),
                "clarifying_questions": data.get("clarifying_questions", []),
                "assessment": data.get("assessment"),
                "recommendations": data.get("recommendations", []),
                "confidence": data.get("confidence"),
            }

        # Store assistant message with metadata
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_text,
            "metadata": final_meta,
        })

        # Render structured metadata expanders
        render_metadata_expanders(final_meta)
