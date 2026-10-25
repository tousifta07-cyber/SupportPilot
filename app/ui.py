
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/ask"
TICKETS_URL = "http://127.0.0.1:8000/tickets"
HEALTH_URL = "http://127.0.0.1:8000/health"

REQUEST_TIMEOUT_SECONDS = 660


st.set_page_config(
    page_title="SupportPilot AI",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 SupportPilot AI")
st.caption(
    "IBM technical support assistant powered by document retrieval "
    "and evidence-grounded AI answers."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Knowledge Base")
    st.write("Search your indexed IBM technical documents.")

    if st.button("Check API status", use_container_width=True):
        try:
            response = requests.get(HEALTH_URL, timeout=10)
            response.raise_for_status()
            health = response.json()

            st.success("API is connected.")
            st.write(f"Collection: {health['collection']}")
            st.write(f"Indexed chunks: {health['indexed_chunks']:,}")

        except requests.RequestException:
            st.error("Could not connect to the SupportPilot API.")

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Check AI answers against the cited source passages.")


def display_sources(sources):
    """Display source passages returned by the API."""
    if not sources:
        return

    with st.expander(f"View retrieved sources ({len(sources)})"):
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**Source {index}: `{source.get('source_id', 'Unknown')}`**"
            )

            chunk_index = source.get("chunk_index")
            if chunk_index is not None:
                st.caption(f"Chunk index: {chunk_index}")

            score = source.get("rerank_score")
            if score is not None:
                st.caption(f"Reranker score: {score:.4f}")

            st.write(source.get("text", ""))

            if index < len(sources):
                st.divider()


# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("sources", []))


# Show ticket creation after a conversation has an answer.
latest_user = next(
    (
        message for message in reversed(st.session_state.messages)
        if message["role"] == "user"
    ),
    None,
)

latest_assistant = next(
    (
        message for message in reversed(st.session_state.messages)
        if message["role"] == "assistant"
    ),
    None,
)

if latest_user and latest_assistant:
    st.divider()
    st.subheader("Need help from human support?")
    st.write(
        "Create a support ticket if the answer did not resolve your issue."
    )

    with st.form("support_ticket_form", clear_on_submit=True):
        description = st.text_area(
            "Describe your issue",
            placeholder="Add the error message or explain what went wrong.",
            max_chars=5000,
        )

        priority = st.selectbox(
            "Priority",
            options=["low", "normal", "high", "urgent"],
            index=1,
        )

        submitted = st.form_submit_button(
            "Create support ticket",
            use_container_width=True,
        )

        if submitted:
            payload = {
                "question": latest_user["content"],
                "description": description,
                "priority": priority,
                "assistant_answer": latest_assistant["content"][:10000],
            }

            try:
                response = requests.post(
                    TICKETS_URL,
                    json=payload,
                    timeout=15,
                )

                if response.status_code == 201:
                    ticket = response.json()

                    st.success("Support ticket created successfully!")
                    st.write(f"**Ticket ID:** `{ticket['ticket_id']}`")
                    st.write(f"**Status:** {ticket['status']}")
                    st.write(f"**Priority:** {ticket['priority']}")

                else:
                    try:
                        details = response.json().get(
                            "detail",
                            response.text,
                        )
                    except ValueError:
                        details = response.text

                    st.error(
                        f"Ticket creation failed "
                        f"(HTTP {response.status_code}): {details}"
                    )

            except requests.RequestException as exc:
                st.error(f"Could not create the ticket: {exc}")


# Accept new questions.
question = st.chat_input(
    "Ask an IBM technical support question..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": [],
    })

    try:
        with st.spinner(
            "Searching technical documents and generating an answer..."
        ):
            response = requests.post(
                API_URL,
                json={"question": question},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        if response.ok:
            data = response.json()

            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "sources": data.get("sources", []),
            })

        else:
            try:
                details = response.json().get(
                    "detail",
                    response.text,
                )
            except ValueError:
                details = response.text

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"Request failed (HTTP {response.status_code}): "
                    f"{details}"
                ),
                "sources": [],
            })

    except requests.exceptions.Timeout:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "The request timed out. Check the API terminal before "
                "trying again; the backend may still be processing."
            ),
            "sources": [],
        })

    except requests.exceptions.ConnectionError:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "Could not connect to SupportPilot. "
                "Check that the API is running on port 8000."
            ),
            "sources": [],
        })

    except requests.RequestException as exc:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Request failed: {exc}",
            "sources": [],
        })

    st.rerun()
