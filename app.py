import asyncio
import streamlit as st
from agent_manager import AgentManager

# Page config
st.set_page_config(page_title="VibeCurator AI Assistant", page_icon="🎵", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_manager" not in st.session_state:
    st.session_state.agent_manager = AgentManager(
        name="VibeCurator Assistant",
        instructions="You are a helpful assistant specialized in music curation and recommendations.",
        model="gpt-5-nano",
    )

# Header
st.title("🎵 VibeCurator AI Assistant")
st.caption("Chat with your AI music curator")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        async def stream_and_display():
            full_response = ""
            async for delta in st.session_state.agent_manager.stream_response(prompt):
                full_response += delta
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            return full_response

        try:
            response = asyncio.run(stream_and_display())
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

# Sidebar
with st.sidebar:
    st.header("Settings")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("Agent Configuration")
    st.text(f"Model: {st.session_state.agent_manager.model_name}")
    st.text(f"Name: {st.session_state.agent_manager.agent_name}")

    st.divider()

    st.markdown("""
    ### About
    VibeCurator AI is your intelligent music curation assistant.
    
    Ask questions, get recommendations, or chat about music!
    """)
