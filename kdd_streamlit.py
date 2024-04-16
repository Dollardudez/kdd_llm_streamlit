import streamlit as st
import json
import threading
from langchain_community.llms import HuggingFaceTextGenInference  # Updated import
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from streamlit.runtime.scriptrunner.script_run_context import add_script_run_ctx

if 'responses' not in st.session_state:
    st.session_state['responses'] = []

# Function to download data as JSON
def download_json(data, filename):
    json_data = json.dumps(data, indent=4)
    st.download_button(
        label="Download data as JSON",
        data=json_data,
        file_name=filename,
        mime="application/json"
    )

class StreamlitStreamCallbackHandler:
    def __init__(self, container):
        self.container = container

    def __call__(self, text):
        self.container.text_area("", value=text, height=100, key=f'response_{self.container}')


# Function to execute LLM response
def execute_llm_response(prompt, params, container, index, results):
    llm = HuggingFaceTextGenInference(
        inference_server_url=params["url"],
        max_new_tokens=params["max_new_tokens"],
        top_k=params["top_k"],
        top_p=params["top_p"],
        typical_p=params["typical_p"],
        temperature=params["temperature"],
        repetition_penalty=params["repetition_penalty"],
        # callbacks=callbacks,
        streaming=True
    )
    response = llm(prompt, callbacks=[StreamlitCallbackHandler(container)])
    results[index] = {
        "prompt": prompt,
        "config": params,
        "response": response
    }
    return response

# Streamlit UI setup
st.title("LLM Comparison Tool")
st.markdown("Send the same prompt to multiple LLMs and compare their outputs.")

# Collecting configurations for each LLM
number_of_llms = st.sidebar.number_input("Number of LLMs", min_value=1, max_value=10, value=2)
llm_configs = []
for i in range(number_of_llms):
    st.sidebar.header(f"LLM {i+1} Configuration")
    llm_config = {
        "url": st.sidebar.text_input(f"URL for LLM {i+1}", value=f"http://localhost:6000{i}/"),
        "max_new_tokens": st.sidebar.number_input(f"Max New Tokens for LLM {i+1}", value=10, min_value=1),
        "top_k": st.sidebar.number_input(f"Top K for LLM {i+1}", value=50, min_value=0),
        "top_p": st.sidebar.number_input(f"Top P for LLM {i+1}", value=0.9, min_value=0.0, max_value=1.0),
        "typical_p": st.sidebar.number_input(f"Typical P for LLM {i+1}", value=0.9, min_value=0.0, max_value=1.0),
        "temperature": st.sidebar.number_input(f"Temperature for LLM {i+1}", value=1.0, min_value=0.0, max_value=2.0),
        "repetition_penalty": st.sidebar.number_input(f"Repetition Penalty for LLM {i+1}", value=1.2, min_value=1.0, max_value=2.0)
    }
    llm_configs.append(llm_config)

# Text input for prompt
prompt = st.text_area("Enter your prompt:", value="", height=150)

response_containers = {}
for i in range(number_of_llms):
    llm_name = f"LLM {i + 1}"
    llm_url = llm_configs[i]['url']
    st.markdown(f"**Response from {llm_name} ({llm_url}):**")
    response_containers[i] = st.empty()

# Handle button click for LLM response
if st.button("Submit") and prompt:
    temp_responses = [{} for _ in range(number_of_llms)]
    threads = []
    for i, config in enumerate(llm_configs):
        thread = threading.Thread(target=execute_llm_response, args=(prompt, config, response_containers[i], i, temp_responses))
        add_script_run_ctx(thread)
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    st.session_state['responses'].extend(temp_responses)

    # Trigger a rerun to display responses
    # st.experimental_rerun()

# Before accessing 'responses', check if it exists

download_json(st.session_state['responses'], "llm_responses.json")
