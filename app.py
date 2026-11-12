import json
import time

import requests
import streamlit as st

from story_generator import generate_story

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="StoryForge AI", page_icon="📖", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.hero {padding: 1.5rem 2rem; border-radius: 18px; background: linear-gradient(135deg,#17152f,#24204b); color:white; margin-bottom:1rem;}
.hero h1 {font-size:2.5rem; margin:0;}
.hero p {color:#c9c5e8; margin:.5rem 0 0;}
.latency-badge {font-size:0.85rem; color:#888; margin-top:0.3rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>📖 StoryForge AI</h1>'
            '<p>Give your story a beginning. Let AI imagine what happens next.</p></div>',
            unsafe_allow_html=True)
st.caption("Powered by DistilGPT2 • AI-generated text may be incomplete or inaccurate.")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Generation Settings")

    # Inference mode selector
    inference_mode = st.radio(
        "Inference mode",
        ["Local PyTorch", "TorchServe API", "SageMaker Endpoint"],
        index=0,
        help="Select where inference runs.  Local PyTorch requires no server.  "
             "TorchServe requires the server to be running locally.  "
             "SageMaker requires a deployed endpoint + AWS credentials."
    )

    st.divider()
    length_label = st.select_slider(
        "Continuation length",
        options=["Short", "Medium", "Long"],
        value="Medium",
    )
    token_map = {"Short": 50, "Medium": 100, "Long": 150}
    max_new_tokens = token_map[length_label]

    temperature = st.slider("Creativity (temperature)", 0.2, 1.5, 0.8, 0.1)
    top_p = st.slider("Nucleus sampling (top_p)", 0.1, 1.0, 0.9, 0.05)
    st.info("Higher temperature usually makes output more varied, but less predictable.")

    # TorchServe URL (only when TorchServe mode selected)
    if inference_mode == "TorchServe API":
        st.divider()
        torchserve_url = st.text_input(
            "TorchServe prediction URL",
            value="http://localhost:8080/predictions/storymodel",
        )

    # SageMaker settings (only when SageMaker mode selected)
    if inference_mode == "SageMaker Endpoint":
        st.divider()
        sm_endpoint_name = st.text_input("SageMaker endpoint name", value="")
        sm_region = st.text_input("AWS region", value="us-east-1")
        st.caption("AWS credentials must be configured via `aws configure` or "
                   "environment variables.  Never paste keys here.")


# ── Inference functions ──────────────────────────────────────────────────────
def infer_local(prompt, max_tokens, temp, tp):
    """Run inference using local PyTorch model."""
    return generate_story(prompt, max_tokens, temp, tp)


def infer_torchserve(prompt, max_tokens, temp, tp, url):
    """Send request to a TorchServe REST endpoint."""
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temp,
        "top_p": tp,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # TorchServe handler returns [{"generated_text": "..."}] or {"generated_text": "..."}
    if isinstance(data, list):
        data = data[0]
    return data.get("generated_text", str(data))


def infer_sagemaker(prompt, max_tokens, temp, tp, endpoint_name, region):
    """Invoke a SageMaker endpoint via boto3."""
    import boto3
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    payload = json.dumps({
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temp,
        "top_p": tp,
    })
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    body = json.loads(response["Body"].read().decode("utf-8"))
    if isinstance(body, list):
        body = body[0]
    return body.get("generated_text", str(body))


# ── Main layout ──────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Your story opening")
    prompt = st.text_area(
        "Start writing here…",
        height=260,
        placeholder="One rainy evening, a student found a mysterious key...",
    )
    generate = st.button("✨ Continue my story", type="primary", use_container_width=True)

with right:
    st.subheader("AI continuation")
    if generate:
        if not prompt.strip():
            st.error("Please enter a story opening first.")
        elif len(prompt) > 4000:
            st.error("Please keep the story opening under 4,000 characters.")
        else:
            with st.spinner("The model is imagining what happens next…"):
                try:
                    start_time = time.time()

                    if inference_mode == "Local PyTorch":
                        result = infer_local(prompt.strip(), max_new_tokens,
                                             temperature, top_p)
                    elif inference_mode == "TorchServe API":
                        result = infer_torchserve(prompt.strip(), max_new_tokens,
                                                  temperature, top_p, torchserve_url)
                    else:  # SageMaker
                        if not sm_endpoint_name.strip():
                            st.error("Please enter a SageMaker endpoint name in the sidebar.")
                            st.stop()
                        result = infer_sagemaker(prompt.strip(), max_new_tokens,
                                                 temperature, top_p,
                                                 sm_endpoint_name.strip(), sm_region)

                    latency_ms = (time.time() - start_time) * 1000
                    st.session_state["generated"] = result
                    st.session_state["source_prompt"] = prompt.strip()
                    st.session_state["latency"] = latency_ms
                    st.session_state["mode_used"] = inference_mode

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to TorchServe.  Is the server running?")
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")
                    if inference_mode == "Local PyTorch":
                        st.info("The first run downloads the model; check your internet connection.")

    if st.session_state.get("generated"):
        st.text_area("Generated text", value=st.session_state["generated"], height=260)

        # Latency badge
        if st.session_state.get("latency"):
            mode = st.session_state.get("mode_used", "")
            lat = st.session_state["latency"]
            st.markdown(
                f'<div class="latency-badge">⏱ {lat:.0f} ms · {mode}</div>',
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download continuation",
                               st.session_state["generated"],
                               file_name="story_continuation.txt",
                               mime="text/plain")
        with col2:
            st.button("🔄 Generate another",
                      on_click=lambda: st.session_state.pop("generated", None))
    else:
        st.info("Your continuation will appear here.")

st.divider()
st.markdown(
    "**Note:** DistilGPT2 is a general text-generation model, not a fact-checked "
    "or instruction-following system. Review its output."
)
