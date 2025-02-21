import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf

# Load Ganga-2-1B from local directory
@st.cache(allow_output_mutation=True)
def load_ganga_model():
    tokenizer_2 = AutoTokenizer.from_pretrained(r"C:\Users\gauth\Downloads\Ganga_model")
    model_2 = AutoModelForCausalLM.from_pretrained(r"C:\Users\gauth\Downloads\Ganga_model")
    return tokenizer_2, model_2

# Load Indic-TTS from local directory
@st.cache(allow_output_mutation=True)
def load_indic_tts():
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(r"C:\Users\gauth\Downloads\Indic_TTS_model")
    tts_tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\gauth\Downloads\Indic_TTS_model")
    description_tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\gauth\Downloads\Indic_TTS_model")
    return tts_model, tts_tokenizer, description_tokenizer

# Main app
def main():
    st.title("Hindi Text-to-Speech using Ganga and Indic-TTS")
    st.text("Enter Hindi text, generate additional text using Ganga, and convert it to speech.")

    # User Input (Empty by default)
    input_text = st.text_area("Enter Hindi Text", "")

    if st.button("Generate Speech"):
        if not input_text.strip():
            st.error("Please enter some text before generating speech.")
            return

        with st.spinner("Generating Text with Ganga..."):
            # Ganga-2-1B Inference
            tokenizer_2, model_2 = load_ganga_model()
            input_ids = tokenizer_2.encode(input_text, return_tensors="pt").to("cpu")  # Ensure CPU usage
            outputs = model_2.generate(input_ids, max_new_tokens=100, do_sample=True, top_k=10, top_p=0.95, temperature=0.7)
            final_output = tokenizer_2.decode(outputs[0])
        
        st.success(f"Generated Text: {final_output}")

        with st.spinner("Synthesizing Speech with Indic-TTS..."):
            # Indic-TTS Inference
            model, tokenizer, description_tokenizer = load_indic_tts()
            description = "Divya's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

            description_input_ids = description_tokenizer(description, return_tensors="pt").to("cpu")
            prompt_input_ids = tokenizer(final_output, return_tensors="pt").to("cpu")

            generation = model.generate(input_ids=description_input_ids.input_ids, 
                                        attention_mask=description_input_ids.attention_mask, 
                                        prompt_input_ids=prompt_input_ids.input_ids, 
                                        prompt_attention_mask=prompt_input_ids.attention_mask)
            audio_arr = generation.cpu().numpy().squeeze()
            sf.write("output_speech.wav", audio_arr, model.config.sampling_rate)

        # Displaying the audio output
        st.audio("output_speech.wav", format="audio/wav")

if __name__ == "__main__":
    main()
