# Required Libraries      
import streamlit as st      
import numpy as np      
import scipy.io.wavfile as wavfile      
import matplotlib.pyplot as plt      
import io      
import datetime      
import json      
import os      

# Gemini 2.5 Setup (NEW)      
import google.generativeai as genai  # Correct import
GEMINI_API_KEY = "AIzaSyDdGv--2i0pMbhH68heurl-LI1qJPJjzD4"  # Replace with your actual key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-2.5-flash")  # Set the correct model

# Streamlit Config      
st.set_page_config(page_title="AI PCG Diagnosis (Research Concept)", layout="wide")      

# Case database      
CASE_DB = "saved_cases.json"      
if not os.path.exists(CASE_DB):      
    with open(CASE_DB, "w") as f:      
        json.dump([], f)      

def load_cases():      
    with open(CASE_DB, "r") as f:      
        return json.load(f)      

def save_case(case):      
    cases = load_cases()      
    cases.append(case)      
    with open(CASE_DB, "w") as f:      
        json.dump(cases, f)      

# Diagnosis Logic (Simulated)      
SIMULATED_DIAGNOSES = {      
    "normal": "**Likely Diagnosis:** Normal Heart Sounds\n\n**Analysis:** Normal S1/S2, no murmurs.",      
    "as": "**Likely Diagnosis:** Aortic Stenosis (AS)\n\n**Analysis:** Crescendo-decrescendo midsystolic murmur.",      
    "ar": "**Likely Diagnosis:** Aortic Regurgitation (AR)\n\n**Analysis:** High-frequency decrescendo diastolic murmur.",      
    "ps": "**Likely Diagnosis:** Pulmonary Stenosis (PS)\n\n**Analysis:** Systolic murmur with ejection click.",      
    "pr": "**Likely Diagnosis:** Pulmonary Regurgitation (PR)\n\n**Analysis:** Early diastolic murmur with high pitch.",      
    "ms": "**Likely Diagnosis:** Mitral Stenosis (MS)\n\n**Analysis:** Low-frequency diastolic rumbling murmur.",      
    "mr": "**Likely Diagnosis:** Mitral Regurgitation (MR)\n\n**Analysis:** Holosystolic blowing murmur.",      
    "ts": "**Likely Diagnosis:** Tricuspid Stenosis (TS)\n\n**Analysis:** Diastolic murmur increases with inspiration.",      
    "tr": "**Likely Diagnosis:** Tricuspid Regurgitation (TR)\n\n**Analysis:** Holosystolic murmur at lower sternal border."      
}      

def get_simulated_diagnosis(audio_data, sample_rate, valve):      
    std_dev = np.std(audio_data)      
    peak_amp = np.max(np.abs(audio_data))      
    zero_crossings = np.count_nonzero(np.diff(np.sign(audio_data)))      
    st.info(f"SimLogic - {valve}: std_dev={std_dev:.0f}, peak={peak_amp:.0f}, zero_crossings={zero_crossings}")      

    if valve == "Aortic Valve":      
        if std_dev > 4000:      
            return SIMULATED_DIAGNOSES["as"]      
        elif peak_amp > 12000:      
            return SIMULATED_DIAGNOSES["ar"]      
    elif valve == "Pulmonary Valve":      
        if std_dev > 3500:      
            return SIMULATED_DIAGNOSES["ps"]      
        elif peak_amp > 11000:      
            return SIMULATED_DIAGNOSES["pr"]      
    elif valve == "Mitral Valve":      
        if 1500 < std_dev <= 3500:      
            return SIMULATED_DIAGNOSES["ms"]      
        elif std_dev <= 1500 and peak_amp > 10000:      
            return SIMULATED_DIAGNOSES["mr"]      
    elif valve == "Tricuspid Valve":      
        if std_dev < 1500 and zero_crossings > 7000:      
            return SIMULATED_DIAGNOSES["ts"]      
        elif peak_amp > 8000:      
            return SIMULATED_DIAGNOSES["tr"]      

    return SIMULATED_DIAGNOSES["normal"]      

# Gemini Diagnosis - Short and Focused
def diagnose_with_gemini_text_only(sim_report, valve):      
    try:      
        prompt = f"""You are a concise AI doctor assistant.
Analyze the following simulated phonocardiogram report for the {valve} and return only:

1. A short diagnosis (e.g., "Aortic Stenosis")
2. A short reason (1 sentence max)

---  
Simulated Report:  
{sim_report}  
---

Format strictly as:  
**Diagnosis:** ...  
**Reason:** ...
"""      
        response = model.generate_content(prompt)      
        return response.text.strip()      
    except Exception as e:      
        return f"Gemini Diagnosis Error: {e}"      

# Waveform Plotting      
def plot_waveform(sample_rate, audio_data, valve, amp_scale, noise_thresh, max_duration):      
    fig, ax = plt.subplots(figsize=(10, 2))      
    duration = len(audio_data) / sample_rate      
    time = np.linspace(0., duration, len(audio_data))      
    audio_data = audio_data * amp_scale      
    if noise_thresh > 0:      
        audio_data = np.where(np.abs(audio_data) < noise_thresh, 0, audio_data)      
    ax.plot(time, audio_data, lw=0.7)      
    ax.set_title(f"{valve} Waveform")      
    ax.set_xlim(0, min(duration, max_duration))      
    ax.grid(True)      
    return fig      

# --------------------- UI ---------------------      

st.title("🔥 AI Phonocardiography Analysis")      
st.warning("**RESEARCH PURPOSE ONLY.** This is a research concept for AI-based detection of valvular heart disease using phonocardiography.", icon="⚠️")      

# Sidebar Patient Info      
st.sidebar.header("🧑‍⚕️ Patient Info")      
name = st.sidebar.text_input("Name")      
age = st.sidebar.number_input("Age", 0, 120, 30)      
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])      
height = st.sidebar.number_input("Height (cm)", 50.0, 250.0, 170.0)      
weight = st.sidebar.number_input("Weight (kg)", 10.0, 200.0, 65.0)      
phone = st.sidebar.text_input("Phone")      
if height > 0:      
    bmi = weight / ((height / 100) ** 2)      
    st.sidebar.markdown(f"**BMI:** {bmi:.1f}")      

# Upload Section      
st.header("1. Upload PCG WAV files")      
cols = st.columns(4)      
valves = ["Aortic Valve", "Pulmonary Valve", "Mitral Valve", "Tricuspid Valve"]      
valve_files = {}      
for i, valve in enumerate(valves):      
    valve_files[valve] = cols[i].file_uploader(valve, type=["wav"])      

st.markdown("---")      
st.header("2. AI Analysis & Report")      

analysis_results = {}      

if st.button("🔬 Generate Diagnostic Report", type="primary"):      
    for valve, file in valve_files.items():      
        if file is not None:      
            st.subheader(f"{valve} Analysis")      
            sr, data = wavfile.read(io.BytesIO(file.getvalue()))      

            st.markdown("**Waveform Controls**")      
            c1, c2, c3 = st.columns(3)      
            max_dur = c1.slider(f"Max Duration ({valve})", 1, 10, 8)      
            amp = c2.slider(f"Amp Scale ({valve})", 0.1, 3.0, 1.0)      
            thresh = c3.slider(f"Noise Threshold ({valve})", 0, 1000, 0)      

            fig = plot_waveform(sr, data, valve, amp, thresh, max_dur)      
            st.pyplot(fig)      

            sim_report = get_simulated_diagnosis(data, sr, valve)      
            st.markdown("##### 🤖 Simulated Report")      
            st.write(sim_report)      
            analysis_results[valve] = sim_report      

            st.markdown("##### 🧠 Gemini Diagnosis")      
            gemini_result = diagnose_with_gemini_text_only(sim_report, valve)      
            st.success(gemini_result)      
            analysis_results[valve + "_gemini"] = gemini_result      
            st.markdown("---")      

# Save Case      
if st.button("💾 Save Case"):      
    case = {      
        "datetime": datetime.datetime.now().isoformat(),      
        "name": name,      
        "age": age,      
        "gender": gender,      
        "height": height,      
        "weight": weight,      
        "phone": phone,      
        "bmi": bmi,      
        "analysis": analysis_results      
    }      
    save_case(case)      
    st.success("Case saved successfully!")      

# View History      
if st.button("📂 View Case History"):      
    st.header("📁 Past Cases")      
    for case in reversed(load_cases()):      
        st.subheader(case["name"] + f" (Age {case['age']}, {case['gender']})")      
        st.caption(f"Recorded: {case['datetime']}")      
        st.markdown(f"**BMI:** {case['bmi']:.1f}")      
        for key, value in case.get("analysis", {}).items():      
            st.markdown(f"**{key}:**")      
            st.write(value)      
        st.markdown("---")
