import plotly.express as px
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
from thefuzz import process

# Page config
st.set_page_config(
    page_title="AI Job Impact Predictor",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .big-title { font-size: 3.8rem; font-weight: 800; text-align: center;
                 color: #0072ff;
                 margin-bottom: 0.1rem; line-height: 1.1; }
    .subtitle { text-align: center; color: #444; font-size: 1.2rem;
                margin-bottom: 0.75rem; font-weight: 400; }
    .fact-box { background: #e8f4fd; border-left: 4px solid #0072ff;
                padding: 10px 14px; border-radius: 8px; margin: 0.5rem 0;
                color: #1a1a1a; font-size: 0.95rem; }
    .block-container { padding-top: 1.5rem !important; }
    hr { margin: 0.75rem 0 !important; }
    input[type="text"]::placeholder { color: #666 !important; }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_resource
def load_model():
    model = joblib.load('job_risk_model.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, le

@st.cache_data
def load_data():
    df = pd.read_csv('data/automation_data_by_state.csv', encoding='latin-1')
    state_columns = df.columns[3:54].tolist()
    df[state_columns] = df[state_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    df['soc_group'] = df['SOC'].str[:2]
    df['total_employment'] = df[state_columns].sum(axis=1)
    df['risk_category'] = df['Probability'].apply(
        lambda x: 'Low Risk' if x <= 0.33 else 'Medium Risk' if x <= 0.66 else 'High Risk'
    )
    df['category_name'] = df['SOC'].str[:2].map({
        '11': 'Management', '13': 'Business & Finance', '15': 'Computer & Math',
        '17': 'Architecture & Eng.', '19': 'Life & Physical Science',
        '21': 'Community & Social Svc.', '23': 'Legal', '25': 'Education',
        '27': 'Arts & Media', '29': 'Healthcare Practitioners', '31': 'Healthcare Support',
        '33': 'Protective Service', '35': 'Food Preparation', '37': 'Building & Grounds',
        '39': 'Personal Care', '41': 'Sales', '43': 'Office & Admin',
        '45': 'Farming & Fishing', '47': 'Construction', '49': 'Installation & Repair',
        '51': 'Production & Mfg.', '53': 'Transportation'
    })
    return df

model, le = load_model()
df = load_data()

soc_labels = {
    '11': 'Management', '13': 'Business & Finance', '15': 'Computer & Math',
    '17': 'Architecture & Eng.', '19': 'Life & Physical Science',
    '21': 'Community & Social Svc.', '23': 'Legal', '25': 'Education',
    '27': 'Arts & Media', '29': 'Healthcare Practitioners', '31': 'Healthcare Support',
    '33': 'Protective Service', '35': 'Food Preparation', '37': 'Building & Grounds',
    '39': 'Personal Care', '41': 'Sales', '43': 'Office & Admin',
    '45': 'Farming & Fishing', '47': 'Construction', '49': 'Installation & Repair',
    '51': 'Production & Mfg.', '53': 'Transportation'
}

facts = [
    "🤖 McKinsey estimates up to 30% of current work tasks could be automated by 2030.",
    "💡 Jobs requiring creativity, empathy, and complex judgment are the hardest to automate.",
    "📊 Higher-wage occupations tend to have significantly lower automation susceptibility.",
    "🔧 Routine, repetitive, and highly structured tasks are easiest for AI to replicate.",
    "🌐 The World Economic Forum projects AI will displace some roles while creating new ones requiring digital and analytical skills.",
    "🎓 Occupations requiring advanced degrees and specialized judgment show lower automation susceptibility.",
    "⚕️ Healthcare roles involving direct patient care often have lower estimated automation susceptibility.",
    "💻 Technical roles that build and maintain AI systems tend to have lower automation exposure — though this varies significantly by specialization.",
]

# Header
st.markdown('<p class="big-title">🤖 AI Job Impact Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Helping students make smarter career decisions in the age of AI</p>', unsafe_allow_html=True)

# Did you know fact
st.markdown(f'<div class="fact-box">💡 <b>Did you know?</b> {random.choice(facts)}</div>', unsafe_allow_html=True)

st.markdown("---")

# Top 5 safest and riskiest
with st.expander("📊 See Top 5 Safest & Riskiest Jobs"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🟢 Top 5 Safest")
        safest = df.nsmallest(5, 'Probability')[['Occupation', 'Probability']]
        for _, row in safest.iterrows():
            st.markdown(f"**{row['Occupation'][:40]}**  \n`{row['Probability']:.1%} risk`")
    with col2:
        st.markdown("### 🔴 Top 5 Riskiest")
        riskiest = df.nlargest(5, 'Probability')[['Occupation', 'Probability']]
        for _, row in riskiest.iterrows():
            st.markdown(f"**{row['Occupation'][:40]}**  \n`{row['Probability']:.1%} risk`")

st.markdown("---")

# Job search with fuzzy matching
st.markdown("### 🔍 Find Your Career Risk")
search_input = st.text_input("Search by job title", placeholder="e.g. Electrical Engineer, Pharmacist, Registered Nurse")
st.caption("Start typing to find the closest matching occupation from 702 US job categories.")

if search_input:
    matches = process.extract(search_input, df['Occupation'].tolist(), limit=8)
    match_options = [m[0] for m in matches if m[1] > 40]
    if match_options:
        selected_job = st.selectbox("Select the closest match:", [''] + match_options)
    else:
        st.warning("No close matches found. Try a different search term.")
        selected_job = ''
else:
    all_jobs = [''] + sorted(df['Occupation'].tolist())
    selected_job = st.selectbox("Or browse all occupations:", all_jobs)

if selected_job:
    job_data = df[df['Occupation'] == selected_job].iloc[0]
    actual_prob = job_data['Probability']
    soc_grp = job_data['soc_group']
    category_name = soc_labels.get(soc_grp, 'Related')

    st.markdown("---")

    # Result card
    rc1, rc2, rc3 = st.columns([2, 1, 1])
    with rc1:
        st.markdown(f"## {selected_job}")
        st.caption(f"Category: {category_name}")
        if actual_prob > 0.66:
            st.error("🔴 High Automation Susceptibility")
        elif actual_prob > 0.33:
            st.warning("🟠 Medium Automation Susceptibility")
        else:
            st.success("🟢 Low Automation Susceptibility")
        st.progress(float(actual_prob))
        st.caption(f"Susceptibility score: {actual_prob:.1%} — Frey & Osborne (2017)")

    with rc2:
        st.metric("Susceptibility Score", f"{actual_prob:.1%}")
        st.caption("From the Frey & Osborne (2017) task-based study via O*NET data.")

    with rc3:
        if actual_prob <= 0.33:
            st.success("✅ Strong long-term outlook. Focus on skills that complement AI.")
        elif actual_prob <= 0.66:
            st.warning("⚠️ Moderate susceptibility. Develop skills harder to automate.")
        else:
            st.error("🚨 High susceptibility. Explore related roles with lower risk.")

    # Career recommendations
    if actual_prob > 0.33:
        st.markdown("---")
        st.markdown("### 💡 Lower-Risk Alternatives in the Same Field")
        safer = df[
            (df['soc_group'] == soc_grp) &
            (df['Probability'] < actual_prob) &
            (df['Occupation'] != selected_job)
        ].nsmallest(3, 'Probability')[['Occupation', 'Probability']]

        if safer.empty:
            st.info("No lower-risk alternatives found in this category.")
        else:
            for _, row in safer.iterrows():
                st.markdown(f"🟢 **{row['Occupation']}** — `{row['Probability']:.1%} susceptibility`")

    # Related jobs
    st.markdown("---")
    st.markdown(f"### 🏢 How does this compare to other **{category_name}** occupations?")
    related = df[df['soc_group'] == soc_grp][['Occupation', 'Probability']].sort_values('Probability')
    related['Risk'] = related['Probability'].apply(
        lambda x: '🟢 Low' if x <= 0.33 else '🟠 Medium' if x <= 0.66 else '🔴 High'
    )
    st.dataframe(related.rename(columns={'Probability': 'Susceptibility Score'}), hide_index=True)

# Data insights
st.markdown("---")
st.markdown("### 📊 Data Insights")

st.markdown("#### Occupation Risk Distribution")
col1, col2, col3 = st.columns(3)

low_jobs = df[df['risk_category'] == 'Low Risk'][['Occupation']].sort_values('Occupation').reset_index(drop=True)
medium_jobs = df[df['risk_category'] == 'Medium Risk'][['Occupation']].sort_values('Occupation').reset_index(drop=True)
high_jobs = df[df['risk_category'] == 'High Risk'][['Occupation']].sort_values('Occupation').reset_index(drop=True)

with col1:
    st.success(f"Low Risk — {len(low_jobs)} jobs")
    with st.expander("See all Low Risk jobs"):
        st.dataframe(low_jobs, height=300, hide_index=True)

with col2:
    st.warning(f"Medium Risk — {len(medium_jobs)} jobs")
    with st.expander("See all Medium Risk jobs"):
        st.dataframe(medium_jobs, height=300, hide_index=True)

with col3:
    st.error(f"High Risk — {len(high_jobs)} jobs")
    with st.expander("See all High Risk jobs"):
        st.dataframe(high_jobs, height=300, hide_index=True)

# Chart: Risk by job category
category_risk = df.groupby('category_name')['Probability'].mean().reset_index()
category_risk.columns = ['Category', 'Average Automation Susceptibility']
category_risk = category_risk.sort_values('Average Automation Susceptibility')
fig2 = px.bar(
    category_risk, x='Average Automation Susceptibility', y='Category',
    orientation='h',
    title='Average Automation Susceptibility by Job Category',
    color='Average Automation Susceptibility',
    color_continuous_scale='RdYlGn_r',
    height=560
)
fig2.update_layout(
    font=dict(size=13),
    yaxis=dict(tickfont=dict(size=12)),
    xaxis=dict(tickfont=dict(size=12))
)
st.plotly_chart(fig2, width='stretch')

# Chart: Automation risk vs total employment scatter
fig3 = px.scatter(
    df, x='Probability', y='total_employment',
    color='risk_category',
    hover_name='Occupation',
    title='Automation Susceptibility vs Total Employment',
    color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#e67e22', 'High Risk': '#e74c3c'},
    labels={'Probability': 'Automation Susceptibility', 'total_employment': 'Total Employment'},
    height=450
)
fig3.update_layout(
    font=dict(size=13),
    legend=dict(font=dict(size=12))
)
st.plotly_chart(fig3, width='stretch')

# Career comparison
st.markdown("---")
st.markdown("### ⚖️ Career Comparison Mode")
st.markdown("Compare 2 occupations side-by-side to see which path has lower automation susceptibility.")

col1, col2 = st.columns(2)
with col1:
    job_a = st.selectbox("Choose first career:", [''] + sorted(df['Occupation'].tolist()), key='compare_a')
with col2:
    job_b = st.selectbox("Choose second career:", [''] + sorted(df['Occupation'].tolist()), key='compare_b')

if job_a and job_b and job_a != job_b:
    data_a = df[df['Occupation'] == job_a].iloc[0]
    data_b = df[df['Occupation'] == job_b].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {job_a}")
        prob_a = data_a['Probability']
        if prob_a > 0.66:
            st.error(f"🔴 High — {prob_a:.1%}")
        elif prob_a > 0.33:
            st.warning(f"🟠 Medium — {prob_a:.1%}")
        else:
            st.success(f"🟢 Low — {prob_a:.1%}")
        st.progress(float(prob_a))
        st.caption(f"Category: {data_a.get('category_name', 'N/A')}")

    with col2:
        st.markdown(f"#### {job_b}")
        prob_b = data_b['Probability']
        if prob_b > 0.66:
            st.error(f"🔴 High — {prob_b:.1%}")
        elif prob_b > 0.33:
            st.warning(f"🟠 Medium — {prob_b:.1%}")
        else:
            st.success(f"🟢 Low — {prob_b:.1%}")
        st.progress(float(prob_b))
        st.caption(f"Category: {data_b.get('category_name', 'N/A')}")

    st.markdown("---")
    if prob_a < prob_b:
        st.success(f"✅ **{job_a}** has lower automation susceptibility ({prob_a:.1%} vs {prob_b:.1%})")
    elif prob_b < prob_a:
        st.success(f"✅ **{job_b}** has lower automation susceptibility ({prob_b:.1%} vs {prob_a:.1%})")
    else:
        st.info("Both occupations have equal automation susceptibility.")

elif job_a and job_b and job_a == job_b:
    st.warning("Please select two different occupations to compare.")

# Skill recommendations
st.markdown("---")
st.markdown("### 🛡️ Skills That Reduce Automation Risk")
st.markdown("These skill areas are commonly identified in workforce research as harder to automate or valuable for adapting to AI-driven change.")

skill_data = {
    "Critical Thinking & Problem Solving": {
        "desc": "Analyzing information, evaluating options, and making complex decisions.",
        "examples": "Engineering, Research, Management, Law",
        "risk_reduction": "High"
    },
    "Social & Emotional Intelligence": {
        "desc": "Empathy, communication, negotiation, and human interaction.",
        "examples": "Healthcare, Counseling, Teaching, Social Work",
        "risk_reduction": "High"
    },
    "Creativity & Original Thinking": {
        "desc": "Generating novel ideas, artistic expression, and innovative design.",
        "examples": "Arts & Media, Marketing, Architecture, R&D",
        "risk_reduction": "High"
    },
    "Technical & Digital Literacy": {
        "desc": "Programming, data analysis, systems thinking, and working with AI tools.",
        "examples": "Software, Data Science, Electrical Engineering, IT",
        "risk_reduction": "High"
    },
    "Adaptability & Lifelong Learning": {
        "desc": "Willingness to continuously learn new tools, methods, and domains.",
        "examples": "All fields — cross-cutting skill",
        "risk_reduction": "Medium–High"
    },
    "Physical Dexterity in Complex Environments": {
        "desc": "Fine motor skills in unpredictable, dynamic settings.",
        "examples": "Surgery, Skilled Trades, Emergency Response",
        "risk_reduction": "Medium"
    },
}

cols = st.columns(2)
for i, (skill, info) in enumerate(skill_data.items()):
    with cols[i % 2]:
        with st.expander(f"{'🟢' if info['risk_reduction'] == 'High' else '🟡'} {skill}"):
            st.markdown(f"**Why it matters:** {info['desc']}")
            st.markdown(f"**Example careers:** {info['examples']}")
            st.markdown(f"**Risk reduction:** `{info['risk_reduction']}`")

# About section
st.markdown("---")
st.markdown("### About This Tool")
st.markdown("""
This tool was built to help students and early career explorers understand how AI and automation 
may affect different occupations, based on a landmark academic study of US job tasks.

**How it works:**
- Search for any occupation to see its estimated automation susceptibility score
- Scores are drawn from the **Frey & Osborne (2017)** study, which analyzed 702 US occupations using O*NET task data
- Occupations are classified into Low, Medium, and High susceptibility tiers using fixed thresholds
- Career alternatives and skill recommendations are generated by filtering same-category occupations

**Note on the machine learning model:**
A Random Forest classifier was trained as an exploratory component to classify occupations by 
risk tier using SOC group and employment features. The public-facing susceptibility score 
displayed in this app comes from the original Frey & Osborne probability estimates, not the 
model prediction. The model achieved 72% accuracy on tier classification and is included for 
reproducibility.

**Who this is for:**
- Students choosing a college major or career path
- Anyone curious about how AI is reshaping the workforce

**A note on job coverage:**
This tool covers 702 broad occupational categories based on official US job classifications. 
Highly specialized roles may not appear individually and should be searched under their broader 
category.

**Data Sources:**
- Frey, C.B. & Osborne, M.A. (2017). The Future of Employment. *Technological Forecasting and Social Change*, 114, 254–280.
- BLS Occupational Employment and Wage Statistics (OEWS), 2023 data release.
- O*NET Online Database, accessed 2026. National Center for O*NET Development.
""")

st.info("""
⚠️ **Limitations:** Scores reflect a historical task-based automation study and should not be 
interpreted as the probability that a job will disappear. Occupations evolve over time, and AI 
is more likely to change individual tasks within a role than to replace entire occupations. 
Use these estimates as a starting point for career exploration, not as a definitive forecast.
""")

st.markdown("---")
st.caption("Data sourced from Frey & Osborne (2017) Automation Study and BLS Occupational Data")
