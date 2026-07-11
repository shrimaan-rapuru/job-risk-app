import plotly.express as px
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
from thefuzz import process

# Page config
st.set_page_config(
    page_title="AI Job Risk Predictor",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .big-title { font-size: 2.5rem; font-weight: 800; text-align: center; 
                 background: linear-gradient(90deg, #00c6ff, #0072ff);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { text-align: center; color: #888; font-size: 1.1rem; margin-bottom: 2rem; }
    .fact-box { background: #1e2130; border-left: 4px solid #0072ff; 
                padding: 12px 16px; border-radius: 8px; margin: 1rem 0; }
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
    df = pd.read_csv('automation_data_by_state.csv', encoding='latin-1')
    state_columns = df.columns[3:54].tolist()
    df[state_columns] = df[state_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    df['soc_group'] = df['SOC'].str[:2]
    df['total_employment'] = df[state_columns].sum(axis=1)
    df['risk_category'] = df['Probability'].apply(
        lambda x: 'Low Risk' if x <= 0.33 else 'Medium Risk' if x <= 0.66 else 'High Risk'
    )
    df['category_name'] = df['SOC'].str[:2].map({
        '11': 'Management', '13': 'Business & Finance', '15': 'Computer & Math',
        '17': 'Architecture & Engineering', '19': 'Life & Physical Science',
        '21': 'Community & Social Service', '23': 'Legal', '25': 'Education',
        '27': 'Arts & Media', '29': 'Healthcare Practitioners', '31': 'Healthcare Support',
        '33': 'Protective Service', '35': 'Food Preparation', '37': 'Building & Grounds',
        '39': 'Personal Care', '41': 'Sales', '43': 'Office & Admin',
        '45': 'Farming & Fishing', '47': 'Construction', '49': 'Installation & Repair',
        '51': 'Production/Manufacturing', '53': 'Transportation'
    })
    return df

model, le = load_model()
df = load_data()

soc_labels = {
    '11': 'Management', '13': 'Business & Finance', '15': 'Computer & Math',
    '17': 'Architecture & Engineering', '19': 'Life & Physical Science',
    '21': 'Community & Social Service', '23': 'Legal', '25': 'Education',
    '27': 'Arts & Media', '29': 'Healthcare Practitioners', '31': 'Healthcare Support',
    '33': 'Protective Service', '35': 'Food Preparation', '37': 'Building & Grounds',
    '39': 'Personal Care', '41': 'Sales', '43': 'Office & Admin',
    '45': 'Farming & Fishing', '47': 'Construction', '49': 'Installation & Repair',
    '51': 'Production/Manufacturing', '53': 'Transportation'
}

facts = [
    "🤖 By 2030, up to 30% of current jobs could be automated according to McKinsey.",
    "💡 Jobs requiring creativity and social skills are the hardest to automate.",
    "📊 Higher wage jobs tend to be significantly less at risk of automation.",
    "🔧 Routine manual tasks are among the easiest for AI to replicate.",
    "🌍 The World Economic Forum estimates AI will create 97 million new jobs by 2025.",
    "🎓 Jobs requiring a bachelor's degree or higher are generally safer from automation.",
    "⚕️ Healthcare jobs are among the most resistant to automation due to human interaction.",
    "💻 Software developers are at low risk — they're the ones building the AI!",
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

# Job search
st.markdown("### 🔍 Search for a Job")
all_jobs = [''] + sorted(df['Occupation'].tolist())
selected_job = st.selectbox("Type or select a job title:", all_jobs)

if selected_job:
    job_data = df[df['Occupation'] == selected_job].iloc[0]
    actual_prob = job_data['Probability']
    soc_grp = job_data['soc_group']
    category_name = soc_labels.get(soc_grp, 'Related')

    st.markdown("---")
    st.subheader(f"Results for: **{selected_job}**")

    # Risk label
    if actual_prob > 0.66:
        st.error("🔴 High Risk of Automation")
    elif actual_prob > 0.33:
        st.warning("🟠 Medium Risk of Automation")
    else:
        st.success("🟢 Low Risk of Automation")

    # Risk meter
    st.markdown("### 📊 Automation Risk Meter")
    st.progress(float(actual_prob))
    st.caption(f"Risk Score: {actual_prob:.1%} — {'🔴 High Risk' if actual_prob > 0.66 else '🟠 Medium Risk' if actual_prob > 0.33 else '🟢 Low Risk'}")

    # Score and explanation
    st.markdown("---")
    st.metric("Automation Probability Score", f"{actual_prob:.1%}")
    st.caption("📊 From the Oxford/Frey & Osborne research study analyzing specific job tasks.")

    # Advice
    if actual_prob <= 0.33:
        st.success("✅ This job has a strong long-term outlook. Focus on skills that complement AI.")
    elif actual_prob <= 0.66:
        st.warning("⚠️ This job has moderate risk. Consider developing skills harder to automate.")
    else:
        st.error("🚨 This job faces significant automation risk. Consider related roles with lower risk.")

    # Career recommendations
    if actual_prob > 0.33:
        st.markdown("---")
        st.markdown("### 💡 Safer Career Alternatives in the Same Field")
        safer = df[
            (df['soc_group'] == soc_grp) &
            (df['Probability'] < actual_prob) &
            (df['Occupation'] != selected_job)
        ].nsmallest(3, 'Probability')[['Occupation', 'Probability']]

        if safer.empty:
            st.info("No safer alternatives found in this category.")
        else:
            for _, row in safer.iterrows():
                st.markdown(f"🟢 **{row['Occupation']}** — `{row['Probability']:.1%} risk`")

    # Related jobs
    st.markdown("---")
    st.markdown(f"### 🏢 How does this compare to other **{category_name}** jobs?")
    related = df[df['soc_group'] == soc_grp][['Occupation', 'Probability']].sort_values('Probability')
    related['Risk'] = related['Probability'].apply(
        lambda x: '🟢 Low' if x <= 0.33 else '🟠 Medium' if x <= 0.66 else '🔴 High'
    )
    st.dataframe(related.rename(columns={'Probability': 'Automation Score'}), hide_index=True)

# Data insights
st.markdown("---")
st.markdown("### 📊 Data Insights")

st.markdown("#### Job Risk Distribution")
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
category_risk.columns = ['Category', 'Average Automation Risk']
category_risk = category_risk.sort_values('Average Automation Risk')
fig2 = px.bar(
    category_risk, x='Average Automation Risk', y='Category',
    orientation='h',
    title='Average Automation Risk by Job Category',
    color='Average Automation Risk',
    color_continuous_scale='RdYlGn_r'
)
st.plotly_chart(fig2, use_container_width=True)

# Chart: Automation risk vs total employment scatter
fig3 = px.scatter(
    df, x='Probability', y='total_employment',
    color='risk_category',
    hover_name='Occupation',
    title='Automation Risk vs Total Employment',
    color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#e67e22', 'High Risk': '#e74c3c'},
    labels={'Probability': 'Automation Probability', 'total_employment': 'Total Employment'}
)
st.plotly_chart(fig3, use_container_width=True)

# About section
st.markdown("---")
st.markdown("### About This Tool")
st.markdown("""
This tool was built as a class project to help high school seniors and early college students 
understand how artificial intelligence and automation may impact their future careers.

**How it works:**
- Search for any job title to see its predicted automation risk
- The risk score is based on the Oxford/Frey & Osborne study which analyzed 702 US occupations
- Wage and education data is sourced from the Bureau of Labor Statistics (2023)
- The machine learning model was trained using Random Forest with 72% accuracy

**Who this is for:**
- Students choosing a college major or career path
- Anyone curious about how AI is reshaping the workforce

**A note on job coverage:**
This tool covers 702 broad occupational categories based on official US job classifications. 
Highly specialized roles may not appear individually and should be searched under their broader 
category. For example, a Nurse Anesthetist would fall under Registered Nurses.

**Data Sources:**
- Oxford/Frey & Osborne Automation Study
- BLS Occupational Employment and Wage Statistics
- O*NET Education and Training Database
""")

st.markdown("---")
st.caption("Data sourced from Oxford/Frey & Osborne Automation Study and BLS Occupational Data")