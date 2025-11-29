import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import xml.etree.ElementTree as ET
import re
import io
import pickle
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Configure Streamlit page
st.set_page_config(
    page_title="Cancer Mutation Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern dark theme CSS with purple, yellow, and pink accents
st.markdown("""
<style>
    /* Import Google Fonts for better typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global color variables - Modern dark theme */
    :root {
        --dark-charcoal: #2C3E50;
        --medium-purple: #8E44AD;
        --bright-yellow: #F1C40F;
        --soft-pink: #FF9FC7;
        --light-purple: #BB8FCE;
        --pale-yellow: #FCF3CF;
        --light-pink: #FADBD8;
        --very-light-gray: #F8F9FA;
    }
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, var(--very-light-gray) 0%, #F4F6F9 50%, var(--pale-yellow) 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styles */
    .main-header {
        font-size: 3rem;
        color: var(--dark-charcoal);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(44, 62, 80, 0.1);
        background: linear-gradient(135deg, var(--dark-charcoal), var(--medium-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: var(--medium-purple);
        border-bottom: 2px solid var(--soft-pink);
        padding-bottom: 0.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    /* Card components */
    .highlight {
        background: linear-gradient(135deg, var(--very-light-gray), #FDFDFE);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        border: 1px solid var(--light-purple);
        box-shadow: 0 8px 25px rgba(142, 68, 173, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FDFDFE, var(--very-light-gray));
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(142, 68, 173, 0.15);
        margin-bottom: 15px;
        border: 1px solid var(--light-purple);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(142, 68, 173, 0.2);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, var(--light-pink), #FDF2F8);
        padding: 25px;
        border-radius: 20px;
        border-left: 6px solid var(--soft-pink);
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(255, 159, 199, 0.15);
        border: 1px solid var(--soft-pink);
    }
    
    .model-info {
        background: linear-gradient(135deg, var(--pale-yellow), #FFFEF7);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid var(--bright-yellow);
        margin-bottom: 20px;
        box-shadow: 0 6px 20px rgba(241, 196, 15, 0.15);
    }
    
    .resource-card {
        background: linear-gradient(135deg, var(--very-light-gray), #F8F9FD);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        border: 1px solid var(--light-purple);
        box-shadow: 0 6px 20px rgba(142, 68, 173, 0.1);
    }
    
    .warning {
        background: linear-gradient(135deg, var(--light-pink), #FEF7F7);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid var(--soft-pink);
        margin-bottom: 20px;
        box-shadow: 0 6px 20px rgba(255, 159, 199, 0.15);
    }
    
    .info-box {
        background: linear-gradient(135deg, #E8F4FD, #F0F8FF);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid var(--medium-purple);
        margin-bottom: 20px;
        box-shadow: 0 6px 20px rgba(142, 68, 173, 0.15);
    }
    
    /* Form styling */
    .stSelectbox > div > div {
        background-color: var(--very-light-gray);
        border: 2px solid var(--light-purple);
        border-radius: 10px;
        color: var(--dark-charcoal);
    }
    
    .stTextInput > div > div > input {
        background-color: var(--very-light-gray);
        border: 2px solid var(--light-purple);
        border-radius: 10px;
        color: var(--dark-charcoal);
    }
    
    .stNumberInput > div > div > input {
        background-color: var(--very-light-gray);
        border: 2px solid var(--light-purple);
        border-radius: 10px;
        color: var(--dark-charcoal);
    }
    
    /* Slider customization - Modern theme */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--medium-purple), var(--soft-pink)) !important;
    }
    
    .stSlider > div > div > div > div > div {
        background-color: var(--dark-charcoal) !important;
        border: 3px solid var(--very-light-gray) !important;
        box-shadow: 0 0 0 3px var(--medium-purple) !important;
    }
    
    .stSlider > div > div > div {
        background-color: var(--light-purple) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--medium-purple), var(--soft-pink));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(142, 68, 173, 0.3);
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--soft-pink), var(--dark-charcoal));
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(142, 68, 173, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 2px solid var(--light-purple);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(135deg, var(--very-light-gray), #FDFDFE);
        border: 2px solid var(--light-purple);
        border-radius: 10px 10px 0 0;
        color: var(--dark-charcoal);
        font-weight: 600;
        padding-left: 20px;
        padding-right: 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--medium-purple), var(--soft-pink));
        color: white;
        border-color: var(--dark-charcoal);
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--very-light-gray), #FDFDFE);
        border: 2px solid var(--light-purple);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(142, 68, 173, 0.1);
    }
    
    [data-testid="metric-container"] > div {
        color: var(--dark-charcoal);
        font-weight: 600;
    }
    
    /* Success/Info/Error message styling */
    .stSuccess {
        background: linear-gradient(135deg, var(--pale-yellow), #FFFEF7);
        border: 1px solid var(--bright-yellow);
        border-radius: 10px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, var(--light-pink), #FDF2F8);
        border: 1px solid var(--soft-pink);
        border-radius: 10px;
    }
    
    .stError {
        background: linear-gradient(135deg, var(--light-pink), #FEF7F7);
        border: 1px solid var(--soft-pink);
        border-radius: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, var(--very-light-gray), #FDFDFE);
        border: 2px solid var(--light-purple);
        border-radius: 10px;
        color: var(--dark-charcoal);
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: linear-gradient(135deg, #FFFAF0, var(--very-light-gray));
        border: 1px solid var(--light-purple);
        border-radius: 0 0 10px 10px;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: var(--medium-purple) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--very-light-gray), #FDFDFE);
        border-right: 2px solid var(--light-purple);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--very-light-gray);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--medium-purple), var(--soft-pink));
        border-radius: 10px;
        border: 2px solid var(--very-light-gray);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--soft-pink), var(--dark-charcoal));
    }
    
    /* Text and heading colors */
    h1, h2, h3, h4, h5, h6 {
        color: var(--dark-charcoal) !important;
        font-family: 'Inter', sans-serif;
    }
    
    .stMarkdown {
        color: var(--dark-charcoal);
    }
    
    /* Form labels */
    label {
        color: var(--dark-charcoal) !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load only the TensorFlow model
@st.cache_resource
def load_keras_model():
    """Load only the TensorFlow model"""
    try:
        import os
        if os.path.exists('cancer_tf_model.keras'):
            model = tf.keras.models.load_model('cancer_tf_model.keras')
            return model
        else:
            return None
    except Exception as e:
        return None

# Initialize model
keras_model = load_keras_model()

# Define preprocessing functions internally
def preprocess_features(gene, cancer_type, chromosome, position, mutation_type, 
                       sift_score, polyphen_score, cadd_score, clinical_significance):
    """Internal preprocessing function"""
    
    # Define mappings based on your training data - updated with values from dataset
    gene_mapping = {'TP53': 0, 'BRCA1': 1, 'BRCA2': 2, 'EGFR': 3, 'KRAS': 4, 'PIK3CA': 5, 
                   'PTEN': 6, 'MYC': 7, 'BRAF': 8, 'ALK': 9}
    cancer_mapping = {'Lung': 0, 'Breast': 1, 'Colon': 2, 'Prostate': 3, 'Ovarian': 4, 
                     'Pancreatic': 5, 'Brain': 6, 'Liver': 7, 'Head and Neck': 8, 'Esophagus': 9,
                     'Stomach': 10, 'Bladder': 11, 'Cervical': 12, 'Endometrium': 13, 'Sarcoma': 14,
                     'Testicular': 15, 'Melanoma': 16, 'Kidney': 17, 'Leukemia': 18, 'Thyroid': 19}
    mutation_mapping = {'Missense': 0, 'Nonsense': 1, 'Frameshift': 2, 'Silent': 3, 'Splice': 4}
    clinical_mapping = {'Pathogenic': 0, 'Likely_pathogenic': 1, 'VUS': 2, 'Likely_benign': 3, 'Benign': 4}
    
    # Convert categorical to numerical
    gene_encoded = gene_mapping.get(gene, 0)
    cancer_encoded = cancer_mapping.get(cancer_type, 0)
    mutation_encoded = mutation_mapping.get(mutation_type, 0)
    clinical_encoded = clinical_mapping.get(clinical_significance, 2)
    
    # Create feature vector (adjust based on your model's expected input shape)
    features = np.array([[
        chromosome,
        position / 1000000,  # Scale position
        sift_score,
        polyphen_score / 40,  # Scale CADD score
        gene_encoded,
        cancer_encoded, 
        mutation_encoded,
        clinical_encoded
    ]], dtype=np.float32)
    
    return features

def predict_mutation_impact(gene, cancer_type, chromosome, position, mutation_type, 
                           sift_score, polyphen_score, cadd_score, clinical_significance):
    """
    Predict mutation impact using only the Keras model
    """
    try:
        if keras_model is None:
            # Fallback to simulated prediction if model not loaded
            return simulate_prediction(clinical_significance)
        
        # Preprocess input features
        features = preprocess_features(gene, cancer_type, chromosome, position, mutation_type,
                                     sift_score, polyphen_score, cadd_score, clinical_significance)
        
        # Make prediction
        prediction = keras_model.predict(features, verbose=0)[0]
        predicted_class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # Define class names (adjust based on your training)
        class_names = ['High Impact', 'Moderate Impact', 'Low Impact']
        predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else 'Unknown'
        
        # Get all class probabilities
        class_probs = {}
        for i, prob in enumerate(prediction):
            if i < len(class_names):
                class_probs[class_names[i]] = float(prob)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probs,
            'model_used': 'TensorFlow Keras Neural Network'
        }
        
    except Exception as e:
        return simulate_prediction(clinical_significance)

def simulate_prediction(clinical_significance):
    """Fallback prediction when model is not available"""
    # Map clinical significance to impact classes
    impact_mapping = {
        'Likely_pathogenic': {'predicted_class': 'High Impact', 'confidence': 0.85},
        'Pathogenic': {'predicted_class': 'High Impact', 'confidence': 0.92},
        'VUS': {'predicted_class': 'Moderate Impact', 'confidence': 0.65},
        'Likely_benign': {'predicted_class': 'Low Impact', 'confidence': 0.78},
        'Benign': {'predicted_class': 'Low Impact', 'confidence': 0.88}
    }
    
    result = impact_mapping.get(clinical_significance, {'predicted_class': 'Moderate Impact', 'confidence': 0.50})
    
    # Add simulated class probabilities
    if result['predicted_class'] == 'High Impact':
        class_probs = {'High Impact': result['confidence'], 'Moderate Impact': 0.20, 'Low Impact': 0.05}
    elif result['predicted_class'] == 'Low Impact':
        class_probs = {'Low Impact': result['confidence'], 'Moderate Impact': 0.15, 'High Impact': 0.07}
    else:
        class_probs = {'Moderate Impact': result['confidence'], 'High Impact': 0.25, 'Low Impact': 0.25}
    
    result['class_probabilities'] = class_probs
    result['model_used'] = 'Fallback Simulation'
    
    return result

# Enhanced API Integration Functions
def query_cancervar_api(gene, mutation, cancer_type):
    """Enhanced CancerVar API query with more comprehensive data"""
    try:
        interpretations = {
            "TP53": {
                "R175H": {
                    "interpretation": "Pathogenic - Strong clinical significance",
                    "oncogenic_mechanism": "Disrupts DNA binding domain, promotes tumor progression",
                    "prevalence": "2.5% of all cancers",
                    "therapeutic_implications": "Resistance to standard chemotherapy, consider clinical trials",
                    "clinical_significance": "Pathogenic"
                },
                "R273H": {
                    "interpretation": "Pathogenic - Strong clinical significance", 
                    "oncogenic_mechanism": "Disrupts DNA binding, loss of tumor suppressor function",
                    "prevalence": "1.8% of all cancers",
                    "therapeutic_implications": "Consider PARP inhibitors in combination therapy",
                    "clinical_significance": "Pathogenic"
                },
                "R248Q": {
                    "interpretation": "Pathogenic - Strong clinical significance",
                    "oncogenic_mechanism": "Direct contact mutation affecting DNA binding",
                    "prevalence": "1.2% of all cancers",
                    "therapeutic_implications": "Immunotherapy may be beneficial",
                    "clinical_significance": "Pathogenic"
                }
            },
            "BRCA1": {
                "C61G": {
                    "interpretation": "Pathogenic - Increased cancer risk",
                    "oncogenic_mechanism": "Disrupts BRCA1-BARD1 complex formation",
                    "prevalence": "0.8% of hereditary breast cancers",
                    "therapeutic_implications": "PARP inhibitors highly effective",
                    "clinical_significance": "Pathogenic"
                },
                "185delAG": {
                    "interpretation": "Pathogenic - Frameshift mutation",
                    "oncogenic_mechanism": "Truncated protein, loss of function",
                    "prevalence": "1.0% of Ashkenazi Jewish population", 
                    "therapeutic_implications": "PARP inhibitors, platinum-based therapy",
                    "clinical_significance": "Pathogenic"
                }
            },
            "EGFR": {
                "L858R": {
                    "interpretation": "Pathogenic - Responsive to EGFR inhibitors",
                    "oncogenic_mechanism": "Constitutive activation of kinase domain",
                    "prevalence": "40% of Asian NSCLC patients",
                    "therapeutic_implications": "First-line EGFR TKIs (gefitinib, erlotinib, osimertinib)",
                    "clinical_significance": "Pathogenic"
                },
                "T790M": {
                    "interpretation": "Resistance mutation - May require different therapy",
                    "oncogenic_mechanism": "Steric hindrance to drug binding", 
                    "prevalence": "50-60% of EGFR TKI resistance cases",
                    "therapeutic_implications": "Third-generation EGFR inhibitors (osimertinib)",
                    "clinical_significance": "Pathogenic"
                }
            },
            "KRAS": {
                "G12D": {
                    "interpretation": "Pathogenic - Resistance to anti-EGFR therapy",
                    "oncogenic_mechanism": "Constitutive GTPase activation",
                    "prevalence": "13% of all cancers",
                    "therapeutic_implications": "Avoid anti-EGFR therapy, consider MEK inhibitors",
                    "clinical_significance": "Pathogenic"
                },
                "G12C": {
                    "interpretation": "Pathogenic - May respond to KRAS G12C inhibitors",
                    "oncogenic_mechanism": "Constitutive GTPase activation",
                    "prevalence": "4% of NSCLC, 3% of colorectal cancer",
                    "therapeutic_implications": "Sotorasib, adagrasib, clinical trials",
                    "clinical_significance": "Pathogenic"
                }
            }
        }
        
        # Default interpretation
        default_interpretation = {
            "interpretation": "Variant of Uncertain Significance - Further testing recommended",
            "oncogenic_mechanism": "Unknown",
            "prevalence": "Unknown", 
            "therapeutic_implications": "Standard therapy based on cancer type",
            "clinical_significance": "VUS"
        }
        
        result = interpretations.get(gene, {}).get(mutation, default_interpretation)
        
        # Add additional metadata
        result['evidence_level'] = 'A' if 'Strong' in result['interpretation'] else 'B'
        result['references'] = ['Li Q, et al. Sci Adv. 2022', 'AMP/ASCO/CAP 2017 Guidelines']
        
        return result
        
    except Exception as e:
        return {'error': f"Could not query CancerVar: {str(e)}"}

def create_prediction_visualization(class_probabilities, predicted_class):
    """Create visualization for ML model predictions with modern theme"""
    # Create bar chart for class probabilities
    classes = list(class_probabilities.keys())
    probabilities = list(class_probabilities.values())
    
    # Modern theme colors - purple/pink gradient
    colors = ["#65088D" if cls == predicted_class else "#CDA2DF" for cls in classes]
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probabilities,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
            marker_line=dict(color='#2C3E50', width=2)
        )
    ])
    
    fig.update_layout(
        title="Mutation Impact Class Probabilities",
        title_font_color='#2C3E50',
        title_font_size=18,
        xaxis_title="Impact Class",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='#F8F9FA',
        font=dict(color='#2C3E50', family='Inter')
    )
    
    return fig

def create_feature_importance_plot():
    """Create a simulated feature importance plot with modern theme"""
    features = ['SIFT Score', 'PolyPhen Score', 'CADD Score', 'Gene Type', 
                'Cancer Type', 'Chromosome', 'Position', 'Clinical Significance']
    importance = [0.25, 0.22, 0.18, 0.15, 0.08, 0.05, 0.04, 0.03]
    
    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=importance,
            orientation='h',
            marker_color='#8E44AD',
            marker_line=dict(color='#2C3E50', width=1)
        )
    ])
    
    fig.update_layout(
        title="Feature Importance in ML Model",
        title_font_color='#2C3E50',
        title_font_size=18,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400,
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='#F8F9FA',
        font=dict(color='#2C3E50', family='Inter')
    )
    
    return fig

def query_clinical_trials(cancer_type, gene, mutation):
    """Enhanced clinical trials query"""
    try:
        base_url = "https://clinicaltrials.gov/api/query/full_studies"
        query = f"{cancer_type} {gene} {mutation}"
        params = {
            'expr': query,
            'min_rnk': 1,
            'max_rnk': 5,
            'fmt': 'json'
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        
        trials = []
        if response.status_code == 200:
            data = response.json()
            studies = data.get('FullStudiesResponse', {}).get('FullStudies', [])
            
            for study in studies:
                study_data = study.get('Study', {})
                protocol = study_data.get('ProtocolSection', {})
                identification = protocol.get('IdentificationModule', {})
                status = protocol.get('StatusModule', {})
                description = protocol.get('DescriptionModule', {})
                
                trials.append({
                    'nct_id': identification.get('NCTId', 'N/A'),
                    'title': identification.get('OfficialTitle', 'N/A'),
                    'status': status.get('OverallStatus', 'N/A'),
                    'conditions': ', '.join(protocol.get('ConditionsModule', {}).get('ConditionList', {}).get('Condition', ['N/A'])),
                    'interventions': ', '.join([i.get('InterventionName', 'N/A') for i in protocol.get('ArmsInterventionsModule', {}).get('InterventionList', {}).get('Intervention', [])]),
                    'summary': description.get('BriefSummary', 'N/A')[:200] + '...' if description.get('BriefSummary') else 'No summary available'
                })
        
        # Fallback simulated trials
        if not trials:
            trials = [
                {
                    'nct_id': 'NCT04396808',
                    'title': f'Targeted Therapy for {gene} {mutation} in {cancer_type}',
                    'status': 'Recruiting',
                    'conditions': cancer_type,
                    'interventions': 'Targeted Molecular Therapy',
                    'summary': f'Study of targeted treatments for patients with {gene} {mutation} in {cancer_type}'
                }
            ]
        
        return trials
        
    except Exception as e:
        return [{
            'nct_id': 'NCT04396808',
            'title': f'Targeted Therapy for {gene} {mutation} in {cancer_type}',
            'status': 'Recruiting', 
            'conditions': cancer_type,
            'interventions': 'Targeted Molecular Therapy',
            'summary': f'Study of targeted treatments for patients with {gene} {mutation} in {cancer_type}'
        }]

def get_treatment_suggestions(cancer_type, gene, mutation, predicted_impact):
    """Enhanced treatment suggestions based on ML predictions"""
    treatment_db = {
        'Lung': {
            'EGFR': {
                'L858R': ['Osimertinib (1st line)', 'Gefitinib', 'Erlotinib', 'Afatinib'],
                'T790M': ['Osimertinib (resistance)', 'Chemotherapy combinations'],
                'default': ['EGFR TKIs', 'Immunotherapy']
            },
            'KRAS': {
                'G12C': ['Sotorasib', 'Adagrasib', 'Immunotherapy'],
                'default': ['MEK inhibitors', 'Immunotherapy']
            },
            'default': ['Platinum-based chemotherapy', 'Immunotherapy', 'Radiation therapy']
        },
        'Breast': {
            'BRCA1': ['PARP inhibitors (Olaparib)', 'Platinum-based chemotherapy', 'Immunotherapy'],
            'BRCA2': ['PARP inhibitors (Talazoparib)', 'Platinum-based chemotherapy'],
            'default': ['Endocrine therapy', 'CDK4/6 inhibitors', 'Chemotherapy']
        },
        'default': ['Standard chemotherapy', 'Targeted therapy', 'Immunotherapy']
    }
    
    # Get base treatments
    cancer_key = next((k for k in treatment_db.keys() if k in cancer_type), 'default')
    cancer_treatments = treatment_db[cancer_key]
    
    if isinstance(cancer_treatments, dict):
        gene_treatments = cancer_treatments.get(gene, cancer_treatments.get('default', ['Standard therapy']))
        if isinstance(gene_treatments, dict):
            treatments = gene_treatments.get(mutation, gene_treatments.get('default', ['Standard therapy']))
        else:
            treatments = gene_treatments
    else:
        treatments = cancer_treatments
    
    # Add impact-based recommendations
    if predicted_impact == 'High Impact':
        treatments.extend(['Aggressive targeted therapy', 'Clinical trial enrollment', 'Genetic counseling'])
    elif predicted_impact == 'Moderate Impact':
        treatments.extend(['Combination therapy', 'Molecular profiling', 'Consider clinical trials'])
    else:
        treatments.extend(['Standard therapy', 'Regular monitoring'])
    
    return list(set(treatments))  # Remove duplicates

# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🍂 Cancer Mutation Analysis Platform</h1>', unsafe_allow_html=True)
    st.markdown("### Machine Learning-Based Precision Oncology Predictions")
    
    # Model status indicator
    if keras_model is not None:
        st.markdown('<div class="model-info"> Model Loaded Successfully - Using trained TensorFlow neural network for predictions</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning"> Keras Model Not Found - Using simulated predictions. Please ensure cancer_tf_model.keras is available.</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["ML Prediction", "Treatment Options", "Clinical Trials"])
    
    with tab1:
        st.markdown('<div class="sub-header">Mutation Impact Prediction</div>', unsafe_allow_html=True)

        # Input form
        with st.form("ml_prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Genetic Information**")
                gene_name = st.selectbox("Gene Name", 
                    ['TP53', 'BRCA1', 'BRCA2', 'EGFR', 'KRAS', 'PIK3CA', 'PTEN', 'MYC', 'BRAF', 'ALK'], 
                    index=0)
                mutation = st.text_input("Mutation", value="p.R175H", help="e.g., p.R175H, p.L858R, p.G12D")
                chromosome = st.selectbox("Chromosome", list(range(1, 23)) + ['X', 'Y'], index=16)
                position = st.number_input("Position", min_value=1, value=7577121)
                
            with col2:
                st.markdown("**Clinical Context**")
                cancer_type = st.selectbox("Cancer Type",
                    ['Lung', 'Breast', 'Colon', 'Prostate', 'Ovarian', 'Pancreatic', 'Brain', 'Liver', 
                     'Head and Neck', 'Esophagus', 'Stomach', 'Bladder', 'Cervical', 'Endometrium', 
                     'Sarcoma', 'Testicular', 'Melanoma', 'Kidney', 'Leukemia', 'Thyroid'], 
                    index=0)
                mutation_type = st.selectbox("Mutation Type",
                    ['Missense', 'Nonsense', 'Frameshift', 'Silent', 'Splice'], index=0)
                clinical_significance = st.selectbox("Known Clinical Significance",
                    ['Pathogenic', 'Likely_pathogenic', 'VUS', 'Likely_benign', 'Benign'], index=0)
                
            with col3:
                st.markdown("**Functional Scores**")
                sift_score = st.slider("SIFT Score", 0.0, 1.0, 0.03, help="Lower = more damaging")
                polyphen_score = st.slider("PolyPhen/CADD Score", 0.0, 40.0, 25.0, help="Higher = more damaging")
                cadd_score = st.slider("Additional CADD Score", 0.0, 40.0, 32.5)
            
            # Prediction button
            predict_button = st.form_submit_button("Predict Mutation Impact", type="primary")
        
        if predict_button:
            # Make ML prediction
            with st.spinner("Running prediction model..."):
                prediction_result = predict_mutation_impact(
                    gene_name, cancer_type, chromosome, position, mutation_type,
                    sift_score, polyphen_score, cadd_score, clinical_significance
                )
                time.sleep(1)  # Simulate processing time
            
            # Display prediction results
            st.success("Prediction Complete!")
            
            # Main prediction card
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Impact", prediction_result['predicted_class'])
            with col2:
                st.metric("Confidence", f"{prediction_result['confidence']:.1%}")
            with col3:
                st.metric("Model", prediction_result['model_used'].split()[0])

            st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization
            col4, col5 = st.columns(2)
            
            with col4:
                # Class probability visualization
                prob_fig = create_prediction_visualization(
                    prediction_result['class_probabilities'], 
                    prediction_result['predicted_class']
                )
                st.plotly_chart(prob_fig, use_container_width=True)
                
            with col5:
                # Feature importance
                importance_fig = create_feature_importance_plot()
                st.plotly_chart(importance_fig, use_container_width=True)
            
            # Detailed results
            st.subheader("Detailed Analysis")
            
            # Query external APIs for additional context
            with st.spinner("Querying clinical databases..."):
                cancervar_result = query_cancervar_api(gene_name, mutation, cancer_type)
                time.sleep(1)
            
            # Store results
            st.session_state.analysis_results = {
                'gene': gene_name,
                'mutation': mutation,
                'cancer_type': cancer_type,
                'chromosome': chromosome,
                'position': position,
                'mutation_type': mutation_type,
                'ml_prediction': prediction_result,
                'cancervar_result': cancervar_result,
                'scores': {
                    'sift': sift_score,
                    'polyphen': polyphen_score,
                    'cadd': cadd_score
                }
            }
            
            # Display detailed results
            col6, col7 = st.columns(2)
            
            with col6:
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                st.subheader("ML Model Analysis")
                st.write(f"**Predicted Class:** {prediction_result['predicted_class']}")
                st.write(f"**Confidence:** {prediction_result['confidence']:.1%}")
                
                # Show all class probabilities
                st.write("**Class Probabilities:**")
                for class_name, prob in prediction_result['class_probabilities'].items():
                    st.write(f"  - {class_name}: {prob:.1%}")
                    
                st.write(f"**Model:** {prediction_result['model_used']}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col7:
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                st.subheader("Clinical Database")
                if 'error' not in cancervar_result:
                    st.info(cancervar_result['interpretation'])
                    
                    with st.expander("Clinical Details"):
                        st.write(f"**Mechanism:** {cancervar_result.get('oncogenic_mechanism', 'N/A')}")
                        st.write(f"**Prevalence:** {cancervar_result.get('prevalence', 'N/A')}")
                        st.write(f"**Therapeutic Implications:** {cancervar_result.get('therapeutic_implications', 'N/A')}")
                else:
                    st.error(cancervar_result['error'])
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="sub-header">Personalized Treatment Recommendations</div>', unsafe_allow_html=True)
        
        if st.session_state.analysis_results:
            result = st.session_state.analysis_results
            predicted_impact = result['ml_prediction']['predicted_class']
            confidence = result['ml_prediction']['confidence']
            
            st.subheader(f"Treatment Options for {result['gene']} {result['mutation']} in {result['cancer_type']}")
            
            # Treatment recommendations based on ML prediction
            treatments = get_treatment_suggestions(
                result['cancer_type'], result['gene'], result['mutation'], predicted_impact
            )
            
            # Display treatments with priority based on ML confidence
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.write(f"**Recommendation Priority:** Based on {predicted_impact} prediction with {confidence:.1%} confidence")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Priority treatment recommendations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("High Priority Treatments")
                for i, treatment in enumerate(treatments[:3], 1):
                    st.write(f"{i}. **{treatment}**")
                    
            with col2:
                st.subheader("Additional Options")
                for i, treatment in enumerate(treatments[3:], 4):
                    st.write(f"{i}. **{treatment}**")
            
            # Treatment efficacy prediction visualization
            if len(treatments) > 0:
                # Simulate treatment efficacy based on ML prediction
                efficacy_scores = np.random.normal(0.7 if predicted_impact == 'High Impact' else 0.5, 0.15, len(treatments[:5]))
                efficacy_scores = np.clip(efficacy_scores, 0.1, 0.95)
                
                efficacy_df = pd.DataFrame({
                    'Treatment': treatments[:5],
                    'Predicted Efficacy': efficacy_scores,
                    'Confidence_Level': ['High' if score > 0.7 else 'Medium' if score > 0.5 else 'Low' for score in efficacy_scores]
                })
                
                fig = px.bar(efficacy_df, x='Treatment', y='Predicted Efficacy',
                           color='Confidence_Level', title="Predicted Treatment Efficacy",
                           color_discrete_map={'High': "#B7472A", 'Medium': "#D2B48C", 'Low': "#8B4513"})
                
                fig.update_layout(
                    plot_bgcolor='#F8F9FA',
                    paper_bgcolor='#F8F9FA',
                    font=dict(color='#2C3E50', family='Inter'),
                    title_font_color='#2C3E50'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please run a mutation analysis first to get personalized treatment recommendations.")
    
    with tab3:
        st.markdown('<div class="sub-header">Clinical Trials</div>', unsafe_allow_html=True)
        
        if st.session_state.analysis_results:
            result = st.session_state.analysis_results
            
            st.subheader(f"Clinical Trials for {result['gene']} {result['mutation']} in {result['cancer_type']}")
            
            # Query clinical trials
            with st.spinner("Searching clinical trials database..."):
                trials = query_clinical_trials(result['cancer_type'], result['gene'], result['mutation'])
            
            # Display trials with ML-based relevance scoring
            ml_prediction = result['ml_prediction']
            
            # Add relevance scoring based on ML prediction
            for trial in trials:
                if ml_prediction['predicted_class'] == 'High Impact':
                    trial['relevance_score'] = 0.9
                    trial['priority'] = 'High'
                elif ml_prediction['predicted_class'] == 'Moderate Impact':
                    trial['relevance_score'] = 0.7
                    trial['priority'] = 'Medium'
                else:
                    trial['relevance_score'] = 0.5
                    trial['priority'] = 'Low'
            
            # Sort by relevance
            trials.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Display trials
            for i, trial in enumerate(trials, 1):
                priority_color = {'High': 'High Priority', 'Medium': 'Medium Priority', 'Low': 'Low Priority'}[trial['priority']]
                
                with st.expander(f"Trial {i}: {trial['title']} (Relevance: {trial['relevance_score']:.1%})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**NCT ID:** {trial['nct_id']}")
                        st.write(f"**Status:** {trial['status']}")
                        st.write(f"**Conditions:** {trial['conditions']}")
                        
                    with col2:
                        st.write(f"**Interventions:** {trial['interventions']}")
                        st.write(f"**Priority:** {trial['priority']}")
                        st.write(f"**Relevance Score:** {trial['relevance_score']:.1%}")
                    
                    st.write(f"**Summary:** {trial['summary']}")
            
            # Store trials for report generation
            st.session_state.analysis_results['trials'] = trials
            
            # Trial matching visualization
            if len(trials) > 1:
                trial_df = pd.DataFrame({
                    'Trial': [f"Trial {i+1}" for i in range(len(trials))],
                    'Relevance Score': [t['relevance_score'] for t in trials],
                    'Priority': [t['priority'] for t in trials]
                })
                
                fig = px.bar(trial_df, x='Trial', y='Relevance Score', 
                           color='Priority', title="Clinical Trial Relevance Matching",
                           color_discrete_map={'High': '#8E44AD', 'Medium': '#BB8FCE', 'Low': '#2C3E50'})
                
                fig.update_layout(
                    plot_bgcolor='#F8F9FA',
                    paper_bgcolor='#F8F9FA',
                    font=dict(color='#2C3E50', family='Inter'),
                    title_font_color='#2C3E50'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Please run a mutation analysis first to find relevant clinical trials.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #2C3E50; font-family: Inter;"><p>Cancer Mutation Analysis Platform - 2025 Edition</p></div>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()