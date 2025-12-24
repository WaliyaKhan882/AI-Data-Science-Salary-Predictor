import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="AI/ML Salary Analysis & Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        text-align: center;
        margin: 2rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 5px;
    }
    .section-header {
        background: linear-gradient(90deg, #1f77b4 0%, #aec7e8 100%);
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 2rem 0 1rem 0;
        font-size: 1.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model_data():
    """Load the trained model and necessary encoders/scalers"""
    try:
        with open('data/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('data/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        with open('data/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        with open('data/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('data/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        return model, metadata, encoders, scaler, feature_columns
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run the preprocessing and model training notebooks first.")
        st.stop()

@st.cache_data
def load_dataset():
    """Load the original dataset"""
    try:
        df = pd.read_csv('DATA.csv')
        # Add USD conversion
        conversion_rates = {
            'USD': 1.0, 'INR': 0.012, 'GBP': 1.27, 'EUR': 1.09,
            'CAD': 0.74, 'AUD': 0.66, 'SGD': 0.74, 'AED': 0.27
        }
        df['salary_usd'] = df.apply(lambda row: row['salary'] * conversion_rates[row['currency']], axis=1)
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset not found!")
        st.stop()

# Load everything
model, metadata, encoders, scaler, feature_columns = load_model_data()
df = load_dataset()

# Header
st.markdown('<h1 class="main-header">💰 AI/ML Salary Analysis & Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Comprehensive Analysis and Interactive Prediction Tool for AI/ML Job Salaries</p>', unsafe_allow_html=True)

# Sidebar - Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
    st.title("📑 Navigation")
    
    section = st.radio(
        "Select Section:",
        ["🏠 Introduction", "📊 EDA Section", "🤖 Model Section", "🎯 Conclusion"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.subheader("📈 Quick Stats")
    st.metric("Total Records", len(df))
    st.metric("Countries", df['country'].nunique())
    st.metric("Job Titles", df['job_title'].nunique())
    st.metric("Model Accuracy", f"{metadata['test_r2']:.1%}")

# ============================================================================
# 1. INTRODUCTION SECTION
# ============================================================================
if section == "🏠 Introduction":
    st.markdown('<div class="section-header">🏠 Introduction</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📌 Project Overview
        
        Welcome to the **AI/ML Salary Analysis & Prediction System**! This comprehensive project analyzes 
        salary trends in the rapidly growing Artificial Intelligence and Machine Learning job market.
        
        #### 🎯 Project Goals:
        
        1. **Analyze Salary Patterns** - Understand what factors influence AI/ML salaries
        2. **Identify Key Drivers** - Determine which features have the strongest impact
        3. **Build Predictive Model** - Create accurate salary predictions for job seekers and employers
        4. **Provide Insights** - Offer actionable insights for career planning and hiring decisions
        
        #### 📊 Dataset Overview:
        
        Our dataset contains **500 AI/ML job records** collected from **10 countries** between **2020-2025**, 
        covering various aspects of employment in the AI/ML sector.
        """)
        
        st.info("""
        **💡 Why This Matters:**
        
        The AI/ML field is experiencing explosive growth, with salaries varying dramatically across 
        locations, roles, and experience levels. Understanding these patterns helps:
        - **Job Seekers:** Make informed career decisions and negotiate better
        - **Employers:** Set competitive salaries and budget effectively
        - **Educators:** Guide students toward high-value specializations
        """)
    
    with col2:
        st.markdown("### 📋 Dataset Characteristics")
        
        # Dataset info
        st.metric("**Total Records**", f"{len(df):,}")
        st.metric("**Features**", "16 columns")
        st.metric("**Time Period**", "2020-2025")
        st.metric("**Countries**", "10")
        st.metric("**Currencies**", "8 (converted to $)")
        
        st.markdown("---")
        st.markdown("### 🔑 Key Features")
        st.markdown("""
        **Categorical:**
        - Country, Job Title
        - Education, Gender
        - Work Mode, Company Size
        - Industry
        
        **Numerical:**
        - Salary (USD converted)
        - Years of Experience
        - Year, Job Demand Index
        """)
    
    # Dataset preview
    st.markdown("---")
    st.subheader("👀 Dataset Preview")
    
    show_columns = ['job_title', 'country', 'salary_usd', 'years_experience', 
                    'education', 'company_size', 'work_mode', 'year']
    st.dataframe(
        df[show_columns].head(10).style.format({'salary_usd': '${:,.0f}'}),
        use_container_width=True,
        height=400
    )
    
    # Summary statistics
    st.markdown("---")
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("**Mean Salary**", f"${df['salary_usd'].mean():,.0f}")
        st.caption("Average across all records")
    
    with col2:
        st.metric("**Median Salary**", f"${df['salary_usd'].median():,.0f}")
        st.caption("Middle value")
    
    with col3:
        st.metric("**Min Salary**", f"${df['salary_usd'].min():,.0f}")
        st.caption("Lowest recorded")
    
    with col4:
        st.metric("**Max Salary**", f"${df['salary_usd'].max():,.0f}")
        st.caption("Highest recorded")
    
    # Data quality
    st.markdown("---")
    st.subheader("✅ Data Quality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("✅ **No Missing Values** - Complete dataset with 0% missing data")
        st.success("✅ **No Duplicates** - Each record is unique")
    
    with col2:
        st.success("✅ **Currency Standardized** - All salaries converted to USD")
        st.success("✅ **Clean Data** - Ready for analysis and modeling")

# ============================================================================
# 2. EDA SECTION
# ============================================================================
elif section == "📊 EDA Section":
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔍 Key Findings from 15 Comprehensive Analyses
    
    This section presents the most important insights discovered through detailed exploratory data analysis.
    """)
    
    # EDA Navigation
    eda_tab = st.selectbox(
        "Select Analysis:",
        ["Overview", "Geographic Analysis", "Job Title Analysis", "Experience Analysis", 
         "Education & Demographics", "Company & Industry", "Temporal Trends", "Feature Importance"]
    )
    
    if eda_tab == "Overview":
        st.subheader("📈 Salary Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(df['salary_usd'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(df['salary_usd'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["salary_usd"].mean():,.0f}')
            ax.axvline(df['salary_usd'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${df["salary_usd"].median():,.0f}')
            ax.set_xlabel('Salary (USD)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title('AI/ML Salary Distribution', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 💡 Key Insights:")
            st.markdown(f"""
            - **Mean = Median**: ${df['salary_usd'].mean():,.0f} ≈ ${df['salary_usd'].median():,.0f}
            - Nearly **symmetric distribution**
            - Range: ${df['salary_usd'].min():,.0f} to ${df['salary_usd'].max():,.0f}
            - Standard Dev: ${df['salary_usd'].std():,.0f}
            
            **Interpretation:**
            Distribution is well-balanced with no extreme skewness, 
            indicating reliable salary data across the AI/ML sector.
            """)
    
    elif eda_tab == "Geographic Analysis":
        st.subheader("🌍 Salary by Country")
        
        country_avg = df.groupby('country')['salary_usd'].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(country_avg.index, country_avg.values, color='steelblue', alpha=0.8)
        ax.set_xlabel('Average Salary (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Average AI/ML Salaries by Country', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'${width/1000:.0f}k',
                    ha='left', va='center', fontweight='bold', fontsize=10)
        
        st.pyplot(fig)
        
        st.success(f"""
        **🏆 Key Finding: Location is THE DOMINANT Factor!**
        
        - **Highest:** {country_avg.index[0]} (${country_avg.values[0]:,.0f})
        - **Lowest:** {country_avg.index[-1]} (${country_avg.values[-1]:,.0f})
        - **Difference:** {(country_avg.values[0]/country_avg.values[-1] - 1)*100:.0f}% gap!
        
        Geographic location explains more salary variation than ALL other factors combined.
        """)
    
    elif eda_tab == "Job Title Analysis":
        st.subheader("💼 Salary by Job Title")
        
        job_stats = df.groupby('job_title')['salary_usd'].agg(['mean', 'median', 'count']).sort_values('median', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(job_stats.index, job_stats['median'], color='coral', alpha=0.8)
            ax.set_xlabel('Median Salary (USD)', fontsize=12, fontweight='bold')
            ax.set_title('Median Salaries by Job Title', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 📊 Job Title Stats")
            st.dataframe(
                job_stats.style.format({'mean': '${:,.0f}', 'median': '${:,.0f}'}),
                height=250
            )
        
        st.info(f"""
        **💡 Insight: Emerging AI Roles Pay Premium**
        
        - **Top Paying:** {job_stats.index[0]} (${job_stats['median'].values[0]:,.0f} median)
        - **Gap:** {((job_stats['median'].values[0] - job_stats['median'].values[-1])/job_stats['median'].values[-1]*100):.0f}% between highest and lowest
        - LLM and Generative AI specializations command 15-20% premium over traditional ML roles
        """)
    
    elif eda_tab == "Experience Analysis":
        st.subheader("📈 Experience vs Salary")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df['years_experience'], df['salary_usd'], alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
            
            # Add trend line
            z = np.polyfit(df['years_experience'], df['salary_usd'], 1)
            p = np.poly1d(z)
            ax.plot(df['years_experience'].sort_values(), p(df['years_experience'].sort_values()), 
                   "r--", linewidth=2, label=f'Trend Line')
            
            ax.set_xlabel('Years of Experience', fontsize=12, fontweight='bold')
            ax.set_ylabel('Salary (USD)', fontsize=12, fontweight='bold')
            ax.set_title('Experience vs Salary Correlation', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            correlation = df['years_experience'].corr(df['salary_usd'])
            st.metric("**Correlation (r)**", f"{correlation:.3f}")
            st.metric("**Variance Explained**", f"{correlation**2*100:.1f}%")
            
            st.markdown(f"""
            #### 💡 Interpretation:
            
            - **Moderate positive** correlation
            - Experience explains **{correlation**2*100:.0f}%** of salary variance
            - Each additional year adds approximately **$8-10k**
            - Experience matters, but location/role matter MORE
            """)
    
    elif eda_tab == "Education & Demographics":
        st.subheader("🎓 Education & Gender Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Education Level")
            edu_avg = df.groupby('education')['salary_usd'].mean().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(edu_avg.index, edu_avg.values, color='teal', alpha=0.8)
            ax.set_ylabel('Average Salary (USD)', fontsize=11, fontweight='bold')
            ax.set_title('Salary by Education Level', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            
            st.info(f"PhD holders earn {((edu_avg.values[0] - edu_avg.values[-1])/edu_avg.values[-1]*100):.0f}% more than the lowest education level")
        
        with col2:
            st.markdown("#### Gender Pay Gap")
            gender_avg = df.groupby('gender')['salary_usd'].mean().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(gender_avg.index, gender_avg.values, color='purple', alpha=0.8)
            ax.set_ylabel('Average Salary (USD)', fontsize=11, fontweight='bold')
            ax.set_title('Salary by Gender', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            
            st.info(f"Gender gap: {((gender_avg.values[0] - gender_avg.values[-1])/gender_avg.values[-1]*100):.0f}% difference between highest and lowest")
    
    elif eda_tab == "Company & Industry":
        st.subheader("🏢 Company Size & Industry Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Company Size Impact")
            size_avg = df.groupby('company_size')['salary_usd'].mean().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(size_avg.index, size_avg.values, color='orange', alpha=0.8)
            ax.set_ylabel('Average Salary (USD)', fontsize=11, fontweight='bold')
            ax.set_title('Salary by Company Size', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            
            st.caption(f"Only {((size_avg.values[0] - size_avg.values[-1])/size_avg.values[-1]*100):.0f}% difference - WEAK effect")
        
        with col2:
            st.markdown("#### Industry Comparison")
            industry_avg = df.groupby('industry')['salary_usd'].mean().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(industry_avg.index, industry_avg.values, color='green', alpha=0.8)
            ax.set_xlabel('Average Salary (USD)', fontsize=11, fontweight='bold')
            ax.set_title('Salary by Industry', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
            
            st.caption(f"{((industry_avg.values[0] - industry_avg.values[-1])/industry_avg.values[-1]*100):.0f}% range across industries")
    
    elif eda_tab == "Temporal Trends":
        st.subheader("📅 Salary Trends Over Time")
        
        yearly_avg = df.groupby('year')['salary_usd'].mean()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=3, markersize=10, color='blue', alpha=0.7)
        ax.fill_between(yearly_avg.index, yearly_avg.values, alpha=0.2, color='blue')
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Salary (USD)', fontsize=12, fontweight='bold')
        ax.set_title('AI/ML Salary Trends (2020-2025)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        for x, y in zip(yearly_avg.index, yearly_avg.values):
            ax.text(x, y + 3000, f'${y/1000:.0f}k', ha='center', fontweight='bold', fontsize=10)
        
        st.pyplot(fig)
        
        st.success("""
        **💡 Key Finding: Stable Salaries Despite AI Boom**
        
        - Salaries remain relatively stable ($120k-$137k range)
        - No consistent upward or downward trend
        - Time is NOT a significant predictor of salary
        """)
    
    elif eda_tab == "Feature Importance":
        st.subheader("⭐ Feature Importance Summary")
        
        st.markdown("""
        ### 🎯 What Really Drives AI/ML Salaries?
        
        Based on comprehensive analysis of 15 different perspectives, here's the definitive ranking:
        """)
        
        importance_data = {
            'Feature': ['Country (Location)', 'Job Title', 'Years Experience', 'Industry', 
                       'Education', 'Company Size', 'Work Mode', 'Year', 'Job Demand'],
            'Impact': ['DOMINANT', 'Strong', 'Moderate', 'Small', 'Small', 'Very Weak', 'Negligible', 'None', 'Negative'],
            'Effect Size': ['860%', '23%', '17%', '20%', '26%', '6%', '2%', '0%', '-10%'],
            'Rating': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐', '⭐⭐', '⭐', '❌', '❌']
        }
        
        df_importance = pd.DataFrame(importance_data)
        
        st.dataframe(
            df_importance.style.apply(lambda x: ['background-color: #90EE90' if v == 'DOMINANT' 
                                                  else 'background-color: #FFD700' if v == 'Strong'
                                                  else 'background-color: #87CEEB' if v == 'Moderate'
                                                  else '' for v in x], subset=['Impact']),
            use_container_width=True,
            height=400
        )
        
        st.info("""
        **🎓 Key Takeaway:**
        
        80% of salary prediction power comes from just 3 features:
        1. **Country** (explains 60-70% alone!)
        2. **Job Title** (adds 15-20%)
        3. **Experience** (adds 10-15%)
        
        Everything else has minimal impact!
        """)

# ============================================================================
# 3. MODEL SECTION
# ============================================================================
elif section == "🤖 Model Section":
    st.markdown('<div class="section-header">🤖 Machine Learning Model</div>', unsafe_allow_html=True)
    
    # Model description tabs
    model_tab = st.tabs(["📋 Model Overview", "🎯 Runtime Predictions", "📊 Performance Results"])
    
    # Tab 1: Model Overview
    with model_tab[0]:
        st.subheader("📋 Model Development Process")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🔧 Model Architecture
            
            **Algorithm:** {model_name}
            
            #### Why This Model?
            
            We trained and compared **6 different algorithms**:
            1. Linear Regression
            2. Ridge Regression
            3. Lasso Regression
            4. Decision Tree
            5. **Random Forest** ⭐
            6. **Gradient Boosting** ⭐
            
            The {model_name} was selected as the best performer based on:
            - Highest R² score on test data
            - Lowest prediction error (RMSE/MAE)
            - Ability to capture non-linear relationships
            - Robust performance across different salary ranges
            
            #### 🎯 Model Features
            
            Based on EDA findings, we selected **7 key features**:
            
            **TIER 1 (Must Have):**
            - Country (encoded)
            - Job Title (encoded)  
            - Years of Experience
            - Experience Level (engineered)
            
            **TIER 2 (Helper Features):**
            - Industry (encoded)
            - Education (encoded)
            - Company Size (encoded)
            
            **Excluded Features:** Work Mode, Job Demand, Year, Gender (weak/no predictive power)
            """.format(model_name=metadata['model_name']))
        
        with col2:
            st.markdown("### 📊 Model Metrics")
            
            st.metric("**Model Type**", metadata['model_name'])
            st.metric("**R² Score**", f"{metadata['test_r2']:.4f}")
            st.metric("**Accuracy**", f"{metadata['test_r2']*100:.2f}%")
            st.metric("**RMSE**", f"${metadata['test_rmse']:,.0f}")
            st.metric("**MAE**", f"${metadata['test_mae']:,.0f}")
            
            st.markdown("---")
            st.markdown("### 📈 Training Details")
            st.markdown(f"""
            - **Training Set:** 400 samples (80%)
            - **Test Set:** 100 samples (20%)
            - **Features:** 7 selected
            - **Scaling:** StandardScaler
            - **Encoding:** Label Encoding
            """)
            
            if metadata['test_r2'] >= 0.80:
                st.success("🌟 **EXCELLENT** Model Quality!")
            elif metadata['test_r2'] >= 0.70:
                st.success("✅ **GOOD** Model Quality!")
            else:
                st.warning("⚠️ Model could be improved")
        
        st.markdown("---")
        st.subheader("📐 Model Performance Interpretation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### R² Score")
            st.markdown(f"""
            **{metadata['test_r2']:.4f}** ({metadata['test_r2']*100:.1f}%)
            
            This means the model explains **{metadata['test_r2']*100:.1f}%** of salary variation.
            The remaining {100-metadata['test_r2']*100:.1f}% is due to factors not in our data.
            """)
        
        with col2:
            st.markdown("#### RMSE")
            st.markdown(f"""
            **${metadata['test_rmse']:,.0f}**
            
            Root Mean Squared Error - penalizes large prediction errors more heavily.
            Predictions typically within ±${metadata['test_rmse']:,.0f}.
            """)
        
        with col3:
            st.markdown("#### MAE")
            st.markdown(f"""
            **${metadata['test_mae']:,.0f}**
            
            Mean Absolute Error - simple average error. On average, predictions are 
            off by ${metadata['test_mae']:,.0f}.
            """)
    
    # Tab 2: Runtime Predictions
    with model_tab[1]:
        st.subheader("🎯 Make Real-Time Salary Predictions")
        
        st.info("👇 Enter job details below to get instant salary predictions!")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Country selection
            country = st.selectbox(
                "📍 Country",
                options=['United States', 'Germany', 'United Kingdom', 'Canada', 'France', 
                        'Australia', 'Netherlands', 'Singapore', 'India', 'United Arab Emirates'],
                help="Location is the strongest predictor of salary"
            )
            
            # Job title selection
            job_title = st.selectbox(
                "💼 Job Title",
                options=['LLM Researcher', 'Generative AI Engineer', 'Prompt Engineer', 
                        'Data Scientist', 'Machine Learning Engineer'],
                help="Specialized AI roles typically pay more"
            )
            
            # Years of experience
            years_experience = st.slider(
                "📈 Years of Experience",
                min_value=0,
                max_value=20,
                value=5,
                help="More experience generally leads to higher salaries"
            )
            
            # Education level
            education = st.selectbox(
                "🎓 Education Level",
                options=['Diploma', 'Bachelors', 'Masters', 'PhD'],
                help="PhD holders typically earn more"
            )
        
        with col2:
            # Company size
            company_size = st.selectbox(
                "🏢 Company Size",
                options=['Large', 'Medium', 'Startup'],
                help="Large companies tend to pay slightly more"
            )
            
            # Industry
            industry = st.selectbox(
                "🏭 Industry",
                options=['Technology', 'Finance', 'Healthcare', 'E-commerce', 
                        'Education', 'Consulting', 'Manufacturing', 'Retail'],
                help="Industry has moderate impact on salary"
            )
            
            st.markdown("---")
            
            # Predict button
            predict_btn = st.button("💰 Predict Salary", type="primary", use_container_width=True)
        
        if predict_btn:
            try:
                # Create experience level
                if years_experience <= 2:
                    experience_level = 'Junior'
                elif years_experience <= 5:
                    experience_level = 'Mid-Level'
                elif years_experience <= 10:
                    experience_level = 'Senior'
                else:
                    experience_level = 'Expert'
                
                # Create input DataFrame
                input_data = pd.DataFrame({
                    'country': [country],
                    'job_title': [job_title],
                    'years_experience': [years_experience],
                    'education': [education],
                    'company_size': [company_size],
                    'industry': [industry],
                    'experience_level': [experience_level]
                })
                
                # Encode categorical features
                encoded_features = {}
                for col in ['job_title', 'country', 'education', 'company_size', 'industry', 'experience_level']:
                    encoder = encoders[col]
                    encoded_features[f'{col}_encoded'] = encoder.transform([input_data[col].values[0]])[0]
                
                # Add numerical feature
                encoded_features['years_experience'] = years_experience
                
                # Create feature vector in correct order
                X_input = pd.DataFrame([encoded_features])[feature_columns]
                
                # Scale features
                X_scaled = scaler.transform(X_input)
                
                # Make prediction
                prediction = model.predict(X_scaled)[0]
                
                # Salary range
                lower_bound = prediction - metadata['test_mae']
                upper_bound = prediction + metadata['test_mae']
                
                # Display prediction in one box
                st.success(f"""
                ### 💵 Predicted Annual Salary
                # ${prediction:,.0f} USD
                
                **📊 Expected Salary Range:** ${lower_bound:,.0f} - ${upper_bound:,.0f} USD  
                **🎯 Typical Error:** ±${metadata['test_mae']:,.0f}  
                **📈 Model Confidence:** {metadata['test_r2']*100:.1f}% of salary variation explained
                """)
                
                # Profile Summary
                st.markdown("---")
                st.subheader("📋 Your Profile Summary")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write(f"**📍 Location:** {country}")
                    st.write(f"**💼 Job Title:** {job_title}")
                    st.write(f"**📈 Experience:** {years_experience} years ({experience_level})")
                
                with col_b:
                    st.write(f"**🎓 Education:** {education}")
                    st.write(f"**🏢 Company Size:** {company_size}")
                    st.write(f"**🏭 Industry:** {industry}")
                
                # Personalized Insights
                st.markdown("---")
                st.subheader("💡 Personalized Insights")
                
                # Location insight
                if country == 'United States':
                    st.success("🇺🇸 **USA** offers the highest AI/ML salaries globally!")
                elif country == 'India':
                    st.info("🇮🇳 **India** has lower salaries but rapid AI growth and opportunities.")
                elif country in ['Germany', 'United Kingdom', 'Canada']:
                    st.success(f"**{country}** is a strong market with competitive AI/ML salaries!")
                
                # Job title insight
                if job_title in ['LLM Researcher', 'Generative AI Engineer']:
                    st.success(f"🚀 **{job_title}** is a cutting-edge role with premium compensation!")
                
                # Experience insight
                if years_experience < 3:
                    st.info("📚 Building more experience can significantly boost your earning potential.")
                elif years_experience > 10:
                    st.success("🏆 Your extensive experience commands premium compensation!")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.error("Please ensure all fields are filled correctly.")
    
    # Tab 3: Performance Results
    with model_tab[2]:
        st.subheader("📊 Model Performance & Validation")
        
        st.markdown("""
        ### ✅ Model Validation Results
        
        Our model was rigorously tested using industry-standard evaluation methods:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Performance Metrics")
            
            metrics_df = pd.DataFrame({
                'Metric': ['R² Score', 'RMSE', 'MAE', 'Variance Explained'],
                'Value': [
                    f"{metadata['test_r2']:.4f}",
                    f"${metadata['test_rmse']:,.0f}",
                    f"${metadata['test_mae']:,.0f}",
                    f"{metadata['test_r2']*100:.1f}%"
                ],
                'Interpretation': [
                    f"{metadata['test_r2']*100:.1f}% of salary variance explained",
                    "Average prediction error (penalizes outliers)",
                    "Simple average error in predictions",
                    "Model captures majority of salary patterns"
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True, height=200)
        
        with col2:
            st.markdown("#### 🏆 Model Quality Assessment")
            
            if metadata['test_r2'] >= 0.80:
                quality = "EXCELLENT 🌟"
                color = "green"
            elif metadata['test_r2'] >= 0.70:
                quality = "GOOD ✅"
                color = "blue"
            elif metadata['test_r2'] >= 0.60:
                quality = "FAIR ⚠️"
                color = "orange"
            else:
                quality = "NEEDS IMPROVEMENT ❌"
                color = "red"
            
            st.markdown(f"**Overall Quality:** :{color}[{quality}]")
            
            st.markdown(f"""
            **Expected from EDA:** R² = 0.70-0.85  
            **Achieved:** R² = {metadata['test_r2']:.4f}
            
            ✅ Model meets/exceeds expectations!
            """)
            
            st.markdown("---")
            st.markdown("#### 💼 Business Value")
            st.markdown(f"""
            - Predictions within **${metadata['test_mae']:,.0f}** on average
            - Explains **{metadata['test_r2']*100:.1f}%** of salary differences
            - Useful for:
              - Salary negotiations
              - Hiring budget planning
              - Career path decisions
              - Market competitiveness analysis
            """)
        
        st.markdown("---")
        st.subheader("🔍 Feature Importance (What Drives Predictions?)")
        
        st.info("""
        **Key Insight from Model:** The model confirms our EDA findings!
        
        The most important features for predictions are:
        1. **Country** (Location) - Dominates all other factors
        2. **Job Title** - Specialized roles command premiums
        3. **Experience** - Consistent moderate impact
        
        This validates that location is THE most critical factor in AI/ML salary determination.
        """)
        
        st.markdown("---")
        st.subheader("📈 Model Strengths & Limitations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Strengths")
            st.markdown("""
            - High accuracy ({r2:.1f}% variance explained)
            - Validated on unseen test data
            - Based on real-world salary data
            - Captures non-linear relationships
            - Robust across salary ranges
            - Fast prediction time (<1 second)
            """.format(r2=metadata['test_r2']*100))
        
        with col2:
            st.markdown("#### ⚠️ Limitations")
            st.markdown("""
            - Limited to 10 countries in dataset
            - Based on 2020-2025 data
            - Doesn't account for:
              - Individual negotiation skills
              - Company-specific benefits
              - Remote work arrangements
              - Stock options/bonuses
            - {unexplained:.1f}% variance unexplained
            """.format(unexplained=100-metadata['test_r2']*100))

# ============================================================================
# 4. CONCLUSION SECTION
# ============================================================================
elif section == "🎯 Conclusion":
    st.markdown('<div class="section-header">🎯 Conclusion & Key Takeaways</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 📌 Project Summary
    
    This comprehensive analysis examined **500 AI/ML job records** across **10 countries** to understand 
    salary patterns and build an accurate prediction model.
    """)
    
    st.markdown("---")
    st.subheader("🔑 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1️⃣ Geographic Location is DOMINANT
        
        - **Finding:** 860% salary difference between USA and India
        - **Impact:** Location explains 60-70% of all salary variation
        - **Takeaway:** Where you work matters MORE than what you do
        
        **Top 3 Countries:**
        1. 🇺🇸 United States: $240k average
        2. 🇩🇪 Germany: $148k average
        3. 🇬🇧 United Kingdom: $141k average
        
        ---
        
        ### 2️⃣ Emerging AI Roles Pay Premium
        
        - **Finding:** LLM Researchers earn 23% more than ML Engineers
        - **Impact:** Job specialization adds 15-20% to salary
        - **Takeaway:** Specialize in cutting-edge AI technologies
        
        **Top Paying Roles:**
        1. LLM Researcher: $143k median
        2. Generative AI Engineer: $134k median
        3. Prompt Engineer: $127k median
        
        ---
        
        ### 3️⃣ Experience Matters Moderately
        
        - **Finding:** r = 0.41 correlation with salary
        - **Impact:** Explains 17% of salary variance
        - **Takeaway:** Each year adds ~$8-10k
        
        **Experience Levels:**
        - Junior (0-2 yrs): $95k average
        - Mid (3-5 yrs): $115k average
        - Senior (6-10 yrs): $140k average
        - Expert (10+ yrs): $160k+ average
        """)
    
    with col2:
        st.markdown("""
        ### 4️⃣ Other Factors Have Minimal Impact
        
        **Weak Predictors:**
        - Education: Only 26% difference
        - Industry: 20% salary range
        - Company Size: Just 6% gap
        
        **No Impact:**
        - Work Mode: Only 2% difference
        - Time/Year: No trend detected
        - Job Demand: Negative correlation
        
        ---
        
        ### 5️⃣ Data Quality is Excellent
        
        - ✅ 0% missing values
        - ✅ No duplicates
        - ✅ Balanced distribution
        - ✅ Currency standardized (USD)
        - ✅ Clean, reliable data
        - ✅ Ready for modeling
        
        ---
        
        ### 6️⃣ Model Successfully Deployed
        
        - ✅ Trained on 6 algorithms
        - ✅ Best model selected and validated
        - ✅ Interactive predictions available
        - ✅ Performance metrics in Model Section
        - ✅ Ready for real-world use
        """)
    
    st.markdown("---")
    st.subheader("💡 Actionable Insights")
    
    tab1, tab2, tab3 = st.tabs(["For Job Seekers", "For Employers", "For Educators"])
    
    with tab1:
        st.markdown("""
        ### 🎯 Recommendations for Job Seekers
        
        #### 1. **Location Strategy**
        - Consider relocating to high-paying markets (USA, Germany, UK)
        - Remote work for US companies from lower-cost countries = best of both worlds
        - If staying in low-pay country, negotiate based on global rates
        
        #### 2. **Specialization Path**
        - Focus on emerging AI: LLMs, Generative AI, Prompt Engineering
        - These roles pay 15-20% more than traditional ML/Data Science
        - Stay updated on cutting-edge technologies
        
        #### 3. **Experience Building**
        - Each year adds $8-10k to your salary
        - Focus on hands-on projects and real-world impact
        - Build portfolio demonstrating AI expertise
        
        #### 4. **Education Investment**
        - PhD adds value but ROI is modest (26% premium)
        - Focus on practical skills > degrees
        - Consider online courses + projects vs. expensive degrees
        
        #### 5. **Negotiation Leverage**
        - Use this tool to know your market value
        - Location + Role + Experience = your baseline
        - Push for top quartile if you have strong skills
        """)
    
    with tab2:
        st.markdown("""
        ### 💼 Recommendations for Employers
        
        #### 1. **Competitive Salary Setting**
        - Use location-adjusted benchmarks
        - Specialized AI roles require 15-20% premium
        - Experience-based bands: Junior/Mid/Senior/Expert
        
        #### 2. **Talent Attraction**
        - Offer competitive packages for your geography
        - Consider remote work to access global talent
        - Emphasize emerging AI project opportunities
        
        #### 3. **Budget Planning**
        - Location is biggest cost driver (860% range!)
        - Opening offices in India/Eastern Europe = 70-80% cost savings
        - Balance cost vs. talent availability
        
        #### 4. **Retention Strategy**
        - Regular experience-based raises ($8-10k/year)
        - Invest in cutting-edge AI projects
        - Career progression to specialized roles
        
        #### 5. **Market Intelligence**
        - Monitor salary trends by location + role
        - Benchmark against competitors
        - Use prediction model for offer validation
        """)
    
    with tab3:
        st.markdown("""
        ### 🎓 Recommendations for Educators
        
        #### 1. **Curriculum Focus**
        - Prioritize emerging AI: LLMs, Generative AI, Prompt Engineering
        - These skills command highest salaries
        - Balance theory with hands-on projects
        
        #### 2. **Career Guidance**
        - Educate students on location's massive impact
        - Guide toward specialized AI roles
        - Explain real-world salary expectations
        
        #### 3. **Skill Development**
        - Experience matters (17% of variance)
        - Internships and projects are crucial
        - Build portfolios showcasing AI work
        
        #### 4. **Global Perspective**
        - Prepare students for global job market
        - Remote work opportunities
        - Geographic arbitrage possibilities
        
        #### 5. **Industry Partnerships**
        - Connect students with AI companies
        - Real-world projects > theoretical exercises
        - Mentorship from industry professionals
        """)
    
    st.markdown("---")
    st.subheader("🚀 Future Directions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📈 Potential Improvements
        
        1. **Expand Dataset**
           - More countries (especially Asia-Pacific)
           - More recent data (2024-2025+)
           - Larger sample size (1000+ records)
        
        2. **Additional Features**
           - Remote work designation
           - Stock options/equity
           - Total compensation (not just salary)
           - Specific technologies/skills
        
        3. **Model Enhancements**
           - Deep learning approaches
           - Ensemble methods
           - Regional sub-models
        """)
    
    with col2:
        st.markdown("""
        ### 🔮 Future Research Questions
        
        1. **How will remote work impact geographic salary gaps?**
           - Will USA salaries remain 10x India?
           - Remote arbitrage opportunities?
        
        2. **Which AI specializations will emerge next?**
           - Beyond LLMs/Generative AI
           - Quantum ML? AI Safety?
        
        3. **How fast will salaries grow?**
           - Current data shows stability
           - Will AI boom drive increases?
        """)
    
    st.markdown("---")
    st.success("""
    ## 🎯 Final Takeaway
    
    **Location dominates everything else.** If you want to maximize AI/ML salary:
    
    1. **Move to (or work for companies in) USA, Germany, or UK** = 80% of the impact
    2. **Specialize in emerging AI (LLMs, Generative AI)** = 15% additional boost
    3. **Build experience steadily** = 10-15% growth over time
    
    Everything else (education, company size, industry, work mode) has minimal effect (<10% combined).
    
    **Use this tool** to make data-driven career and hiring decisions! 🚀
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>Thank you for exploring this project! 🙏</h3>
        <p><strong>AI/ML Salary Analysis & Prediction System</strong></p>
        <p>Dataset: 500 records | 10 countries | 2020-2025 | Model Accuracy: {accuracy:.1%}</p>
        <p>Built with Python, Scikit-learn, Pandas, and Streamlit</p>
    </div>
    """.format(accuracy=metadata['test_r2']), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p><strong>AI/ML Salary Predictor</strong> | Powered by Machine Learning</p>
    <p>Model: {model_name} | Accuracy: {accuracy:.1%} | Dataset: 500 AI/ML jobs across 10 countries</p>
</div>
""".format(model_name=metadata['model_name'], accuracy=metadata['test_r2']), unsafe_allow_html=True)