import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Road Accident ML Dashboard", layout="wide", page_icon="🚗")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🚗 Road Accident Analysis & Prediction Dashboard")
st.markdown("---")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Display basic info
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.write(f"**Rows:** {df.shape[0]}")
    st.sidebar.write(f"**Columns:** {df.shape[1]}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🔍 EDA", "🤖 Model Training", "📈 Predictions", "📉 Comparisons"])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Statistics")
            missing = df.isnull().sum().sum()
            st.metric("Missing Values", missing)
            st.metric("Numeric Columns", df.select_dtypes(include=[np.number]).shape[1])
            st.metric("Categorical Columns", df.select_dtypes(include=['object']).shape[1])
        
        st.subheader("Data Description")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        st.header("Exploratory Data Analysis")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 States by Accidents (2020)")
            if 'Road Accidents  during 2020' in df.columns and 'State' in df.columns:
                top_states = df.nlargest(10, 'Road Accidents  during 2020')[['State', 'Road Accidents  during 2020']]
                fig = px.bar(top_states, x='State', y='Road Accidents  during 2020',
                           color='Road Accidents  during 2020',
                           color_continuous_scale='Reds')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top 10 States by Deaths (2020)")
            if 'Persons Killed 2020' in df.columns and 'State' in df.columns:
                top_deaths = df.nlargest(10, 'Persons Killed 2020')[['State', 'Persons Killed 2020']]
                fig = px.bar(top_deaths, x='State', y='Persons Killed 2020',
                           color='Persons Killed 2020',
                           color_continuous_scale='Oranges')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Trend Analysis
        st.subheader("Accident Trends Over Years")
        accident_cols = ['Road Accidents  during 2018', ' Road Accidents  during 2019', 'Road Accidents  during 2020']
        if all(col in df.columns for col in accident_cols):
            yearly_total = {
                '2018': df['Road Accidents  during 2018'].sum(),
                '2019': df[' Road Accidents  during 2019'].sum(),
                '2020': df['Road Accidents  during 2020'].sum()
            }
            fig = go.Figure(data=[go.Scatter(x=list(yearly_total.keys()), y=list(yearly_total.values()),
                                            mode='lines+markers', line=dict(width=3))])
            fig.update_layout(title="Total Accidents Trend", xaxis_title="Year", yaxis_title="Total Accidents")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Heatmap
        st.subheader("Correlation Matrix")
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Model Training & Evaluation")
        
        # Target selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = st.selectbox("Select Target Variable", numeric_cols, 
                                 index=numeric_cols.index('Road Accidents  during 2020') if 'Road Accidents  during 2020' in numeric_cols else 0)
        
        # Feature selection
        feature_cols = st.multiselect("Select Features", 
                                     [col for col in numeric_cols if col != target_col],
                                     default=[col for col in numeric_cols if col != target_col][:5])
        
        if len(feature_cols) > 0:
            # Prepare data
            X = df[feature_cols].fillna(df[feature_cols].median())
            y = df[target_col].fillna(df[target_col].median())
            
            # Split configuration
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
            with col2:
                random_state = st.number_input("Random State", 0, 100, 42)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Model selection
            st.subheader("Select Models to Train")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                use_rf = st.checkbox("Random Forest", value=True)
                use_lr = st.checkbox("Linear Regression", value=True)
            with col2:
                use_gb = st.checkbox("Gradient Boosting", value=True)
                use_ridge = st.checkbox("Ridge Regression", value=True)
            with col3:
                use_dt = st.checkbox("Decision Tree", value=False)
                use_lasso = st.checkbox("Lasso Regression", value=False)
            with col4:
                use_svr = st.checkbox("SVR", value=False)
            
            if st.button("🚀 Train Models", type="primary"):
                results = {}
                
                with st.spinner("Training models..."):
                    progress_bar = st.progress(0)
                    models_to_train = []
                    
                    if use_rf:
                        models_to_train.append(("Random Forest", RandomForestRegressor(n_estimators=100, random_state=random_state)))
                    if use_lr:
                        models_to_train.append(("Linear Regression", LinearRegression()))
                    if use_gb:
                        models_to_train.append(("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=random_state)))
                    if use_ridge:
                        models_to_train.append(("Ridge Regression", Ridge()))
                    if use_dt:
                        models_to_train.append(("Decision Tree", DecisionTreeRegressor(random_state=random_state)))
                    if use_lasso:
                        models_to_train.append(("Lasso Regression", Lasso()))
                    if use_svr:
                        models_to_train.append(("SVR", SVR()))
                    
                    for idx, (name, model) in enumerate(models_to_train):
                        if name in ["Linear Regression", "Ridge Regression", "Lasso Regression", "SVR"]:
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        results[name] = {
                            'model': model,
                            'predictions': y_pred,
                            'MSE': mse,
                            'RMSE': rmse,
                            'MAE': mae,
                            'R2': r2
                        }
                        
                        progress_bar.progress((idx + 1) / len(models_to_train))
                
                st.success("✅ Training completed!")
                
                # Store results in session state
                st.session_state['results'] = results
                st.session_state['y_test'] = y_test
                st.session_state['X_test'] = X_test
                st.session_state['X_test_scaled'] = X_test_scaled
                st.session_state['scaler'] = scaler
                st.session_state['feature_cols'] = feature_cols
                
                # Display results
                st.subheader("Model Performance")
                
                results_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'RMSE': [results[m]['RMSE'] for m in results.keys()],
                    'MAE': [results[m]['MAE'] for m in results.keys()],
                    'R² Score': [results[m]['R2'] for m in results.keys()]
                })
                
                st.dataframe(results_df.style.highlight_max(subset=['R² Score'], color='lightgreen')
                           .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'), 
                           use_container_width=True)
                
                # Best model
                best_model = max(results.keys(), key=lambda x: results[x]['R2'])
                st.success(f"🏆 Best Model: **{best_model}** (R² = {results[best_model]['R2']:.4f})")
    
    with tab4:
        st.header("Predictions")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            y_test = st.session_state['y_test']
            
            model_name = st.selectbox("Select Model for Predictions", list(results.keys()))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Actual vs Predicted")
                pred_df = pd.DataFrame({
                    'Actual': y_test.values,
                    'Predicted': results[model_name]['predictions']
                })
                fig = px.scatter(pred_df, x='Actual', y='Predicted', 
                               title=f"{model_name} - Actual vs Predicted")
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                        y=[y_test.min(), y_test.max()],
                                        mode='lines', name='Perfect Prediction',
                                        line=dict(color='red', dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Residual Plot")
                residuals = y_test.values - results[model_name]['predictions']
                fig = px.scatter(x=results[model_name]['predictions'], y=residuals,
                               labels={'x': 'Predicted', 'y': 'Residuals'},
                               title=f"{model_name} - Residual Plot")
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance (for tree-based models)
            if model_name in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state['feature_cols'],
                    'Importance': results[model_name]['model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                           title=f"{model_name} - Feature Importance")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Please train models first in the 'Model Training' tab")
    
    with tab5:
        st.header("Model Comparison")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            # Metrics comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("RMSE Comparison")
                rmse_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'RMSE': [results[m]['RMSE'] for m in results.keys()]
                }).sort_values('RMSE')
                fig = px.bar(rmse_df, x='Model', y='RMSE', color='RMSE',
                           color_continuous_scale='Reds')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("R² Score Comparison")
                r2_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'R² Score': [results[m]['R2'] for m in results.keys()]
                }).sort_values('R² Score', ascending=False)
                fig = px.bar(r2_df, x='Model', y='R² Score', color='R² Score',
                           color_continuous_scale='Greens')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics table
            st.subheader("Detailed Metrics Comparison")
            comparison_df = pd.DataFrame({
                'Model': list(results.keys()),
                'MSE': [f"{results[m]['MSE']:.2f}" for m in results.keys()],
                'RMSE': [f"{results[m]['RMSE']:.2f}" for m in results.keys()],
                'MAE': [f"{results[m]['MAE']:.2f}" for m in results.keys()],
                'R² Score': [f"{results[m]['R2']:.4f}" for m in results.keys()]
            })
            st.dataframe(comparison_df, use_container_width=True)
            
            # Download results
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="model_comparison_results.csv",
                mime="text/csv"
            )
        else:
            st.info("👆 Please train models first in the 'Model Training' tab")

else:
    st.info("👆 Please upload a CSV file to begin analysis")
    
    # Instructions
    st.markdown("""
    ### 📋 Instructions:
    1. Upload your road accident CSV file
    2. Explore the data in the **Data Overview** and **EDA** tabs
    3. Train multiple ML models in the **Model Training** tab
    4. View predictions and analyze results in the **Predictions** tab
    5. Compare model performance in the **Comparisons** tab
    
    ### 🤖 Available Models:
    - Random Forest Regressor
    - Linear Regression
    - Gradient Boosting Regressor
    - Ridge Regression
    - Decision Tree Regressor
    - Lasso Regression
    - Support Vector Regressor (SVR)
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.info("This dashboard provides comprehensive ML analysis for road accident prediction using multiple regression models.")
