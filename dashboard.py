import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="User Funnel Conversion Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Data Loading ---
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    # Ensure correct ordering of stages
    stage_order = ['Homepage', 'Product_Page', 'Cart', 'Checkout', 'Purchase']
    df['stage'] = pd.Categorical(df['stage'], categories=stage_order, ordered=True)
    return df

try:
    df = load_data("user_funnel_data.csv")
except FileNotFoundError:
    st.error("Data file 'user_funnel_data.csv' not found. Please ensure it exists in the directory.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Device Filter
all_devices = list(df['device'].unique())
selected_devices = st.sidebar.multiselect("Select Devices", all_devices, default=all_devices)

# Source Filter
all_sources = list(df['source'].unique())
selected_sources = st.sidebar.multiselect("Select Traffic Sources", all_sources, default=all_sources)

# Apply Filters
filtered_df = df[
    (df['device'].isin(selected_devices)) & 
    (df['source'].isin(selected_sources))
]

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# --- Metrics Calculation ---
def calculate_metrics(data):
    # Unique users per stage
    stage_counts = data.groupby('stage')['user_id'].nunique()
    # Reindex to ensure all stages are present, even if count is 0
    stage_order = ['Homepage', 'Product_Page', 'Cart', 'Checkout', 'Purchase']
    stage_counts = stage_counts.reindex(stage_order).fillna(0)
    
    metrics = pd.DataFrame({'Users': stage_counts})
    metrics['Prev_Users'] = metrics['Users'].shift(1)
    metrics['Conversion_Rate_Next'] = (metrics['Users'] / metrics['Prev_Users']).fillna(1.0)
    
    # Percentage of Total (PoT)
    initial_users = metrics['Users'].iloc[0]
    metrics['Pct_of_Total'] = (metrics['Users'] / initial_users).fillna(0)
    
    return metrics

metrics = calculate_metrics(filtered_df)

# --- KPI Section ---
total_users = metrics['Users'].iloc[0]
purchases = metrics['Users'].iloc[-1]
overall_conversion = (purchases / total_users) * 100 if total_users > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Unique Users", f"{int(total_users):,}")
col2.metric("Total Conversions (Purchase)", f"{int(purchases):,}")
col3.metric("Overall Funnel Efficiency", f"{overall_conversion:.2f}%")

# --- Visualizations ---

st.markdown("---")
st.subheader("Funnel Visualization")

tab1, tab2 = st.tabs(["Sankey Diagram (Flow)", "Bar Chart (Metrics)"])

with tab1:
    st.markdown("**Sankey Diagram**: Visualizing user flow and drop-offs between stages.")
    
    # Prepare Sankey Data
    stages = metrics.index.tolist()
    labels = stages + ["Drop-off"] # Nodes: Stages 0..4, Drop-off 5
    
    # Create source, target, value lists
    # Source indices: 0 (Home), 1 (Product), 2 (Cart), 3 (Checkout) -> Target: Next Stage or Drop-off
    # We don't map from Purchase (4) anywhere.
    
    sources = []
    targets = []
    values = []
    colors = []
    
    # Colors for links
    link_color_flow = "rgba(31, 119, 180, 0.4)" # Blueish translucent
    link_color_drop = "rgba(255, 0, 0, 0.2)"   # Reddish translucent
    
    for i in range(len(stages) - 1):
        current_stage = stages[i]
        next_stage = stages[i+1]
        
        users_current = metrics.loc[current_stage, "Users"]
        users_next = metrics.loc[next_stage, "Users"]
        drop_offs = users_current - users_next
        
        # Link to next stage
        if users_next > 0:
            sources.append(i)
            targets.append(i+1)
            values.append(users_next)
            colors.append(link_color_flow)
            
        # Link to drop-off
        # For simplicity, we can route all drop-offs to a single "Drop-off" node (index 5)
        # Or individual drop-off nodes per stage. Single node is cleaner for "leakage" pile.
        if drop_offs > 0:
            sources.append(i)
            targets.append(len(stages)) # Index of "Drop-off" node
            values.append(drop_offs)
            colors.append(link_color_drop)
            
    # Plot Sankey
    fig_sankey = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = labels,
            color = "blue"
        ),
        link = dict(
            source = sources,
            target = targets,
            value = values,
            color = colors
        )
    )])
    
    fig_sankey.update_layout(title_text="User Flow & Drop-offs", font_size=10, height=500)
    st.plotly_chart(fig_sankey, use_container_width=True)

with tab2:
    st.markdown("**Detailed Funnel Metrics**")
    
    # Calculate Error Bars (95% CI) for User Counts based on PoT
    p = metrics['Pct_of_Total']
    n = total_users
    se = np.sqrt(p * (1 - p) / n).fillna(0)
    error_95 = 1.96 * se * n
    
    # Bar Chart with PoT and Error Bars
    fig_bar = go.Figure()
    
    # User Count Bar
    fig_bar.add_trace(go.Bar(
        x=metrics.index,
        y=metrics['Users'],
        error_y=dict(type='data', array=error_95, visible=True),
        text=metrics['Users'].astype(int),
        textposition='auto',
        name='Users',
        marker_color='skyblue',
        hovertemplate='%{y} Users<br>PoT: %{customdata:.1%}<extra></extra>',
        customdata=metrics['Pct_of_Total']
    ))
    
    # Conversion Rate Line (Secondary Y)?? Or just Tooltip?
    # Let's keep it simple: Counts on bars.
    
    # Add annotations for PoT
    # We can add a second trace for text or just use the bar text. 
    # Let's add the PoT text above the bars?
    
    fig_bar.update_layout(
        title="Funnel Counts with 95% Confidence Intervals",
        yaxis_title="Number of Users",
        xaxis_title="Stage",
        height=500
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Data Table
    st.dataframe(metrics.style.format({
        "Users": "{:,.0f}",
        "Prev_Users": "{:,.0f}",
        "Conversion_Rate_Next": "{:.1%}",
        "Pct_of_Total": "{:.1%}"
    }))

# --- A/B Testing Section ---
st.markdown("---")
st.subheader("Experimental: A/B Testing")

st.markdown("Compare conversion rates between two segments.")

col_ab1, col_ab2 = st.columns(2)
with col_ab1:
    ab_metric = st.selectbox("Select Segment to Test", ["device", "source"])

if st.button("Run A/B Test"):
    # Group data by selected segment and calculate conversion to Purchase
    # Converted = Reached Purchase
    
    ab_data = df.copy()
    conversions = ab_data[ab_data['stage'] == 'Purchase'].groupby(ab_metric)['user_id'].nunique()
    totals = ab_data.groupby(ab_metric)['user_id'].nunique()
    
    summary = pd.DataFrame({'Total': totals, 'Converted': conversions}).fillna(0)
    summary['Not_Converted'] = summary['Total'] - summary['Converted']
    summary['Conversion_Rate'] = summary['Converted'] / summary['Total']
    
    st.write("### Segment Performance")
    st.dataframe(summary.style.format({'Conversion_Rate': '{:.2%}'}))
    
    # Chi-Squared Test
    observed = summary[['Converted', 'Not_Converted']].values
    chi2, p_val, dof, expected = stats.chi2_contingency(observed)
    
    st.write("### Statistical Test Results (Chi-Squared)")
    st.write(f"**P-Value**: `{p_val:.4f}`")
    
    if p_val < 0.05:
        st.success("Result: **Statistically Significant Difference** detected (Reject Null Hypothesis).")
    else:
        st.info("Result: **No Significant Difference** detected (Fail to Reject Null Hypothesis).")

    # Visualization of Conversion Rates with Error Bars
    # SE of proportion = sqrt(p(1-p)/n)
    summary['SE'] = np.sqrt(summary['Conversion_Rate'] * (1 - summary['Conversion_Rate']) / summary['Total'])
    summary['CI_95'] = 1.96 * summary['SE']
    
    fig_ab = go.Figure(go.Bar(
        x=summary.index,
        y=summary['Conversion_Rate'],
        error_y=dict(type='data', array=summary['CI_95']),
        text=summary['Conversion_Rate'].apply(lambda x: f"{x:.1%}"),
        textposition='auto',
        marker_color='orange'
    ))
    
    fig_ab.update_layout(
        title=f"Conversion Rate by {ab_metric} (with 95% CI)",
        yaxis_title="Conversion Rate",
        yaxis_tickformat='.0%'
    )
    
    st.plotly_chart(fig_ab, use_container_width=True)
