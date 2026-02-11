import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title='Customer Churn Portfolio',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/bank_churn_data.csv")

df = load_data()
df.columns = df.columns.str.lower().str.replace(' ', '_')

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.header("📌 Navigation")

pages = st.sidebar.radio(
    "Select Page:",
    [
        "Project Overview",
        "Exploratory Data Analysis",
        "Churn Prediction Model",
        "Model Evaluation",
        "Insight & Recommendation"
    ]
)

# =========================
# PAGE 1 — PROJECT OVERVIEW
# =========================
if pages == "Project Overview":
    st.title("Bank Customer Churn Analysis")

    st.markdown(
    """
    <div style="text-align: justify;">
        <p>
        This project is conducted in the <b>banking and credit card industry</b>, where customer retention plays a
        critical role in maintaining long-term profitability.
        The main business problem addressed in this analysis is <b>customer churn</b>, which occurs when customers
        stop using the bank's credit card services. High churn rates often indicate issues related to customer
        engagement, satisfaction, or product suitability.
        Customer churn has a significant business impact, including <b>loss of recurring revenue</b> and
        <b>higher customer acquisition costs</b>, as acquiring new customers is generally more expensive than
        retaining existing ones.
        </p>
    </div>
    """,
    unsafe_allow_html=True
    )

    # METRIC SUMMARY
    col1, col2, col3, col4 = st.columns(4)

    total_customer = df.shape[0]
    existing_customer = df[df['attrition_flag'] == 'Existing Customer'].shape[0]
    attrited_customer = df[df['attrition_flag'] == 'Attrited Customer'].shape[0]
    churn_rate = attrited_customer / total_customer * 100

    with col1:
        with st.container(border=True):
            st.metric("Total Customers", f"{total_customer:,}")
    with col2:
        with st.container(border=True):
            st.metric("Existing Customers", f"{existing_customer:,}")
    with col3:
        with st.container(border=True):
            st.metric("Attrited Customers", f"{attrited_customer:,}")
    with col4:
        with st.container(border=True):
            st.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns([1.5, 1])
 
    # DATASET OVERVIEW (TABLE & KEY INSIGHTS)
    dataset_overview = pd.DataFrame({
        "Column Name": [
            "user_id", "attrition_flag", "customer_age", "gender", "dependent_count",
            "education_level", "marital_status", "income_category", "card_category",
            "months_on_book", "total_relationship_count", "months_inactive_12_mon",
            "contacts_count_12_mon", "credit_limit", "total_revolving_bal",
            "avg_open_to_buy", "total_amt_chng_q4_q1", "total_trans_amt",
            "total_trans_ct", "total_ct_chng_q4_q1", "avg_utilization_ratio"
        ],
        "Description": [
            "Customer account number",
            "Customer status (Existing or Attrited)",
            "Age of the customer",
            "Gender of customer",
            "Number of dependents",
            "Customer education level",
            "Customer marital status",
            "Customer income category",
            "Type of credit card used",
            "Length of relationship with bank (months)",
            "Number of bank products used",
            "Inactive months in last 12 months",
            "Number of contacts in last 12 months",
            "Credit card limit",
            "Total revolving balance",
            "Remaining available credit",
            "Transaction amount change from Q4 to Q1",
            "Total transaction amount in last 12 months",
            "Total transaction count in last 12 months",
            "Transaction count change from Q4 to Q1",
            "Percentage of credit card usage"
        ],
        "Type": [
            "Identifier", "Target", "Numerical", "Categorical", "Numerical",
            "Categorical", "Categorical", "Categorical", "Categorical",
            "Numerical", "Numerical", "Numerical",
            "Numerical", "Numerical", "Numerical",
            "Numerical", "Numerical", "Numerical",
            "Numerical", "Numerical", "Numerical"
        ]
    })
    
    with col1:
        st.markdown("### Dataset Overview")
        st.dataframe(dataset_overview, use_container_width=True)

    with col2:
        st.markdown("### Key Feature Highlight")

        st.markdown("""
        <div style="text-align: justify;">
        
        - **total_trans_ct** (🔻 Decrease Churn)  
        The most influential feature. Customers who frequently use their credit card for transactions are significantly less likely to churn.

        - **total_trans_amt** (🔺 Increase Churn)  
        Higher transaction amounts combined with lower frequency may indicate transactional, non-loyal behavior.

        - **total_revolving_bal** (🔻 Decrease Churn)  
        Active credit card usage is associated with lower churn risk.

        - **months_inactive_12_mon** (🔺 Increase Churn)  
        Inactive customers over the past year have a higher likelihood of churning.

        - **total_relationship_count** (🔻 Decrease Churn)  
        Customers with fewer products are more likely to churn.

        </div>
        """, unsafe_allow_html=True)



# =========================
# PAGE 2 — EDA (PLACEHOLDER)
# =========================
elif pages == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("======")
    
    #AGE GROUPING
    bins = [20, 40, 50, 60, 79]
    labels = ['20-39', '40-49', '50-59', '60-79']

    df['age_group'] = pd.cut(
        df['customer_age'],
        bins=bins,
        labels=labels,
        right=False
    )

    st.sidebar.header("Filter Data")
    
    # FILTERS
    selected_age = st.sidebar.multiselect(
    "Select Age Group",
    options=df['age_group'].dropna().unique(),
    default=df['age_group'].dropna().unique()
    )

    selected_edu = st.sidebar.multiselect(
        "Select Education Level",
        options=df['education_level'].unique(),
        default=df['education_level'].unique()
    )

    selected_gender = st.sidebar.multiselect(
        "Select Gender",
        options=df['gender'].unique(),
        default=df['gender'].unique()
    )

    selected_income = st.sidebar.multiselect(
        "Select Income Category",
        options=df['income_category'].unique(),
        default=df['income_category'].unique()
    )

    filtered_df = df[
        (df['gender'].isin(selected_gender)) &
        (df['income_category'].isin(selected_income)) &
        (df['age_group'].isin(selected_age)) &
        (df['education_level'].isin(selected_edu))
    ]

    
    # CHURN DISTRIBUTION & PROPORTION
    col1, col2 = st.columns([1,2])

    churn_counts = filtered_df['attrition_flag'].value_counts().reset_index()
    churn_counts.columns = ['attrition_flag', 'count']

    fig_bar = px.bar(
        churn_counts,
        x='attrition_flag',
        y='count',
        color='attrition_flag',
        text='count')

    fig_bar.update_layout(hovermode=False, showlegend=False,font=dict(size=16),width=400,height=460)
    fig_bar.update_yaxes(
    showticklabels=False,  
    title_text=None        
    )

    fig_bar.update_xaxes(
        tickfont=dict(size=16),title_text=None
    )

    fig_pie = px.pie(
        churn_counts,
        names='attrition_flag',
        values='count',
        hole=0.4
    )

    fig_pie.update_layout(hovermode=False, font=dict(size=16),legend=dict( font=dict(size=14)),width=400,height=460                                    )
    fig_pie.update_yaxes(
    showticklabels=False,  
    title_text=None        
    )

    fig_pie.update_xaxes(
        tickfont=dict(size=16)
    )  




    fig_age = px.histogram(
    filtered_df,
    y='age_group',
    color='attrition_flag',
    barmode='group',
    text_auto=True,
    category_orders={
        'age_group': labels
    }
    )
    
    fig_age.update_layout(hovermode=False,font=dict(size=18),
                         legend=dict( font=dict(size=14)),
                        height=400,margin=dict(t=20, b=20),showlegend=False)
    
    fig_age.update_xaxes(
    showticklabels=False,  
    title_text=None        
    )
    fig_age.update_yaxes(
        tickfont=dict(size=18),title_text=None
    )


    fig_edu = px.histogram(
    filtered_df,
    y='education_level',
    color='attrition_flag',
    barmode='group',
    text_auto=True,
    )
    fig_edu.update_layout(hovermode=False,font=dict(size=18),
                         legend=dict( font=dict(size=14)),
                        height=400,margin=dict(t=10, b=10),showlegend=False)
    fig_edu.update_traces(textangle=0)

    fig_edu.update_xaxes(
    showticklabels=False,  
    title_text=None        
    )

    fig_edu.update_yaxes(
        tickfont=dict(size=18),title_text=None
    )

    fig_income = px.histogram(
    filtered_df,
    y='income_category',
    color='attrition_flag',
    barmode='group',
    text_auto=True,
    )
    fig_income.update_layout(hovermode=False,font=dict(size=18),
                         legend=dict( font=dict(size=14)),
                        height=400,margin=dict(t=10, b=10),showlegend=False)

    fig_income.update_xaxes(
    showticklabels=False,  
    title_text=None        
    )

    fig_income.update_yaxes(
        tickfont=dict(size=18),title_text=None
    )

    fig_gender = px.histogram(
    filtered_df,
    y='gender',
    color='attrition_flag',
    barmode='group',
    text_auto=True    )
    fig_gender.update_layout(hovermode=False,font=dict(size=18),
                         legend=dict( font=dict(size=14)),
                        height=400,margin=dict(t=20, b=20),showlegend=False)

    fig_gender.update_xaxes(
    showticklabels=False,  
    title_text=None        
    )

    fig_gender.update_yaxes(
        tickfont=dict(size=18),title_text=None
    )

    
    def churn_rate_df(filtered_df, category):
        ct = pd.crosstab(
            filtered_df[category],
            filtered_df['attrition_flag'],
            normalize='index'
        ) * 100

        ct = ct.reset_index().melt(
            id_vars=category,
            var_name='attrition_flag',
            value_name='percentage'
        )
        ct['percentage'] = ct['percentage'].round(1)
        ct['percentage_label'] = ct['percentage'].astype(str) + '%'
        return ct


    age_rate = churn_rate_df(filtered_df, 'age_group')

    fig_age_rate = px.bar(
        age_rate,
        y='age_group',
        x='percentage',
        color='attrition_flag',
        barmode='group',
        text='percentage_label',
        category_orders={'age_group': labels}
    )

    fig_age_rate.update_layout(
        hovermode=False,font=dict(size=18),
                         legend=dict( font=dict(size=14)),
                        height=400,margin=dict(t=20, b=20),showlegend=False
    )
    fig_age_rate.update_xaxes(
    showticklabels=False,  
    title_text=None        
    )

    fig_age_rate.update_yaxes(
        tickfont=dict(size=18),title_text=None
    )
    edu_rate = churn_rate_df(filtered_df, 'education_level')

    fig_edu_rate = px.bar(
        edu_rate,
        y='education_level',
        x='percentage',
        color='attrition_flag',
        barmode='group',
        text='percentage_label' )

    fig_edu_rate.update_layout(
        uniformtext_minsize=14,
        hovermode=False,font=dict(size=18),
                         legend=dict( font=dict(size=14)),
                        height=400,margin=dict(t=10, b=10),showlegend=False
    )
    fig_edu_rate.update_xaxes(
    showticklabels=False,  
    title_text=None        
    )

    fig_edu_rate.update_yaxes(
        tickfont=dict(size=18),title_text=None
    )

    income_rate = churn_rate_df(filtered_df, 'income_category')

    fig_income_rate = px.bar(
        income_rate,
        y='income_category',
        x='percentage',
        color='attrition_flag',
        barmode='group',
        text='percentage_label',
    )

    fig_income_rate.update_layout(
        uniformtext_minsize=14,
       hovermode=False,font=dict(size=18),
                         legend=dict( font=dict(size=14)),
                        height=400,margin=dict(t=10, b=10),showlegend=False
    )

    fig_income_rate.update_xaxes(
    showticklabels=False,  
    title_text=None        
    )

    fig_income_rate.update_yaxes(
        tickfont=dict(size=18),title_text=None
    )
    
    gender_rate = churn_rate_df(filtered_df, 'gender')

    fig_gender_rate = px.bar(
        gender_rate,
        y='gender',
        x='percentage',
        color='attrition_flag',
        barmode='group',
        text='percentage_label'    )

    fig_gender_rate.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        hovermode=False,font=dict(size=18),
                         legend=dict( font=dict(size=14)),
                        height=400,margin=dict(t=20, b=20),showlegend=False
    )
    fig_gender_rate.update_xaxes(
    showticklabels=False,  
    title_text=None        
    )

    fig_gender_rate.update_yaxes(
        tickfont=dict(size=18),title_text=None
    )

    with col1:
        st.markdown("### Churn Distribution Count")
        with st.container(border=True):
            st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("### Churn Distribution Proportion")
        with st.container(border=True):
            st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.markdown("### Customer Demographics Overview")
        tab1, tab2, tab3,tab4 = st.tabs(["Churn vs Age Group", "Churn vs Education Level", "Churn vs Income Category", "Churn vs Gender"])
        with tab1:
            with st.container(border=True):
                st.plotly_chart(fig_age, use_container_width=True)
        with tab2:
            with st.container(border=True):
                st.plotly_chart(fig_edu, use_container_width=True)
        with tab3:
            with st.container(border=True):
                 st.plotly_chart(fig_income, use_container_width=True)
        with tab4:
            with st.container(border=True):
                 st.plotly_chart(fig_gender, use_container_width=False)
        
        st.markdown("### Churn Rate by Demographics")
        tab1, tab2, tab3,tab4 = st.tabs(["Churn Rate by Age Group", "Churn Rate by Education Level", "Churn Rate by Income Category", "Churn Rate by Gender"])
        with tab1:
            with st.container(border=True):
                st.plotly_chart(fig_age_rate, use_container_width=False)
        with tab2:
            with st.container(border=True):
                st.plotly_chart(fig_edu_rate, use_container_width=False)
        with tab3:
            with st.container(border=True):
                 st.plotly_chart(fig_income_rate, use_container_width=False)
        with tab4:
            with st.container(border=True):
                 st.plotly_chart(fig_gender_rate, use_container_width=False)
    
    st.subheader("Key Insights from Demographic Analysis")

    st.markdown(
    """
    <div style="text-align: justify;">
        <ul>
            <li><b>Middle-aged customers (40-60 years old) are more likely to churn:</b> They form the majority of the customer base, especially 40-49 and 50-59 age groups, with a churn rate of <b>16.9%</b>. Retention efforts should focus on this core segment.</li>
            <li><b>Higher education correlates with higher churn:</b> Customers with Doctorate (<b>21.1%</b>) and Post-Graduate (<b>17.8%</b>) degrees churn more than the Graduate group (<b>15.6%</b>), even though they are fewer in number. This indicates that highly educated customers may require more personalized engagement.</li>
            <li><b>Income extremes have higher churn:</b> Customers earning above USD 120K, a <b>minority group</b>, have a churn rate of <b>17.3%</b>, while those earning below USD 40K, the <b>majority group</b>, have a churn rate of <b>17.2%</b>. Middle-income customers churn less, showing that both high and low extremes are more likely to leave.</li>
            <li><b>Gender differences exist in churn:</b> Female customers, who are the majority, have a higher churn rate (<b>17.4%</b>) than male customers, suggesting different engagement or satisfaction patterns between genders.</li>
            <li><b>Customer count does not equal churn risk:</b> Smaller segments may exhibit disproportionately high churn rates, emphasizing the importance of segment-level analysis rather than absolute counts.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
    )




# =========================
# PAGE 3 — MODELING (PLACEHOLDER)
# =========================
elif pages == "Churn Prediction Model":
    st.title("Churn Prediction Model")
    st.info("TBA.")
    st.image("assets/miku question mark", width=200)


# =========================
# PAGE 4 — EVALUATION (PLACEHOLDER)
# =========================
elif pages == "Model Evaluation":
    st.title("Model Evaluation")
    st.info("TBA.")
    st.image("assets/miku question mark", width=200)


# =========================
# PAGE 5 — INSIGHT
# =========================
elif pages == "Insight & Recommendation":
    st.title("Insight & Recommendation")
    st.info("TBA.")
    st.image("assets/miku question mark", width=200)

