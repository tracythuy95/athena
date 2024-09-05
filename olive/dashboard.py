import altair as alt
import streamlit as st
import pandas as pd
from datetime import datetime

# Loading in the data sources ------------------------------------------------
df = pd.read_csv("data/processed/clinical_trials_lead.csv")
df_stock = pd.read_csv("data/raw/stock.csv")
df_all_forecasts = pd.read_csv("data/output/model_stock.csv")
df_abstracts = pd.read_csv("data/processed/aacr_keywords.csv")
df_abstract_hits = pd.read_csv("data/output/top_keywords_by_company.csv")
df_model = pd.read_csv("data/output/model_results.csv")

# Setting up tabs and page settinngs -----------------------------------------
st.set_page_config(layout="wide")

tab1, tab2, tab3, tab4 = st.tabs(["Explore: Clinical Trials", "Modeling: Clinical Trials", "Explore: Abstracts (AACR)", "Explore: Financial"])

# Helper Functions -----------------------------------------------------------
def make_clickable(text):
    return f'<a href="https://clinicaltrials.gov/study/{text}" target="_blank">{text}</a>'


# # List of quantitative data items
# item_list = [
#     col for col in df.columns if df[col].dtype in ['float64', 'int64']]

# List of Sponsors

# Minimal Derivation on data sources -----------------------------------------
# Create the column of YYYY 
# Convert the date strings to datetime objects
df['Start Date'] = pd.to_datetime(df['Start Date'])
df['Primary Completion Date'] = pd.to_datetime(df['Primary Completion Date'])

df['YYYY'] = df['Start Date'].apply(lambda x: x.year)
min_year = df['YYYY'].min().item()
max_year = df['YYYY'].max().item()


# Sidebar --------------------------------------------------------------------
st.sidebar.title("Dashboard of NGS Cancer Diagnostic Space")
st.sidebar.markdown('###')
st.sidebar.markdown("### *Settings*")
## Study Start Date ----------------------------------------------------------
start_year, end_year = st.sidebar.slider(
    "Study Start Date",
    min_value=min_year, max_value=max_year,
    value=(min_year, max_year))
## Sponsors ------------------------------------------------------------------
st.sidebar.markdown('###')
sponsor_list = list(df['Lead Sponsor'].unique())
origins = st.sidebar.multiselect('Lead Sponsor',
                                 sponsor_list,
                                 default=sponsor_list)
## Indication ----------------------------------------------------------------
add_selectbox = st.sidebar.selectbox(
    "Indication of Interest",
    ("All", "Multi-cancer", "Breast", "Lung", "Colorectal", "Prostate")
)
# Filtering based on Side Bar ------------------------------------------------
## Filter by Date ------------------------------------------------------------
df_rng = df[(df['YYYY'] >= start_year) & (df['YYYY'] <= end_year)]
## Filter by Sponsor ---------------------------------------------------------
source = df_rng[df_rng['Lead Sponsor'].isin(origins)]
## Filter by Indication ------------------------------------------------------
if add_selectbox == "Multi-cancer":
    source = source[(source['is_multicancer'] == True)]
elif add_selectbox == "Breast": 
    source = source[(source['is_breast'] == True) & (source['is_multicancer'] == False)]
elif add_selectbox == "Lung":
    source = source[(source['is_lung'] == True) & (source['is_multicancer'] == False)]
elif add_selectbox == "Colorectal":
    source = source[(source['is_colorectal'] == True) & (source['is_multicancer'] == False)]
elif add_selectbox == "Prostate":
    source = source[(source['is_prostate'] == True) & (source['is_multicancer'] == False)]
elif add_selectbox == "All":
    source = source

# Create plots --------------------------------------------------------------
## Clinical Study Plots -----------------------------------------------------
base = alt.Chart(source).properties(height=600)
today = pd.to_datetime(datetime.today().date())

# Create a vertical dashed line for today's date
today_line = alt.Chart(pd.DataFrame({'Date': [today]})).mark_rule(
    color='black',
    strokeDash=[5, 5]
).encode(
    x='Date:T'
)

gantt_chart = base.mark_bar().encode(
    x='Start Date:T',
    x2='Primary Completion Date:T',
    y=alt.Y('NCT ID:N', title='NCT ID'),
    color='Lead Sponsor:N',
    tooltip=['NCT ID', 'Official Title', 'Lead Sponsor',
             'Start Date', 'Primary Completion Date', 'Enrollment Info Count',
             'Number of Locations']
).properties(
    title='Clinical Trial Timeline'
)

text_labels = gantt_chart.mark_text(
    align='left',
    baseline='middle',
    dx = 10  # Adjusts the position of the text
).encode(
    text='Overall Status:N',
    color=alt.value('black')
)

point = base.mark_circle(size=100).encode(
    x=alt.X('Enrollment Info Count:Q', scale=alt.Scale(type='log')),
    y=alt.Y('Number of Locations:Q', scale=alt.Scale(type='log')),
    color=alt.Color('Lead Sponsor:N', title='',
                    legend=alt.Legend(orient='top-left')),
    tooltip=['NCT ID', 'Official Title', 'Lead Sponsor',
             'Start Date', 'Primary Completion Date', 'Enrollment Info Count',
             'Number of Locations', 'Overall Status']
).properties(
    title='Enrollment Count vs Number of Locations'
)

color_scale = alt.Scale(
    domain=["RECRUITING", "ACTIVE_NOT_RECRUITING",
            "COMPLETED", "TERMINATED"],
    range=['#3eb59f', '#224e67', "#2a2227", "#e8433f"]
)

bar_plot_study_status = base.mark_bar().encode(
    x='Lead Sponsor:N', 
    y='count(Overall Status)',
    color=alt.Color('Overall Status:N', title='', scale=color_scale,
                    legend=alt.Legend(orient='top-left'))
).properties(
    title='Count of Study Status by Sponsor'
)

### Model Plot of Clinical Trials ----------------------------------
df_model['Start Date'] = pd.to_datetime(df_model['Start Date'])
df_model['Primary Completion Date'] = pd.to_datetime(df_model['Primary Completion Date'])
df_model['Planned Duration'] = (df_model['Primary Completion Date'] - df_model['Start Date']).dt.days

base2 = alt.Chart(df_model).properties(height=600, width = 800)

point_model = base2.mark_circle(size=100).encode(
    x=alt.X('Planned Duration:Q'),
    y=alt.Y('Predicted Duration:Q'),
    color=alt.Color('Lead Sponsor:N', title='',
                    legend=alt.Legend(orient='bottom-left')),
    tooltip=['NCT ID', 'Official Title', 'Lead Sponsor',
             'Start Date', 'Primary Completion Date', 'Enrollment Info Count',
             'Number of Locations', 'Planned Duration', 'Predicted Duration']
).properties(
    title='Planned Duration vs Predicted Duration in Days'
)

## Abstract Data ---------------------------------------------------
base3 = alt.Chart(df_abstract_hits).properties(height=600)
bar_plot_abstracts = base3.mark_bar().encode(
    x='keyword:N', 
    y='count:Q',
    color=alt.Color('Company:N', title='',
                    legend=alt.Legend(orient='top-left'))
).properties(
    title='Top 10 Keywords by Company'
)

## Stock Plots -----------------------------------------------------
base = alt.Chart(df_stock).encode(
    x='Date:T',
    y=alt.Y('Close:Q', title='Stock Price'),
    color=alt.Color('Stock:N', title='',
                    legend=alt.Legend(orient='top-left'))
)

actual_line = base.mark_line(color='blue').encode(
    y='Close:Q',
    tooltip=['Date:T', 'Close:Q']
)

stock_chart = actual_line

# Create an Altair plot
chart_forecast = alt.Chart(df_all_forecasts).mark_line().encode(
    x='ds:T',
    y='yhat:Q',
    color='Stock:N',
    tooltip=['ds:T', 'yhat:Q', 'Stock:N']
).properties(
    title='Stock Price Forecast'
).interactive()

# Layout for Each Tab (Content) ------------------------------------
with tab1:
    tab1.subheader("Summary of Status")
    left_column, right_column = st.columns(2)
    left_column.altair_chart(gantt_chart + text_labels + today_line, use_container_width=True)
    right_column.altair_chart(point, use_container_width=True)
    left_column.altair_chart(bar_plot_study_status, use_container_width=True)
    tab1.subheader("Detail")
    source_display = source[["NCT ID", "Official Title" , "Lead Sponsor", "Study Type", "Biospecimen Description", "Brief Summary", "Conditions", "Enrollment Info Count", "Overall Status"]]
    source_hyperlink = source_display.copy()
    source_hyperlink['NCT ID'] = source_display.apply(lambda x: make_clickable(x['NCT ID']), axis=1)
    st.markdown(source_hyperlink.to_html(escape=False), unsafe_allow_html=True)
    
with tab2:
    tab2.subheader("Data for Model that was randomly split into a Training and Holdout set")
    source_tab2 = df[(df['Overall Status'] == "COMPLETED") | (df['Overall Status'] == "ACTIVE_NOT_RECRUITING")]
    source_tab2 = source_tab2[['NCT ID', 'Official Title', 'Lead Sponsor', 'Enrollment Info Count', 'Number of Locations', "Start Date", "Primary Completion Date"]]
    st.write(source_tab2)
    tab2.subheader("Predicted Data from the Model")
    predicted_model = df_model.copy()
    predicted_model = predicted_model[['NCT ID', 'Official Title', 'Lead Sponsor', 'Enrollment Info Count', 'Number of Locations', "Start Date", "Primary Completion Date", "Predicted Primary Completion Date", "Predicted Duration"]]
    st.write(predicted_model)
    tab2.subheader("Comparison of Planned Duration vs Predicted Duration")
    point_model
    
with tab3:
    tab3.subheader("AACR 2024 Abstracts (6000 to 52)")
    abstract_display = df_abstracts[['Company', 'Title', 'text']]
    st.write(abstract_display)
    tab3.subheader("Used NLP to identify the top keywords")
    bar_plot_abstracts
    
with tab4:
    tab4.subheader("Stock")
    stock_chart = stock_chart.properties(
        width=1200,
        height=800
    )
    stock_chart
    

    
