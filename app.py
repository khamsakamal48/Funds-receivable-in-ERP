import pandas as pd
import streamlit as st
from datetime import datetime
from datetime import date

# Default page configuration
# ----------------------------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title='Funds Receivable Summary',
    page_icon=':bar_chart:',
    layout="wide")

# Hide Streamlit menus
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Functions
def load_to_dataframe(file, file_type):
    if file_type == 'text/csv':
        return pd.read_csv(file)

    else:
        return pd.read_excel(file)


# Page Title
# ----------------------------------------------------------------------------------------------------------------------
st.title('Funds Receivable in ERP')
st.divider()

# Adding Sidebar for Input files upload
# ----------------------------------------------------------------------------------------------------------------------
with st.sidebar:
    st.header('Date Selection')
    st.caption('Document Date')

    # Get the current date
    current_date = datetime.now()

    target_year = [current_date.year if current_date.month >= 4 else current_date.year - 1][0]

    # Start Date
    st.subheader('Start Date')
    start_date = st.date_input('Start Date', format='DD-MM-YYYY', label_visibility='collapsed',
                               value=date(target_year, 4, 1))

    # End Date
    st.subheader('End Date')
    end_date = st.date_input('End Date', format='DD-MM-YYYY', label_visibility='collapsed')

    # Use datetime objects for the date range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    st.divider()

    st.header('Upload Files:')

    # Upload Transaction file
    st.subheader('1. Transaction file')
    transaction = st.file_uploader('Upload the Transaction file from ERP', type=['csv', 'xlsx'],
                                   label_visibility='collapsed')

    # Load to Dataframe

    # Upload Transaction file
    st.subheader('2. WBS file')
    wbs = st.file_uploader('Upload the WBS file from ERP', type=['csv', 'xlsx'], label_visibility='collapsed')


# Streamline the data
# ----------------------------------------------------------------------------------------------------------------------
# Convert Document date to datetime format
if transaction is not None:
    # Load to a Pandas Dataframe
    transaction = load_to_dataframe(transaction, transaction.type)

    transaction['Document Date'] = pd.to_datetime(transaction['Document Date'], format='%Y-%m-%d %H:%M:%S',
                                                  errors='coerce')

    # Replace invalid dates with values from the 'Posting Date' column
    transaction['Document Date'] = transaction['Document Date'].fillna(transaction['Posting Date'])

    # Drop blank columns
    transaction = transaction.dropna(subset=['Document type']).reset_index(drop=True)

if wbs is not None:
    wbs = load_to_dataframe(wbs, wbs.type)

# Perform WBS mapping
transaction = transaction.merge(wbs[['WBS ', 'WBS Details ']].drop_duplicates(), how='left',
                                left_on='Project definition', right_on='WBS ')

transaction = transaction.merge(wbs[['Sub WBS ', 'SUB WBS Details']].drop_duplicates(), how='left',
                                left_on='Object', right_on='Sub WBS ')

# Indian Donations
indian_donations = transaction[
    (transaction['Document type'] == 'DR') &
    (transaction['Document Date'].between(start_date, end_date)) &
    ~(transaction['Cost Element'].isin([550510, 550511]))
]

indian_donations_total = abs(indian_donations['Val/COArea Crcy'].sum())

# Non-Indian & Non-HF Donations
non_indian_hf_donations = transaction[
    (transaction['Document type'] == 'DR') &
    (transaction['Document Date'].between(start_date, end_date)) &
    (transaction['Cost Element'] == 550510)
]

non_indian_hf_donations_total = abs(non_indian_hf_donations['Val/COArea Crcy'].sum())

# Total HF Raised
hf_donations_raised = transaction[
    (transaction['Document type'] == 'DR') &
    (transaction['Document Date'].between(start_date, end_date)) &
    (transaction['Cost Element'] == 550511)
]

hf_donations_raised_total = abs(hf_donations_raised['Val/COArea Crcy'].sum())

# HF Raised - Allocated
hf_donations_allocated = transaction[
    (transaction['Document type'] == 'SB') &
    (transaction['Document Date'].between(start_date, end_date)) &
    (transaction['Cost Element'] == 550511) &
    ~(transaction['Object'].str.contains('IITBHF'))
]

hf_donations_allocated_total = abs(hf_donations_allocated['Val/COArea Crcy'].sum())

# HF Raised - Not Allocated
hf_donations_not_allocated_total = abs(hf_donations_raised_total - hf_donations_allocated_total)

# Final Data
final_data = pd.DataFrame()

if transaction is not None and wbs is not None:
     final_data = transaction[
         (transaction['Document Date'].between(start_date, end_date))
     ]


total_raised =  transaction[
    (transaction['Document type'] == 'DR') &
    (transaction['Document Date'].between(start_date, end_date))
]

total_raised = abs(total_raised['Val/COArea Crcy'].sum())

# Present the data
# ----------------------------------------------------------------------------------------------------------------------
if final_data.shape[0] == 0:
    st.warning('Upload the data to proceed')

else:
    # Donation Received Metrics
    st.header('Summary')
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9, vertical_alignment='center')

    col1.metric('India', f'₹ {round(indian_donations_total / 10000000, 2)} Cr.')

    col2.header('+')

    col3.metric('US `(Allocated)`', f'₹ {round(hf_donations_allocated_total / 10000000, 2)} Cr.')

    col4.header('+')

    col5.metric('US `(Not Allocated)`', f'₹ {round(hf_donations_not_allocated_total / 10000000, 2)} Cr.')

    col6.header('+')

    col7.metric('ROW', f'₹ {round(non_indian_hf_donations_total / 10000000, 2)} Cr.')

    col8.header('=')

    col9.metric('Total Received', f'₹ {round(total_raised / 10000000, 2)} Cr.')

    st.subheader('')
    st.dataframe(final_data, hide_index=True)