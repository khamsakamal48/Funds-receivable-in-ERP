import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from datetime import date
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from collections import defaultdict

# Load Environment variables
# ----------------------------------------------------------------------------------------------------------------------
# Load the .env file
load_dotenv()

# Fetch environment variables
db_address = os.getenv('DB_ADDRESS')
db_port = os.getenv('DB_PORT')
db_username = os.getenv('DB_USERNAME')
db_password = quote_plus(os.getenv('DB_PASSWORD'))
db_name = os.getenv('DB')

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
# ----------------------------------------------------------------------------------------------------------------------
# Function to load file into DataFrame based on file type
def load_to_dataframe(file, file_type):
    """
    Load the uploaded file into a Pandas DataFrame.

    :param file: The uploaded file object.
    :param file_type: The type of the uploaded file ('text/csv' or xlsx).
    :return: pd.DataFrame: A DataFrame containing the data from the uploaded file.
    """
    if file_type == 'text/csv':
        return pd.read_csv(file)

    else:
        return pd.read_excel(file)


# Function to create SQLAlchemy engine:
#   - This function establishes a connection to the PostgreSQL database using SQLAlchemy.
def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database using SQLAlchemy.
    - The `postgresql+psycopg2` dialect indicates that we're using PostgreSQL as the database backend.
    - The `{DB_USER}:{DB_PASS}` part specifies the username and password to use for connections.
    - The `@{DB_IP}:{DB_PORT}` part specifies the hostname and port number to use when connecting to the database.
    - The `/`${DB}` part specifies the database name or schema to connect to.

    :return: engine.connect(): An active connection object to the PostgreSQL database.
    """
    engine = create_engine(f'postgresql://{db_username}:{db_password}@{db_address}:{db_port}/{db_name}')

    # Establish a connection to the database:
    #   - This returns an active connection object, which can be used to execute queries and interact with the database.
    return engine.connect()


# Cache the function that fetches data from the SQL table:
#   - This is a Streamlit caching mechanism that stores the result of this function for 1 day.
@st.cache_data(ttl='1d')
def fetch_data_from_sql(query):
    """
    Fetches data from a SQL table and returns it as a pandas DataFrame.

    :param query: (str) The SQL query to execute on the database.
    :return: pd.DataFrame: A pandas DataFrame containing the results of the SQL query.
    """

    # Establish a connection to the PostgreSQL database using get_db_connection():
    conn = get_db_connection()

    # Execute the SQL query on the database and store the result in a pandas DataFrame:
    df = pd.read_sql(query, conn)

    # Close the connection to the database to free up resources:
    conn.close()

    # Return the pandas DataFrame containing the results of the SQL query:
    return df


# Function to update SQL table:
#   - This function updates an existing SQL table by adding or replacing data.
def update_sql_table(data, table_name, schema, replace=True):
    """
    Updates an existing SQL table by adding or replacing data.

    :param data: (pd.DataFrame) The data to be updated in the SQL table.
    :param table_name: (str) The name of the SQL table to be updated.
    :param schema: (str) The schema of the SQL table.
    :param replace: (bool, optional) Whether to replace existing data. Defaults to True.
    :return: None
    """

    # Establish a connection to the PostgreSQL database using get_db_connection():
    conn = get_db_connection()

    # Update the SQL table by adding or replacing data.
    # - If `replace` is True, the existing data in the table will be replaced with new data.
    # - If `replace` is False (default), new data will be appended to the existing data in the table.
    data.to_sql(table_name, conn, if_exists='replace' if replace else 'append', index=False, schema=schema)

    # Close the connection to the database to free up resources:
    conn.close()


def change_column_names(column_list, std=True):
    if std:
        return [col.strip().title().replace('_', ' ') for col in column_list]

    else:
        return [col.strip().lower().replace(' ', '_') for col in column_list]


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(lineterminator='\r\n', index=False, quoting=1).encode('utf-8')


def get_amount_in_cr(amount):
    # if amount > 0:
    #     return f'₹ {round(amount / 10000000, 2)} Cr.'
    #
    # else:
    #     return 0
    return f'₹ {round(amount / 10000000, 2)} Cr.'


# Find donations that nullifies each other
def find_nullifying_groups(df):
    """
    Efficiently find and return rows where values in 'Value in Obj. Crcy' nullify each other.

    Args:
    df (pd.DataFrame): The input DataFrame with columns 'Ref. document number' and 'Value in Obj. Crcy'.

    Returns:
    pd.DataFrame: A DataFrame containing the 'Ref. document number' and 'Value in Obj. Crcy'
                  of nullifying groups.
    """
    # Sort values by absolute amount in descending order (higher values first)
    df = df.reindex(df['Val/COArea Crcy'].abs().sort_values(ascending=False).index)

    # Dictionary to store unmatched values with their index positions
    unmatched = defaultdict(list)
    nullifying_indices = set()

    for idx, row in df.iterrows():
        value = row['Val/COArea Crcy']
        opposite_value = -value

        if opposite_value in unmatched:
            # Match with the first occurrence of opposite value
            match_idx = unmatched[opposite_value].pop(0)
            nullifying_indices.update([idx, match_idx])

            # Remove the key if no more unmatched values exist
            if not unmatched[opposite_value]:
                del unmatched[opposite_value]
        else:
            # Store index of the current unmatched value
            unmatched[value].append(idx)

    # Filter DataFrame to include only nullifying rows
    return df.loc[list(nullifying_indices)]


# Get Fund Utilisation Report for a WBS code
def get_fur(wbs_code):

    # Create an empty dataframe
    wbs_breakup = pd.DataFrame()

    # Check for SB entries
    df = transaction[
        (transaction['Project definition'] == wbs_code) &
        (transaction['Document type'] == 'SB')
    ]

    # Step 2
    if df.shape[0] == 0:
        wbs_breakup = pd.concat([
            wbs_breakup,
            transaction[
                transaction['Project definition'] == wbs_code
            ]
        ], ignore_index=True)

        return wbs_breakup

    else:
        # Sort
        df = df.sort_values(by=['Fiscal Year', 'Val/COArea Crcy'], ascending=[True, False]).copy()

        # Step 3
        df_nullified = find_nullifying_groups(df)

        # Step 4
        data_non_null = df[~df['Ref. document number'].isin(df_nullified['Ref. document number'])]

        # Step 5.1 and 5.2
        data_sb = transaction[transaction['Ref. document number'].isin(data_non_null['Ref. document number'])]

        # Step 5.3
        data_proj = data_sb[data_sb['Project definition'] != wbs_code]
        # data_proj = data_sb.copy()

        # Concatenate the final data
        project_breakup = pd.concat([wbs_breakup, data_proj], ignore_index=True)

        return project_breakup

def get_flow(df, flow):

    df = df.copy()

    # Change Year to String
    df['Fiscal Year'] = df['Fiscal Year'].astype(str)

    if flow == 'inflow':
        # Prepare data of the amount which flowed 'From' other WBS Codes
        df = df[df['Val/COArea Crcy'] < 0].copy()

    elif flow == 'outflow':
        # Prepare data of the amount which flowed 'to' other WBS Code
        df = df[df['Val/COArea Crcy'] > 0].copy()

    # Change the amount to positive
    df['Val/COArea Crcy'] = df['Val/COArea Crcy'].abs()

    df = df.pivot_table(
        index=['Ref. document number', 'Fiscal Year', 'Project definition', 'Object'],
        values=['Val/COArea Crcy'],
        aggfunc='sum'
    )

    df = df.reset_index(drop=False).copy()

    df['Ref. document number'] = df['Ref. document number'].astype(str)

    return df


def get_admin_expense(wbs_code, flow):
    # Records with Cost Element = 510225
    df = transaction[
        (
            (transaction['Cost Element'] == 510225) |
            (transaction['Cost Element'] == '510225')
        ) &
        (transaction['Project definition'] == wbs_code) &
        (transaction['Document type'] == 'SB')
    ]

    df = get_flow(df, flow)

    return df


def get_interests(wbs_code, flow):
    # Records with offsetting account = 811002
    df = transaction[
        (
            (transaction['Offsetting acct no.'] == '811002') |
            (transaction['Offsetting acct no.'] == 811002)
        ) &
        (transaction['Project definition'] == wbs_code) &
        (transaction['Document type'] == 'SB')
        ]

    df = get_flow(df, flow)

    return df


# Page Title
# ----------------------------------------------------------------------------------------------------------------------
st.title('Funds Receivable in ERP')
st.divider()

# Adding Sidebar for Input files upload
# ----------------------------------------------------------------------------------------------------------------------
with st.sidebar:
    st.header('Date Selection', help='The date is filtered on the basis of Document Date')

    # Get the current date
    current_date = datetime.now()

    target_year = [current_date.year if current_date.month >= 4 else current_date.year - 1][0]

    col1, col2 = st.columns(2)

    with col1:
        # Start Date
        st.subheader('Start Date')
        # Assuming fiscal year starts April 1st
        start_date = st.date_input('Start Date', format='DD-MM-YYYY', label_visibility='collapsed',
                                   value=date(target_year, 4, 1))

    with col2:
        # End Date
        st.subheader('End Date')
        # Assuming fiscal year ends March 31st of the next year
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

    # Upload WBS file
    st.subheader('2. WBS file')
    wbs = st.file_uploader('Upload the WBS file from ERP', type=['csv', 'xlsx'], label_visibility='collapsed')

    # Upload Project Master
    st.subheader('3. Project Master file', help="It should contain columns: 'Project Name' and 'Category'")
    project_directory_new = st.file_uploader('Upload the Project Master created manually', type=['csv', 'xlsx'],
                                      label_visibility='collapsed')

    # Get existing master
    query_project_dir_old = '''
            SELECT
                *
            FROM
                erp_data.funds_received.project_directory
            '''

    # Check if there's new project master
    if project_directory_new is not None:
        project_directory_new = load_to_dataframe(project_directory_new, project_directory_new.type)

        # Ignore missing values
        project_directory_new = project_directory_new.dropna().copy()

        # Add Project IDs
        project_directory_new['project_id'] = project_directory_new['project_name'].str.strip().str.lower().str.replace(
            ' ', '_')

        # Change case and replace space with '_' in column names
        project_directory_new.columns = [col.strip().lower().replace(' ', '_') for col in project_directory_new.columns]

        # Change column order
        project_directory_new = project_directory_new[['project_id', 'project_name', 'category']]

        # Clear Cache data
        fetch_data_from_sql.clear(query_project_dir_old)

    else:
        # Create an empty dataframe
        project_directory_new = pd.DataFrame()

    project_directory_old = fetch_data_from_sql(query_project_dir_old)

    # Merge old (existing) and new data
    project_directory = pd.concat([project_directory_new, project_directory_old])

    # Consider the latest classification
    project_directory = project_directory.drop_duplicates(subset=['project_id'], keep='first', ignore_index=True).copy()

    # Upload to SQL table if any new data was provided
    if project_directory_new.shape[0] > 0:
        update_sql_table(project_directory, 'project_directory', schema='funds_received', replace=True)

    # Standardise column names
    project_directory.columns = change_column_names(project_directory.columns)

    st.divider()

    # Clear Cache
    if st.button("Clear Cache"):
        # Clear values from *all* all in-memory and on-disk data caches:
        # i.e. clear values from both square and cube
        st.cache_data.clear()

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

    update_sql_table(transaction, 'transactions', schema='funds_received', replace=True)

else:
    query_transaction = '''
    SELECT
        *
    FROM
        erp_data.funds_received.transactions
    '''

    transaction = fetch_data_from_sql(query_transaction)

if wbs is not None:
    wbs = load_to_dataframe(wbs, wbs.type)
    update_sql_table(wbs, 'wbs', schema='funds_received', replace=True)

else:
    query_transaction = '''
        SELECT
            *
        FROM
            erp_data.funds_received.wbs
        '''

    wbs = fetch_data_from_sql(query_transaction)

# Remove extra spaces from column names
transaction_col = transaction.columns
transaction_col = [col.strip() for col in transaction_col]
transaction.columns = transaction_col

wbs_col = wbs.columns
wbs_col = [col.strip() for col in wbs_col]
wbs.columns = wbs_col

if transaction.shape[0] > 0 and wbs.shape[0] > 0 and project_directory.shape[0] > 0:
    # Perform WBS mapping
    transaction = transaction.merge(wbs[['WBS', 'WBS Details']].drop_duplicates(), how='left',
                                    left_on='Project definition', right_on='WBS')

    transaction = transaction.merge(wbs[['Sub WBS', 'SUB WBS Details']].drop_duplicates(), how='left',
                                    left_on='Object', right_on='Sub WBS')

    transaction['Project Id'] = transaction['SUB WBS Details'].apply(
        lambda x: str(x).strip().lower().replace(' ', '_'))

    # Assign Category
    transaction = transaction.merge(project_directory[['Project Id', 'Category']], how='left', on='Project Id').copy()

# Final Data
final_data = transaction[(transaction['Document Date'].between(start_date, end_date))]

# if transaction is None:
#     final_data = final_data.merge(project_directory[['Project Id', 'Category']], how='left', on='Project Id')
#
# if 'Category_x' in final_data.columns and not 'Category' in final_data.columns:
#     final_data = final_data.rename(columns={'Category_x': 'Category'})


# Present the data
# ----------------------------------------------------------------------------------------------------------------------
if final_data.shape[0] == 0:
    st.warning('Upload the data to proceed')

else:
    # Donation Received Metrics
    st.header('Summary')

    # Calculate metrics
    # Indian Donations
    indian_donations = final_data[
        (final_data['Document type'] == 'DR') &
        ~(final_data['Cost Element'].isin([550510, 550511]))
        ]

    indian_donations_total = abs(indian_donations['Val/COArea Crcy'].sum())

    # Non-Indian & Non-HF Donations
    non_indian_hf_donations = final_data[
        (final_data['Document type'] == 'DR') &
        (final_data['Cost Element'] == 550510)
        ]

    non_indian_hf_donations_total = abs(non_indian_hf_donations['Val/COArea Crcy'].sum())

    # Total HF Raised
    hf_donations_raised = final_data[
        (final_data['Document type'] == 'DR') &
        (final_data['Cost Element'] == 550511)
        ]

    hf_donations_raised_total = abs(hf_donations_raised['Val/COArea Crcy'].sum())

    # HF Raised - Allocated
    hf_donations_allocated = final_data[
        (final_data['Document type'] == 'SB') &
        (final_data['Cost Element'] == 550511) &
        ~(final_data['Object'].str.contains('IITBHF'))
        ]

    hf_donations_allocated_total = abs(hf_donations_allocated['Val/COArea Crcy'].sum())

    # HF Raised - Not Allocated
    hf_donations_not_allocated_total = abs(hf_donations_raised_total - hf_donations_allocated_total)

    total_raised = final_data[
        (final_data['Document type'] == 'DR')
        ]

    total_raised = abs(total_raised['Val/COArea Crcy'].sum())

    match int(hf_donations_not_allocated_total):
        case 0:
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7, vertical_alignment='center')

            col1.metric('India', get_amount_in_cr(indian_donations_total))
            col2.header('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; +')
            col3.metric('United States', get_amount_in_cr(hf_donations_allocated_total))
            col4.header('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; +')
            col5.metric('Rest of the World', get_amount_in_cr(non_indian_hf_donations_total))
            col6.header('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; =')
            col7.metric('Total Received', get_amount_in_cr(total_raised))

        case _:
            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9, vertical_alignment='center')

            col1.metric('India', get_amount_in_cr(indian_donations_total))
            col2.header('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; +')
            col3.metric('United States `(Allocated)`', get_amount_in_cr(hf_donations_allocated_total))
            col4.header('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; +')
            col5.metric('United States `(Not Allocated)`', get_amount_in_cr(hf_donations_not_allocated_total))
            col6.header('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; +')
            col7.metric('Rest of the World', get_amount_in_cr(non_indian_hf_donations_total))
            col8.header('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; =')
            col9.metric('Total Received', get_amount_in_cr(total_raised))

    # Cause Classification
    st.write('')
    st.subheader('Project Classification')

    col10, col11 = st.columns(2)

    # Extract only DR Data
    dr_data = final_data[final_data['Document type'] == 'DR'].copy()

    # Extract LP Projects
    lp_data = dr_data[
        (dr_data['Project definition'].str.contains('LP')) |
        (dr_data['Object'].str.contains('LP'))
        ]

    # Extract HF Data
    hf_data = dr_data[
        (dr_data['Project definition'].str.contains('HF')) |
        (dr_data['Object'].str.contains('HF'))
        ]

    # Extract CSR Projects
    csr_data = dr_data[
        (dr_data['Project definition'].str.contains('CSRP')) |
        (dr_data['Object'].str.contains('CSRP'))
        ]

    # Cause-wise donation
    donations_by_cause = dr_data[
        # Ignore Legacy Projects
        (
                ~(dr_data['Project definition'].isin(lp_data['Project definition'])) |
                ~(dr_data['Object'].isin(lp_data['Object']))
        ) &

        # Ignore HF Projects
        (
                ~(dr_data['Project definition'].isin(hf_data['Project definition'])) |
                ~(dr_data['Object'].isin(hf_data['Object']))
        ) &

        # Ignore CSR Projects
        (
                ~(dr_data['Project definition'].isin(csr_data['Project definition'])) |
                ~(dr_data['Object'].isin(csr_data['Object']))
        )
    ].groupby('Category', dropna=False)['Val/COArea Crcy'].sum().abs()
    donations_by_cause = donations_by_cause.reset_index(drop=False)

    donations_by_cause.columns = ['Cause', 'Amount Received']

    with col10:
        st.dataframe(donations_by_cause, use_container_width=True, hide_index=True)

    with col11:

        st.write('##### Missing categories')
        with st.expander('## List of Donations with missing categories'):

            # Projects with missing categories
            merged_donations = pd.concat([
                indian_donations, hf_donations_raised, non_indian_hf_donations
            ])

            final_data_with_missing_category = merged_donations.loc[
                (final_data['Category'].isnull()),
                ['Name of offsetting account', 'Document Date', 'Val/COArea Crcy', 'WBS',
                 'WBS Details', 'Sub WBS', 'SUB WBS Details', 'Category']
            ].rename(columns={'SUB WBS Details': 'Project Name'})

            csv = convert_df(final_data_with_missing_category)

            st.download_button(
                label='Download Projects with missing categories',
                data=csv,
                file_name='Project Directory.csv',
                mime='text/csv',
                use_container_width=True,
                type='primary'
            )

            st.dataframe(final_data_with_missing_category, hide_index=True, use_container_width=True)

        st.write('')
        st.write('##### Current Project Master')
        with st.expander('## Complete list of Project Master/Directory'):

            st.dataframe(project_directory[['Project Name', 'Category']], hide_index=True, use_container_width=True)


    # ------------------------------------------------------------------------------------------------------------------
    # Tracking distribution through SB
    # ------------------------------------------------------------------------------------------------------------------

    # Projects list
    # project_lists = final_data.loc[final_data['Document type'] == 'DR', 'Object'].drop_duplicates().to_list()

    # # Projects that were reallocated
    # project_lists_alloc = [project for project in
    #                        final_data.loc[final_data['Document type'] == 'DR', 'Object'].drop_duplicates().to_list() if
    #                        project[-3].isnumeric()]
    #
    # # Projects that were not reallocated
    # project_lists_non_alloc = [project for project in
    #                            final_data.loc[final_data['Document type'] == 'DR', 'Object'].drop_duplicates().to_list()
    #                            if not project[-3].isnumeric()]

    # donations_by_cause_1 = final_data[
    #     (final_data['Document type'] == 'DR') &
    #     (final_data['Object'].isin(project_lists))
    # ].reset_index(drop=True).copy()

    # def get_sb_donations(obj):
    #
    #     # Get subset of data belonging to a reference number
    #     df = final_data[final_data['Ref. document number'] == obj].copy()
    #
    #     # Filling na's with blank
    #     for col in df.columns:
    #         df[col] = df[col].fillna('')
    #
    #     # Create a pivot report
    #     df_pivot = df.groupby(['Ref. document number', 'WBS',
    #                              'WBS Details', 'Sub WBS', 'SUB WBS Details'])['Val/COArea Crcy'].sum()
    #
    #     # Reset the index for further filtering
    #     df_pivot = df_pivot.reset_index(drop=False).copy()
    #
    #     # if obj == 2406012170:
    #     #     print('new')
    #     #     print(df.shape[0])
    #     #     print(df_pivot.shape[0])
    #
    #     wbs_list = df_pivot['WBS'].str.lower().drop_duplicates().to_list()
    #     sub_wbs_list = df_pivot['SUB WBS Details'].str.lower().drop_duplicates().to_list()
    #
    #     wbs_list.extend(sub_wbs_list)
    #
    #     # Check if there's a WBS related to IITBHF
    #     is_hf = False
    #     for w in wbs_list:
    #         if 'iitbhf' in w or 'grant' in w or 'iitb hf' in w:
    #             is_hf = True
    #             break
    #
    #     # Check for shape
    #     if df_pivot.shape[0] > 1:
    #
    #         # Get donations greater than 0 as long as it's not from HF
    #         if is_hf:
    #             df_pivot = df_pivot[~df_pivot['WBS'].str.lower().str.contains('iitbhf')].copy()
    #
    #             df_pivot = df_pivot[~(df_pivot['SUB WBS Details'].str.lower().str.contains('grant'))].copy()
    #
    #             df_pivot = df_pivot[df_pivot['Val/COArea Crcy'] > 0].copy()
    #
    #         else:
    #             df_pivot = df_pivot[df_pivot['Val/COArea Crcy'] < 0].copy()
    #
    #     elif df_pivot.shape[0] == 1:
    #         if is_hf:
    #             df_pivot = pd.DataFrame(columns=['Ref. document number', 'WBS',
    #                              'WBS Details', 'Sub WBS', 'SUB WBS Details', 'Val/COArea Crcy'], index=[0])
    #
    #     # # Change the negative numbers
    #     # df_pivot['Val/COArea Crcy'] = df_pivot['Val/COArea Crcy'].abs()
    #
    #     return df_pivot
    #
    # donations_by_cause_1 = pd.DataFrame()
    #
    # # Get Unique reference document numbers
    # ref_nums = final_data.loc[
    #     (final_data['Document type'].isin(['DR','SB'])),
    #     'Ref. document number'
    # ].drop_duplicates().to_list()
    #
    # for ref in ref_nums:
    #     donations_by_cause_2 = get_sb_donations(ref)
    #
    #     donations_by_cause_1 = pd.concat([donations_by_cause_1, donations_by_cause_2])
    #
    # st.dataframe(donations_by_cause_1, use_container_width=True, hide_index=True)

    # # Subheader for detailed transaction data
    # st.write('')
    # st.subheader("Detailed Transaction Data")
    #
    # # Display the final data in a dataframe format
    # # st.dataframe(merged_donations, hide_index=True, use_container_width=True)
    #
    # st.dataframe(final_data, hide_index=True, use_container_width=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Get FUR for a WBS Code
    # ------------------------------------------------------------------------------------------------------------------
    # Get Projects
    project_lists = transaction['Project definition'].drop_duplicates().to_list()

    st.divider()

    st.header('Get Fund Utilisation for a WBS Code')
    st.write('')
    col12, col13 = st.columns(2)

    # WBS Code selection
    with col12:
        wbscode = st.selectbox(
            'Please select a WBS Code',
            project_lists,
            index=None,
            placeholder='Select WBS Code',
            label_visibility='collapsed'
        )

    with col13:
        if wbscode:

            wbs_sb = transaction[
                (transaction['Project definition'] == wbscode) &
                (transaction['Document type'] == 'SB')
            ].copy()

            # wbs_sb_value = wbs_sb['Val/COArea Crcy'].sum()
            #
            # wbs_sb_value = abs(wbs_sb_value)
            #
            # st.subheader(f'Cumulative Amount: {get_amount_in_cr(wbs_sb_value)}')

            wbs_sb_csv = convert_df(wbs_sb)

            st.download_button(
                label='Download all Transactions',
                data=wbs_sb_csv,
                file_name=f'Transactions for {wbscode}.csv',
                mime='text/csv',
                use_container_width=True,
                type='primary'
            )

        else:
            st.write()

    # Display Flow breakup
    st.write('')

    # If a WBS code is selected, generate and display FUR
    if wbscode:
        # Generate FUR & store it into a DataFrame
        df_wbs_code = get_fur(wbscode)

        # Calculate amount values for different breakup
        ## Fund
        ### Inflow
        fund_inflow = get_flow(df_wbs_code, 'inflow')

        # Get the Amount Value
        try:
            fund_inflow_value = fund_inflow['Val/COArea Crcy'].sum()

        except KeyError:
            fund_inflow_value = 0

        ### Outflow
        fund_outflow = get_flow(df_wbs_code, 'outflow')

        try:
            fund_outflow_value = fund_outflow['Val/COArea Crcy'].sum()

        except KeyError:
            fund_outflow_value = 0

        ## Admin
        ### Inflow
        admin_inflow = get_admin_expense(wbscode, 'inflow')

        # Get the Amount Value
        try:
            admin_inflow_value = admin_inflow['Val/COArea Crcy'].sum()

        except KeyError:
            admin_inflow_value = 0

        ### Outflow
        admin_outflow = get_admin_expense(wbscode, 'outflow')

        # Get the Amount Value
        try:
            admin_outflow_value = admin_outflow['Val/COArea Crcy'].sum()

        except KeyError:
            admin_outflow_value = 0

        ## Interests
        ### Inflow
        interests_inflow = get_interests(wbscode, 'inflow')

        # Get the Amount Value
        try:
            interests_inflow_value = interests_inflow['Val/COArea Crcy'].sum()

        except KeyError:
            interests_inflow_value = 0

        ### Outflow
        interests_outflow = get_interests(wbscode, 'outflow')

        # Get the Amount Value
        try:
            interests_outflow_value = interests_outflow['Val/COArea Crcy'].sum()

        except KeyError:
            interests_outflow_value = 0

        ## Others
        ### Inflow
        # other_inflow = get_flow(transaction, 'inflow')
        #
        # # Filter for the required WBS code
        # other_inflow = other_inflow[other_inflow['Project definition'] == wbscode]

        other_inflow = get_flow(wbs_sb, 'inflow')

        inflow_ref = pd.concat([fund_inflow['Ref. document number'], admin_inflow['Ref. document number'],
                                interests_inflow['Ref. document number']], ignore_index=True)

        other_inflow = other_inflow[~other_inflow['Ref. document number'].isin(inflow_ref)]

        # Get the Amount Value
        try:
            other_inflow_value = other_inflow['Val/COArea Crcy'].sum()
            other_inflow_value = other_inflow_value - (fund_inflow_value + admin_inflow_value + interests_inflow_value)

            if other_inflow_value < 0:
                other_inflow_value = 0

        except KeyError:
            other_inflow_value = 0

        ### Outflow
        # other_outflow = get_flow(transaction, 'outflow')
        #
        # # Filter for the required WBS code
        # other_outflow = other_outflow[other_outflow['Project definition'] == wbscode]

        other_outflow = get_flow(wbs_sb, 'outflow')

        outflow_ref = pd.concat([fund_outflow['Ref. document number'], admin_outflow['Ref. document number'],
                                interests_outflow['Ref. document number']], ignore_index=True)

        other_outflow = other_outflow[~other_outflow['Ref. document number'].isin(outflow_ref)]

        # Get the Amount Value
        try:
            other_outflow_value = other_outflow['Val/COArea Crcy'].sum()
            other_outflow_value = other_outflow_value - (
                    fund_outflow_value + admin_outflow_value + interests_outflow_value)

            if other_outflow_value < 0:
                other_outflow_value = 0

        except KeyError:
            other_outflow_value = 0

        # Display Breakup in Table
        st.write('### WBS Breakup')

        wbs_breakup = pd.DataFrame(
            data = {
                'Fund': [fund_inflow_value, fund_outflow_value],
                'Admin Expenses': [admin_inflow_value, admin_outflow_value],
                'Interests': [interests_inflow_value, interests_outflow_value],
                'Other (IITB Main Accounts / IRCC)': [other_inflow_value, other_outflow_value]
            }, index = ['From', 'To']
        )

        st.dataframe(wbs_breakup, hide_index=False, use_container_width=True,
                     # column_config={
                     #     'Fund': st.column_config.NumberColumn('Fund', format='₹'),
                     # }
                     )

        st.write('### Detailed WBS Breakup')
        with st.expander('### Detailed WBS Breakup'):

            # Display Fund
            if fund_inflow.shape[0] != 0 or fund_outflow.shape[0] != 0:
                col14, col15 = st.columns(2)

                with col14:
                    # Display Fund Inflow
                    st.subheader(f'⬇️ Fund Inflow: {get_amount_in_cr(fund_inflow_value)}')
                    st.dataframe(fund_inflow, hide_index=True, use_container_width=True)

                with col15:
                    # Display Fund Outflow
                    st.subheader(f'⬆️ Fund Outflow: {get_amount_in_cr(fund_outflow_value)}')
                    st.dataframe(fund_outflow, hide_index=True, use_container_width=True)

            # Display Admin Expense
            if admin_inflow.shape[0] != 0 or admin_outflow.shape[0] != 0:
                col16, col17 = st.columns(2)

                with col16:
                    # Display Admin Expense Inflow
                    st.subheader(f'⬇️ Admin Expense Inflow: {get_amount_in_cr(admin_inflow_value)}')
                    st.dataframe(admin_inflow, hide_index=True, use_container_width=True)

                with col17:
                    # Display Admin Expense Outflow
                    st.subheader(f'⬆️ Admin Expense Outflow: {get_amount_in_cr(admin_outflow_value)}')
                    st.dataframe(admin_outflow, hide_index=True, use_container_width=True)

            # Display Interests
            if interests_inflow.shape[0] != 0 or interests_outflow.shape[0] != 0:
                col18, col19 = st.columns(2)

                with col18:
                    # Display Interests Inflow
                    st.subheader(f'⬇️ Interests Inflow: {get_amount_in_cr(interests_inflow_value)}')
                    st.dataframe(interests_inflow, hide_index=True, use_container_width=True)

                with col19:
                    # Display Interests Outflow
                    st.subheader(f'⬆️ Interests Outflow: {get_amount_in_cr(interests_outflow_value)}')
                    st.dataframe(interests_outflow, hide_index=True, use_container_width=True)

            # Other expenses
            if other_inflow_value != 0 or other_outflow_value != 0:
                col20, col21 = st.columns(2)

                with col20:
                    # Display Main account & IRCC Inflow
                    st.subheader(
                        f'⬇️ Others (Transferred from IITB Main Accounts / IRCC): {get_amount_in_cr(other_inflow_value)}')
                    st.dataframe(other_inflow, hide_index=True, use_container_width=True)

                with col21:
                    # Display Main account & IRCC Inflow
                    st.subheader(
                        f'⬇️ Others (Transferred to IITB Main Accounts / IRCC): {get_amount_in_cr(other_outflow_value)}')
                    st.dataframe(other_outflow, hide_index=True, use_container_width=True)

    # if wbscode:
    #     st.dataframe(df_wbs_code, hide_index=True, use_container_width=True)