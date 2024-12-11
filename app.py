import pandas as pd
import streamlit as st
from datetime import datetime
from datetime import date
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from urllib.parse import quote_plus

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
    return f'₹ {round(amount / 10000000, 2)} Cr.'

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
    st.subheader('3. Project Master file', help="It should only contain columns: 'Project Name' and 'Category'")
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

    # st.divider()
    #
    # # Clear Cache
    # if st.button("Clear Cache"):
    #     # Clear values from *all* all in-memory and on-disk data caches:
    #     # i.e. clear values from both square and cube
    #     st.cache_data.clear()

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

    if transaction is not None:
        # Perform WBS mapping
        transaction = transaction.merge(wbs[['WBS ', 'WBS Details ']].drop_duplicates(), how='left',
                                        left_on='Project definition', right_on='WBS ')

        transaction = transaction.merge(wbs[['Sub WBS ', 'SUB WBS Details']].drop_duplicates(), how='left',
                                        left_on='Object', right_on='Sub WBS ')

        transaction['Project Id'] = transaction['SUB WBS Details'].apply(
            lambda x: str(x).strip().lower().replace(' ', '_'))

if transaction is not None:
    # Assign Category
    transaction = transaction.merge(project_directory[['Project Id', 'Category']], how='left', on='Project Id').copy()

# Final Data
# Get existing data
query_final_data = '''
    SELECT
        *
    FROM
        erp_data.funds_received.transactions
'''

if transaction is not None or wbs is not None or project_directory is not None:
     final_data = transaction[
         (transaction['Document Date'].between(start_date, end_date))
     ]

     update_sql_table(final_data, 'transactions', schema='funds_received', replace=True)

     fetch_data_from_sql.clear(query_final_data)

final_data = fetch_data_from_sql(query_final_data)

# Present the data
# ----------------------------------------------------------------------------------------------------------------------
if final_data.shape[0] == 0:
    st.warning('Upload the data to proceed')

else:
    # Donation Received Metrics
    st.header('Summary')

    final_data = (final_data[final_data['Document Date'].between(start_date, end_date)])

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
    if wbs is not None:
        st.write('')
        st.subheader('Project Classification')

        col10, col11 = st.columns(2)

        # Cause-wise donation
        donations_by_cause = final_data.groupby('Category')['Val/COArea Crcy'].sum().abs()
        donations_by_cause = donations_by_cause.reset_index(drop=False)
        donations_by_cause.columns = ['Cause', 'Amount Received']

        with col10:
            st.dataframe(donations_by_cause, use_container_width=True, hide_index=True)

        with st.expander('## Complete list of Project Master/Directory'):
            csv = convert_df(project_directory[['Project Name', 'Category']])

            st.download_button(
                label='Download',
                data=csv,
                file_name='Project Directory.csv',
                mime='text/csv',
                use_container_width=True,
                type='primary'
            )

            st.dataframe(project_directory[['Project Name', 'Category']], hide_index=True, use_container_width=True)

    # Subheader for detailed transaction data
    st.write('')
    st.subheader("Detailed Transaction Data")

    # Display the final data in a dataframe format
    st.dataframe(final_data, hide_index=True)
