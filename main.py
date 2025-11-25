import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import numpy as np # Added for data analysis

# --- Configuration & Data Loading (Adapted Helper Function) ---

@st.cache_data
def load_and_clean_data():
    """
    Simulates a 'get_data()' helper function.
    Downloads the dataset from Kaggle and performs cleaning.
    """
    with st.spinner('Downloading the dataset from Kaggle...'):
        try:
            # Note: KaggleHub download simulates a network fetch (GET request)
            path = kagglehub.dataset_download("risakashiwabara/tokyo-weatherdata")
        except Exception as e:
            st.error(f"Error during dataset download: {e}")
            return pd.DataFrame() # Return empty DataFrame on failure

    try:
        df = pd.read_csv(f"{path}/weather_tokyo_data.csv")
        
        # Data Cleaning and Transformation
        df['Date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['day'], format='%Y-%m/%d')
        df.columns = [item.strip() for item in df.columns]

        # Handle temperature data with parentheses
        df['temperature'] = df['temperature'].apply(lambda x: float(x.replace('(', '-').replace(')', '')))
            
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        
        st.success("Dataset successfully loaded and cleaned!")
        return df

    except Exception as e:
        st.error(f"Error during data cleaning or transformation: {e}")
        return pd.DataFrame()

# --- 1. Main UI Functions (Dashboard, Filter, Analyze Extremes) ---

def render_dashboard(df):
    """Sets up the Dashboard section with overview plots and metrics."""
    st.title("📊 Weather Data Dashboard")
    st.markdown("---")

    # --- Data Display (st.subheader & st.dataframe) ---
    st.subheader("Tokyo Weather Data Overview (First 10 Rows)")
    st.dataframe(df.head(10))
    
    st.subheader("Key Statistics")
    
    avg_temperature = df['temperature'].mean()
    st.metric(label="Overall Average Temperature", value=f"{avg_temperature:.2f}°C")
    
    # --- Monthly Average Plot ---
    st.subheader("Monthly Average Temperature Trend")
    monthly_avg_temp = df.groupby('Month')['temperature'].mean()

    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    monthly_avg_temp.plot(kind='bar', color='skyblue', ax=ax_bar)
    ax_bar.set_title('Monthly Average Temperature')
    ax_bar.set_xlabel('Month')
    ax_bar.set_ylabel('Average Temperature (°C)')
    ax_bar.set_xticks(ticks=range(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
    plt.tight_layout()
    st.pyplot(fig_bar)


def render_filter_data(df):
    """Sets up the Filter Data section using forms and inputs."""
    st.title("🔎 Filter Data")
    st.markdown("Use the form below to filter the dataset.")
    st.markdown("---")

    # --- Form Creation (st.text_input, st.selectbox, st.button) ---
    with st.form("filter_form"):
        st.header("Filter Parameters")
        
        # Category Selection (st.selectbox - adapted)
        all_years = sorted(df['Year'].unique())
        selected_year = st.selectbox("Select Year:", options=['All'] + all_years)
        
        all_months = list(range(1, 13))
        selected_month = st.selectbox("Select Month:", options=['All'] + all_months)
        
        # Text Input (st.text_input - adapted for min/max temp)
        min_temp = st.number_input("Minimum Temperature (°C):", value=-10.0, step=0.5)
        max_temp = st.number_input("Maximum Temperature (°C):", value=40.0, step=0.5)
        
        submitted = st.form_submit_button("Apply Filters")

    if submitted:
        filtered_df = df.copy()
        
        if selected_year != 'All':
            filtered_df = filtered_df[filtered_df['Year'] == selected_year]
            
        if selected_month != 'All':
            filtered_df = filtered_df[filtered_df['Month'] == selected_month]
            
        filtered_df = filtered_df[(filtered_df['temperature'] >= min_temp) & (filtered_df['temperature'] <= max_temp)]

        st.subheader(f"Filtered Results ({len(filtered_df)} Rows)")
        st.dataframe(filtered_df)
        
        if not filtered_df.empty:
            st.subheader("Filtered Daily Temperature Trends")
            fig_line, ax_line = plt.subplots(figsize=(12, 6))
            ax_line.plot(filtered_df['Date'], filtered_df['temperature'], color='red', linewidth=1)
            ax_line.set_title(f'Temperature Trends (Filtered Data)')
            ax_line.set_xlabel('Date')
            ax_line.set_ylabel('Temperature (°C)')
            ax_line.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig_line)
        else:
            st.warning("No data matches the selected filter criteria.")


def render_analyze_extremes(df):
    """Sets up the Analyze Extremes section."""
    st.title("🔥 Analyze Extremes")
    st.markdown("---")

    # --- Helper Functions (used internally for analysis) ---
    
    # Simulating a call to an 'update_data()'/'delete_data()' concept
    # by showing the most extreme rows as dataframes (instead of forms for editing)
    
    st.subheader("Temperature Extremes")

    hottest_day = df.loc[df['temperature'].idxmax()]
    coldest_day = df.loc[df['temperature'].idxmin()]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔥 Hottest Day:")
        # Displaying the record using st.dataframe as per instructions
        st.dataframe(hottest_day[['Date', 'temperature', 'year', 'day']].to_frame().T, hide_index=True)

    with col2:
        st.markdown("### 🧊 Coldest Day:")
        # Displaying the record using st.dataframe as per instructions
        st.dataframe(coldest_day[['Date', 'temperature', 'year', 'day']].to_frame().T, hide_index=True)

    st.markdown("---")
    
    st.subheader("Seasonal Average Temperatures")

    seasons = {
        'Spring 🌸': [3, 4, 5],
        'Summer ☀️': [6, 7, 8],
        'Autumn 🍂': [9, 10, 11],
        'Winter ☃️': [12, 1, 2]
    }

    seasonal_avg_temp = {}
    for season, months in seasons.items():
        seasonal_data = df[df['Month'].isin(months)]
        # This is where a helper function would retrieve the average
        seasonal_avg_temp[season] = seasonal_data['temperature'].mean()

    seasonal_df = pd.DataFrame(seasonal_avg_temp.items(), columns=['Season', 'Average Temperature (°C)'])
    seasonal_df['Average Temperature (°C)'] = seasonal_df['Average Temperature (°C)'].map('{:.2f}°C'.format)

    # Displaying results using st.table/st.dataframe as per instructions
    st.table(seasonal_df)


# --- 2. Main Application Logic (Using st.sidebar.selectbox) ---

def main():
    st.set_page_config(page_title="Tokyo Weather Analysis App", layout="wide")
    st.title('Tokyo Weather Data Analysis ☀️❄️')

    # Load and clean data (Simulating the 'get_recipes()' helper function)
    data_df = load_and_clean_data()
    
    if data_df.empty:
        st.stop()
        
    # Sidebar Navigation (st.sidebar.selectbox)
    menu_options = ["Dashboard", "Filter Data", "Analyze Extremes"]
    selection = st.sidebar.selectbox("Menu Items:", menu_options)
    
    # Main Area Rendering
    if selection == "Dashboard":
        render_dashboard(data_df)
    
    elif selection == "Filter Data":
        render_filter_data(data_df)
        
    elif selection == "Analyze Extremes":
        render_analyze_extremes(data_df)

if __name__ == "__main__":
    main()