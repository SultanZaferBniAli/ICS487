import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import plotly.graph_objects as go
import ta  # For technical indicators

# Configure page layout
st.set_page_config(
    page_title="Stock Analysis App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load and preprocess the dataset
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Ensure 'Date' is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert 'Date' to datetime
    
    # Rename columns for consistency
    df = df.rename(columns={'Price': 'Close', 'Vol.': 'Volume', 'Change %': 'Change'})
    
    # Ensure numeric conversions
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    
    # Convert 'Change' to numeric, handle errors
    df['Change'] = pd.to_numeric(df['Change'], errors='coerce')
    
    # Handle missing 'Change' column or invalid values
    if 'Change' not in df.columns or df['Change'].isna().all():
        df['Change'] = ((df['Close'] - df['Open']) / df['Open']) * 100  # Example calculation
    
    # Add the decision column
    def classify_change(change):
        if change > 1:
            return 'Buy'
        elif change < -1:
            return 'Sell'
        else:
            return 'Hold'
    
    df['Decision'] = df['Change'].apply(classify_change)
    return df


# Sidebar for file selection
st.sidebar.title("Options")
predefined_files = {
    "Zain": "Zain1.csv",
    "Almarai": "AlmaraiStocksData1.csv",
    "AlRajihi Bank": "Al-Rajhi-Bank.csv",
    "Sasco": "Sasco-Stocks.csv"
}
file_selection = st.sidebar.selectbox("Choose a File", list(predefined_files.keys()))

# Load the selected file
data = load_data(predefined_files[file_selection])
st.sidebar.success(f"Loaded predefined file: {file_selection}")

# Navbar
selected = option_menu(
    menu_title="CBR for Financial Investment Decision Support",
    options=["CART Algorithm", "Chart & Historical Data"],
    icons=["graph-down-arrow", "graph-up"],
    menu_icon="menu-app",
    default_index=0,
    orientation="horizontal",
)

# Page 1: CART Algorithm
if selected == "CART Algorithm":
    st.title("Decision Making using CART Algorithm")

    # Sidebar inputs for prediction using sorted selectbox
    st.sidebar.subheader("Predict Decision")
    price = st.sidebar.selectbox("Select Price", sorted(data['Close'].unique()))
    open_price = st.sidebar.selectbox("Select Open Price", sorted(data['Open'].unique()))
    high = st.sidebar.selectbox("Select High", sorted(data['High'].unique()))
    low = st.sidebar.selectbox("Select Low", sorted(data['Low'].unique()))
    volume = st.sidebar.selectbox("Select Volume", sorted(data['Volume'].unique()))
    submit = st.sidebar.button("Submit")

    # Preprocess the dataset
    model_data = data.drop(['Date', 'Change'], axis=1)
    X = model_data.drop('Decision', axis=1)
    y = model_data['Decision']

    # Train the CART model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cart_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    cart_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = cart_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Predict decision based on user input
    def predict_decision(price, open_price, high, low, volume):
        input_data = pd.DataFrame([[price, open_price, high, low, volume]],
                                  columns=['Close', 'Open', 'High', 'Low', 'Volume'])
        decision = cart_model.predict(input_data)
        return decision[0]

    if submit:
        # Display prediction at the top
        prediction = predict_decision(price, open_price, high, low, volume)
        if prediction == 'Buy':
            st.markdown(
                f"<h2 style='text-align: center; color: green;'>The suggested action is: <b>{prediction.upper()}</b></h2>",
                unsafe_allow_html=True
            )
        elif prediction == 'Sell':
            st.markdown(
                f"<h2 style='text-align: center; color: red;'>The suggested action is: <b>{prediction.upper()}</b></h2>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h2 style='text-align: center; color: yellow;'>The suggested action is: <b>{prediction.upper()}</b></h2>",
                unsafe_allow_html=True
            )
        
        # Safeguard for missing labels
        def safe_get_metric(report, label, metric):
            return report.get(label, {}).get(metric, 0.0)

        st.subheader("Classification Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Buy")
            st.write(f"Precision: {safe_get_metric(report_dict, 'Buy', 'precision'):.2f}")
            st.write(f"Recall: {safe_get_metric(report_dict, 'Buy', 'recall'):.2f}")
            st.write(f"F1-Score: {safe_get_metric(report_dict, 'Buy', 'f1-score'):.2f}")
            st.write(f"Support: {safe_get_metric(report_dict, 'Buy', 'support'):.0f}")
        with col2:
            st.markdown("### Hold")
            st.write(f"Precision: {safe_get_metric(report_dict, 'Hold', 'precision'):.2f}")
            st.write(f"Recall: {safe_get_metric(report_dict, 'Hold', 'recall'):.2f}")
            st.write(f"F1-Score: {safe_get_metric(report_dict, 'Hold', 'f1-score'):.2f}")
            st.write(f"Support: {safe_get_metric(report_dict, 'Hold', 'support'):.0f}")
        with col3:
            st.markdown("### Sell")
            st.write(f"Precision: {safe_get_metric(report_dict, 'Sell', 'precision'):.2f}")
            st.write(f"Recall: {safe_get_metric(report_dict, 'Sell', 'recall'):.2f}")
            st.write(f"F1-Score: {safe_get_metric(report_dict, 'Sell', 'f1-score'):.2f}")
            st.write(f"Support: {safe_get_metric(report_dict, 'Sell', 'support'):.0f}")

        # Display accuracy and overall metrics
        st.subheader("Overall Metrics")
        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown("### Accuracy")
            st.write(f"{accuracy:.2f}")
        with col5:
            st.markdown("### Macro Avg")
            st.write(f"Precision: {safe_get_metric(report_dict, 'macro avg', 'precision'):.2f}")
            st.write(f"Recall: {safe_get_metric(report_dict, 'macro avg', 'recall'):.2f}")
            st.write(f"F1-Score: {safe_get_metric(report_dict, 'macro avg', 'f1-score'):.2f}")
        with col6:
            st.markdown("### Weighted Avg")
            st.write(f"Precision: {safe_get_metric(report_dict, 'weighted avg', 'precision'):.2f}")
            st.write(f"Recall: {safe_get_metric(report_dict, 'weighted avg', 'recall'):.2f}")
            st.write(f"F1-Score: {safe_get_metric(report_dict, 'weighted avg', 'f1-score'):.2f}")

        # Visualize decision tree
        st.subheader("Decision Tree Visualization")
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(cart_model, feature_names=X.columns, class_names=cart_model.classes_, filled=True, ax=ax)
        st.pyplot(fig)

# Page 2: Chart & Historical Data
elif selected == "Chart & Historical Data":
    st.title("Chart & Historical Data")

    # Sidebar inputs for filtering
    st.sidebar.subheader("Filter Data")
    start_date = st.sidebar.date_input("Start Date", min_value=data['Date'].min(), max_value=data['Date'].max(), value=data['Date'].min())
    end_date = st.sidebar.date_input("End Date", min_value=data['Date'].min(), max_value=data['Date'].max(), value=data['Date'].max())
    indicators = st.sidebar.multiselect(
        "Choose Technical Indicators",
        ["SMA 20", "EMA 20", "Bollinger Bands"]
    )
    apply_filter = st.sidebar.button("Apply Filters")

    if apply_filter:
        filtered_data = data[(data['Date'] >= pd.Timestamp(start_date)) & (data['Date'] <= pd.Timestamp(end_date))]
        if filtered_data.empty:
            st.error("No data available for the selected date range.")
        else:
            st.subheader("Filtered Data Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("High Price", f"{filtered_data['High'].max():.2f}")
            col2.metric("Low Price", f"{filtered_data['Low'].min():.2f}")
            col3.metric("Total Volume", f"{filtered_data['Volume'].sum():,.2f}")

            # Add indicators
            if "SMA 20" in indicators:
                filtered_data['SMA_20'] = ta.trend.sma_indicator(filtered_data['Close'], window=20)
            if "EMA 20" in indicators:
                filtered_data['EMA_20'] = ta.trend.ema_indicator(filtered_data['Close'], window=20)
            if "Bollinger Bands" in indicators:
                filtered_data['UpperBB'] = filtered_data['Close'].rolling(window=20).mean() + 2 * filtered_data['Close'].rolling(window=20).std()
                filtered_data['LowerBB'] = filtered_data['Close'].rolling(window=20).mean() - 2 * filtered_data['Close'].rolling(window=20).std()

            st.subheader("Candlestick Chart with Indicators")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=filtered_data['Date'],
                open=filtered_data['Open'],
                high=filtered_data['High'],
                low=filtered_data['Low'],
                close=filtered_data['Close'],
                name="Candlestick"
            ))

            if "SMA 20" in indicators:
                fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['SMA_20'], mode='lines', name='SMA 20'))
            if "EMA 20" in indicators:
                fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['EMA_20'], mode='lines', name='EMA 20'))
            if "Bollinger Bands" in indicators:
                fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['UpperBB'], mode='lines', name='Upper BB'))
                fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['LowerBB'], mode='lines', name='Lower BB'))

            fig.update_layout(
                title="Candlestick Chart with Indicators",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Filtered Data")
            st.dataframe(filtered_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Decision']])
