import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Flight Price Prediction System",
    page_icon="✈️",
    layout="wide"
)
#-----------------
import pandas as pd
import numpy as np
import joblib
import gdown
import os

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    MODEL_PATH = "flight_price_prediction_model.pkl"
    #FILE_ID = "1bBW5tmh4SDyo255cWyzPv-ZRxnhn8jPB" 
    #commented at 9.33am 31st july
    FILE_ID ="1mykKGcRz_QdaHUaXeyQ0vdyWn_D44ggP"

    URL = f"https://drive.google.com/uc?id={FILE_ID}"

    if not os.path.exists(MODEL_PATH):
        #st.write("Downloading model...")
        gdown.download(URL, MODEL_PATH, quiet=False)

    try:
        #st.write(f"Model file size: {os.path.getsize(MODEL_PATH)/(1024*1024):.2f} MB")
       # st.write("About to load model...")

       # model = joblib.load(MODEL_PATH)
        import gc

        #st.write("Loading model...")
        gc.collect()

        model = joblib.load(MODEL_PATH)

        #st.success("Model loaded!")

        #st.success("Model loaded successfully!")

    except Exception as e:
        st.error(f"Error loading model: {type(e).__name__}: {e}")
        st.stop()

    return model


# Call the function OUTSIDE the function definition
model = load_model()
#added
st.write(type(model))
st.write(model)
st.write(model.n_estimators)
st.write(model.feature_names_in_)
st.write(input_data.columns[input_data.columns.str.contains("dep_daytime")])
#end sheer

# ---------------- Preprocessing FUNCTIONS ----------------

def get_period(hour):
    """
    Returns the period of the day.
    """

    if 0 <= hour < 6:
        return "Early_morning"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    else:
        return "Night"


def get_departure_daytime(hour):

    if hour >= 18 or hour < 6:
        return "Night Departure"

    return "Day Departure"


def get_arrival_daytime(hour):

    if hour >= 18 or hour < 6:
        return "Night Arrival"

    return "Day Arrival"


def cyclic_hour(hour):

    sin_value = np.sin(2 * np.pi * hour / 24)
    cos_value = np.cos(2 * np.pi * hour / 24)

    return sin_value, cos_value

# ---------------- MAIN PREDICTION FUNCTION ----------------

# ------- Clear previous prediction whenever inputs change--------------
def clear_prediction():
    st.session_state.prediction = None
    st.session_state.prediction_status = "🟡 Waiting"
    st.session_state.prediction_made = False

def predict_price( 
    airline,
    departure_city,
    destination_city,
    month_category,
    day,
    travel_class,
    stops, 
    departure_time,
    arrival_time,
    duration
):

    dep_hour = departure_time.hour
    arr_hour = arrival_time.hour

    feature_names = model.feature_names_in_

    input_data = pd.DataFrame(
        [[0] * len(feature_names)],
        columns=feature_names
    )

    # ---------------- NUMERICAL FEATURES ----------------

    input_data["day"] = day
    input_data["dep_hour"] = dep_hour
    input_data["arr_hour"] = arr_hour
    input_data["duration_in_min"] = duration

    # ---------------- AIRLINE ----------------

    column = f"airline_{airline}"

    if column in input_data.columns:
        input_data[column] = 1

    # ---------------- FROM CITY ----------------

    column = f"from_{departure_city}"

    if column in input_data.columns:
        input_data[column] = 1

    # ---------------- TO CITY ----------------

    column = f"to_{destination_city}"

    if column in input_data.columns:
        input_data[column] = 1

    # ---------------- ROUTE ----------------

    route = f"{departure_city}-{destination_city}"

    column = f"route_{route}"

    if column in input_data.columns:
        input_data[column] = 1

    # ---------------- CLASS ----------------

    if travel_class == "Economy":
        input_data["class_category_Economy"] = 1

    # ---------------- MONTH ----------------

    if month_category == "March":
        input_data["month_category_March"] = 1

    # ---------------- STOPS ----------------

    if stops == "Non-stop":
        input_data["stops_category_Non-stop"] = 1

    elif stops == "Multiple-Stops":
        input_data["stops_category_Multiple-Stops"] = 1

    # ---------------- DEPARTURE PERIOD ----------------

    dep_period = get_period(dep_hour)

    column = f"dep_period_{dep_period}"

    if column in input_data.columns:
        input_data[column] = 1

    # ---------------- ARRIVAL PERIOD ----------------

    arr_period = get_period(arr_hour)

    column = f"arr_period_{arr_period}"

    if column in input_data.columns:
        input_data[column] = 1

    # ---------------- DAYTIME CATEGORY ----------------

    dep_day = get_departure_daytime(dep_hour)

    column = f"dep_daytime_category_{dep_day}"

    if column in input_data.columns:
        input_data[column] = 1

    arr_day = get_arrival_daytime(arr_hour)

    column = f"arr_daytime_category_{arr_day}"

    if column in input_data.columns:
        input_data[column] = 1

    # ---------------- CYCLICAL FEATURES ----------------

    dep_sin, dep_cos = cyclic_hour(dep_hour)

    arr_sin, arr_cos = cyclic_hour(arr_hour)

    input_data["dep_hour_sin"] = dep_sin
    input_data["dep_hour_cos"] = dep_cos

    input_data["arr_hour_sin"] = arr_sin
    input_data["arr_hour_cos"] = arr_cos

    # ---------------- PREDICT ----------------
    # added
    st.write("Model expects", len(model.feature_names_in_), "features")

    st.write("Features sent to model:")
    st.dataframe(input_data.loc[:, (input_data != 0).any(axis=0)].T)
    #endss here
    prediction = model.predict(input_data)[0]
   # print(input_data.loc[:, (input_data != 0).any(axis=0)].T)
    #st.write(input_data.loc[:, (input_data != 0).any(axis=0)].T)
    return round(prediction, 2)



#-----------------Dropdowns values--------------------
AIRLINES = [
    "Air India",
    "AirAsia",
    "GO FIRST",
    "Indigo",
    "SpiceJet",
    "StarAir",
    "Trujet",
    "Vistara",
]

DEPARTURE_CITIES = [
    "Bangalore",
    "Chennai",
    "Delhi",
    "Hyderabad",
    "Kolkata",
    "Mumbai",
]

DESTINATION_CITIES = [
    "Bangalore",
    "Chennai",
    "Delhi",
    "Hyderabad",
    "Kolkata",
    "Mumbai",
]

MONTHS = [
    "February",
    "March",
]

TRAVEL_CLASSES = [
    "Economy",
    "Business",
]

STOPS = [
    "Non-stop",
    "1-Stop",
    "Multiple-Stops",
]
# ---------------- SESSION STATE ----------------

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "prediction_status" not in st.session_state:
    st.session_state.prediction_status = "🟡 Waiting"

if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

if "model_name" not in st.session_state:
    st.session_state.model_name = "Random Forest"
#------ Header---------------

with st.container(border=True):

    st.subheader("✈️ Flight Price Prediction System")
    st.caption("Predict airline ticket prices using Machine Learning")


# ---------------- MAIN CONTENT ----------------

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:

    #flight_data = show_input_form()

    st.subheader("Flight Information")
    col1, col2 = st.columns(2)
    
        # ---------- LEFT COLUMN ----------
    with col1:
    
            airline = st.selectbox(
                "Airline",
                AIRLINES,
                on_change=clear_prediction
            )
    
            departure_city = st.selectbox(
                "Departure City",
                DEPARTURE_CITIES,
                on_change=clear_prediction
            )
    
            month_category = st.selectbox(
                "Travel Month",
                MONTHS,
                on_change=clear_prediction
            )
    
            departure_time = st.time_input(
                "Departure Time",
                on_change=clear_prediction
            )
    
            stops = st.selectbox(
                "Stops",
                STOPS,
                index=0,
                on_change=clear_prediction
            )
    
    # ---------- RIGHT COLUMN ----------
    with col2:
    
            travel_class = st.selectbox(
                "Travel Class",
                TRAVEL_CLASSES,
                on_change=clear_prediction
            )
    
            destination_city = st.selectbox(
                "Destination City",
                DESTINATION_CITIES,
                index=1,
                on_change=clear_prediction
            )
    
            day = st.selectbox(
                "Travel Day",
                list(range(1, 32)),
                on_change=clear_prediction
            )
    
            arrival_time = st.time_input(
                "Arrival Time",
                on_change=clear_prediction
            )
    
            duration = st.number_input(
                "Flight Duration (minutes)",
                min_value=30,
                step=1,
                on_change=clear_prediction
            )
    
    #st.write("")
    
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #0B3C6F;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    predict = st.button(
            "Predict Flight Price",
            use_container_width=True
        )

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

if predict:

    if departure_city == destination_city:

        st.session_state.prediction = None
        st.session_state.prediction_status = "❌ Invalid Input"
        st.session_state.prediction_made = False

        st.error("Departure City and Destination City cannot be the same.")

    elif departure_time == arrival_time:

        st.session_state.prediction = None
        st.session_state.prediction_status = "❌ Invalid Input"
        st.session_state.prediction_made = False

        st.error(
            "Departure Time and Arrival Time cannot be the same. "
            "Please enter a valid flight schedule."
        )

    else:

        try:

            price = predict_price(
                airline=airline,
                departure_city=departure_city,
                destination_city=destination_city,
                month_category=month_category,
                day=day,
                travel_class=travel_class,
                stops=stops,
                departure_time=departure_time,
                arrival_time=arrival_time,
                duration=duration,
            )

            st.session_state.prediction = price
            st.session_state.prediction_status = "✅ Completed"
            st.session_state.prediction_made = True

        except Exception as e:

            st.session_state.prediction = None
            st.session_state.prediction_status = "⚠️ Prediction Failed"
            st.session_state.prediction_made = False

            st.error(f"Prediction failed: {e}")

# ---------------- PREDICTION CARD ----------------

with right_col:
    st.subheader(" ₹ Flight Price Prediction")
    
    if st.session_state.prediction is None:
            predicted_price = "--"
    else:
            predicted_price = f"₹ {st.session_state.prediction:,.2f}"
    
    with st.container(border=True):
    
            st.caption("Estimated Flight Price")
            st.markdown(
                f"<h2 style='margin-top:0'>{predicted_price}</h2>",
                unsafe_allow_html=True
            )
        # Display only after a successful prediction
            if st.session_state.prediction_made:
                st.caption(
                "This is the predicted airfare based on the selected flight details and using the trained Random Forest regression model."
            )

    
            st.divider()
    
            st.caption("Prediction Status")
            st.markdown(
                f"**{st.session_state.prediction_status}**"
            )

# ---------------- MODEL INFO ----------------
st.subheader("🤖 Model Information")

model_info = pd.DataFrame({
        "Algorithm": ["Random Forest Regressor"],
        "Feature Set": ["FS4"],
        "R² Score": ["0.988314"],
        "MAE": ["1008.84"],
        "RMSE": ["2448.83"],
        "MSE":["5996782.92"]
        })

st.table(model_info.style.hide(axis="index"))

# ---------------- FOOTER ----------------
st.markdown(
        """
        <div style="
            text-align:center;
            color:#6B7280;
            font-size:14px;
            padding-top:10px;
            padding-bottom:5px;
        ">
            Flight Price Prediction Dashboard |
            Machine Learning for Developers (MLDP) Project |
            Developed by Trisha Balani
        </div>
        """,
        unsafe_allow_html=True
    )

#st.markdown("""
#<style>

#.stApp {
#    background: linear-gradient(
#        to bottom,
#        #87CEEB 0%,
#        #BFE8FF 35%,
#        #EAF7FF 70%,
#        #FFFFFF 100%
#    );
#}

#</style>
#""", unsafe_allow_html=True)
