import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import random
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import datetime

@st.cache_data
def load_data():
    return pd.read_csv("gd.csv")

# Preprocess the data
def preprocess_data(df, features):
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=["Gender", "Race/Ethnicity", "Parental level of education", "Lunch", "Test preparation course"], drop_first=True)
    
    # Align user input features with encoded features
    features_encoded = pd.get_dummies(features, drop_first=True)
    features_aligned, _ = features_encoded.align(df_encoded, join='left', axis=1, fill_value=0)
    
    return features_aligned


# Train the model
def train_model(X_train, y_train, model_type):
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100)
    
    model.fit(X_train, y_train)
    return model


# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
        return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
        if make_hashes(password) == hashed_text:
                return hashed_text
        return False

# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
        c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
        c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
        conn.commit()

def login_user(username,password):
        c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
        data = c.fetchall()
        return data


def view_all_users():
        c.execute('SELECT * FROM userstable')
        data = c.fetchall()
        return data
def calculate_gold_rate(grams, place):
    # Assume a range of gold rates per gram for different places in India
    gold_rate_ranges = {
        "Delhi": (6150, 6180),
        "Mumbai": (6050, 670),
        "Chennai": (5980, 6020),
        "Kolkata": (6000, 6005)
    }
    
    # Get the range of gold rates for the specified place
    min_rate, max_rate = gold_rate_ranges.get(place, (5915, 6180))
    
    # Calculate the gold rate
    gold_rate = random.randint(min_rate, max_rate)
    
    # Calculate the total amount
    amount = grams * gold_rate
    
    return amount, gold_rate

def str2():
       st.title("Gold Price Prediction")
       # User input: grams of gold and place
       grams = st.number_input("Enter grams of gold", min_value=0.0, step=0.01)
       place = st.selectbox("Select place", ("Delhi", "Mumbai", "Chennai", "Kolkata"))
       # Calculate gold rate and amount
       if st.button("Predict"):
              amount, gold_rate = calculate_gold_rate(grams, place)
              st.write(f"Gold Rate in {place}: Rs. {gold_rate} per gram")
              st.write(f"Total amount for {grams} grams of gold: Rs. {amount}")

def main():

        st.markdown("<h1 style='text-align: center; color: green;'>Gold Price Prediction System</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: green;'>Intelligent Gold Price Prediction System using Machine Learning</h4>", unsafe_allow_html=True)

        @st.cache(persist=True)
        def load_data():
                data = pd.read_csv('gd.csv')
                label_encoder = LabelEncoder()

                # Encode categorical features
                categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
                for col in categorical_cols:
                        data[col] = label_encoder.fit_transform(data[col])
                        return data

        @st.cache(persist=True)
        def split_data(df):
                y = df['writing score']  # Target variable
                X = df.drop(columns=['writing score'])  # Features
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
                return X_train, X_test, y_train, y_test

        def predict_student_performance(X_test):
                model = RandomForestClassifier()  # You can choose any classifier here
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                return accuracy

        menu = ["HOME","ADMIN LOGIN","USER LOGIN","SIGN UP"]
        choice = st.sidebar.selectbox("Menu",menu)

        if choice == "HOME":
                st.markdown("<h1 style='text-align: center;'>HOMEPAGE</h1>", unsafe_allow_html=True)
                image = Image.open(r"image.jpg")
                st.image(image, caption='',use_column_width=True)
                st.subheader(" ")
                st.write("     <p style='text-align: center;'> Machine learning has emerged as a prominent research area for predicting gold prices, utilizing historical data and algorithms. The field aims to uncover patterns, trends, and connections among various factors that influence gold prices, including economic indicators, geopolitical events, and supply and demand dynamics.", unsafe_allow_html=True)
                time.sleep(3)
                st.warning("Goto Menu Section To Login !")

        elif choice == "ADMIN LOGIN":
                 st.markdown("<h1 style='text-align: center;'>Admin Login Section</h1>", unsafe_allow_html=True)
                 user = st.sidebar.text_input('Username')
                 passwd = st.sidebar.text_input('Password',type='password')
                 if st.sidebar.checkbox("LOGIN"):

                         if user == "Admin" and passwd == 'admin123':

                                                st.success("Logged In as {}".format(user))
                                                task = st.selectbox("Task",["Home","Profiles"])
                                                if task == "Profiles":
                                                        st.subheader("User Profiles")
                                                        user_result = view_all_users()
                                                        clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
                                                        st.dataframe(clean_db)
                                                str2()
                                                
                         else:
                                st.warning("Incorrect Admin Username/Password")
          
        elif choice == "USER LOGIN":
                st.markdown("<h1 style='text-align: center;'>User Login Section</h1>", unsafe_allow_html=True)
                username = st.sidebar.text_input("User Name")
                password = st.sidebar.text_input("Password",type='password')
                if st.sidebar.checkbox("LOGIN"):
                        # if password == '12345':
                        create_usertable()
                        hashed_pswd = make_hashes(password)

                        result = login_user(username,check_hashes(password,hashed_pswd))
                        if result:

                                st.success("Logged In as {}".format(username))
                                str2()
                                         
                        else:
                                st.warning("Incorrect Username/Password")
                                st.warning("Please Create an Account if not Created")

        elif choice == "SIGN UP":
                st.subheader("Create New Account")
                new_user = st.text_input("Username")
                new_password = st.text_input("Password",type='password')

                if st.button("SIGN UP"):
                        create_usertable()
                        add_userdata(new_user,make_hashes(new_password))
                        st.success("You have successfully created a valid Account")
                        st.info("Go to User Login Menu to login")

if __name__ == '__main__':
        main()
