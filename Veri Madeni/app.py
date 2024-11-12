import pandas as pd
import streamlit as st
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data and update column names
df = pd.read_csv('dataset.csv')
df = df.drop(['id','Class'], axis=1)
df.columns = df.columns.str.replace(r'[\s\.]', '_', regex=True)
df.columns = df.columns.str.replace(r'Gender&Type', 'Gender_Type', regex=True)
df.columns = df.columns.str.replace(r'#', '', regex=True)

df[['Monthly_Period', 'Credit1', 'InstallmentRate', 'Tenancy_Period', 'Age', 'Credits', 'Authorities']] = df[['Monthly_Period', 'Credit1', 'InstallmentRate', 'Tenancy_Period', 'Age', 'Credits', 'Authorities']].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
df[['InstallmentCredit', 'Yearly_Period']] = df[['InstallmentCredit', 'Yearly_Period']].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

# Select dependent and independent variables
x = df

# Preprocessing (StandardScaler)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ["Monthly_Period", "Credit1", "InstallmentRate", "Tenancy_Period", "Age", "Credits", "Authorities", "InstallmentCredit", "Yearly_Period"]),
        ('cat', OneHotEncoder(), ["Account1", "History", "Motive", "Account2", "Employment_Period", "Gender_Type", "Sponsors", "Plotsize", "Plan", "Housing", "Post", "Phone", "Expatriate"])
    ]
)

# Streamlit application
def cluster_pred(Account1, Monthly_Period, History, Motive, Credit1, Account2, Employment_Period, InstallmentRate, Gender_Type, Sponsors, Tenancy_Period,
                 Plotsize, Age, Plan, Housing, Credits, Post, Authorities, Phone, Expatriate, InstallmentCredit, Yearly_Period):
    input_data = pd.DataFrame({
        'Account1': [Account1],
        'Monthly_Period': [Monthly_Period],
        'History': [History],
        'Motive': [Motive],
        'Credit1': [Credit1],
        'Account2': [Account2],
        'Employment_Period': [Employment_Period],
        'InstallmentRate': [InstallmentRate],
        'Gender_Type': [Gender_Type],
        'Sponsors': [Sponsors],
        'Tenancy_Period': [Tenancy_Period],
        'Plotsize': [Plotsize],
        'Age': [Age],
        'Plan': [Plan],
        'Housing': [Housing],
        'Credits': [Credits],
        'Post': [Post],
        'Authorities': [Authorities],
        'Phone': [Phone],
        'Expatriate': [Expatriate],
        'InstallmentCredit': [InstallmentCredit],
        'Yearly_Period': [Yearly_Period]
    })

    model = joblib.load('Veri.pkl')
    scaler = StandardScaler()

    input_data_transformed = pd.get_dummies(input_data, drop_first=True)
    input_data_transformed = input_data_transformed.reindex(columns=x.columns, fill_value=0)
    input_data_transformed = scaler.transform(input_data_transformed)


    input_data_transformed = preprocessor.fit_transform(input_data)

    

    prediction = model.predict(input_data_transformed)
    return float(prediction[0])

st.title("KMeans Clustering Model")
st.write("Enter Input Data to Predict Cluster")

Account1 = st.selectbox('Account1', df['Account1'].unique())
Monthly_Period = st.slider('Monthly_Period', int(df['Monthly_Period'].min()), int(df['Monthly_Period'].max()))
History = st.selectbox('History', df['History'].unique())
Motive = st.selectbox('Motive', df['Motive'].unique())
Credit1 = st.slider('Credit1', int(df['Credit1'].min()), int(df['Credit1'].max()))
Account2 = st.selectbox('Account2', df['Account2'].unique())
Employment_Period = st.selectbox('Employment_Period', df['Employment_Period'].unique())
InstallmentRate = st.slider('InstallmentRate', int(df['InstallmentRate'].min()), int(df['InstallmentRate'].max()))
Gender_Type = st.selectbox('Gender_Type', df['Gender_Type'].unique())
Sponsors = st.selectbox('Sponsors', df['Sponsors'].unique())
Tenancy_Period = st.slider('Tenancy_Period', int(df['Tenancy_Period'].min()), int(df['Tenancy_Period'].max()))
Plotsize = st.selectbox('Plotsize', df['Plotsize'].unique())
Age = st.slider('Age', int(df['Age'].min()), int(df['Age'].max()))
Plan = st.selectbox('Plan', df['Plan'].unique())
Housing = st.selectbox('Housing', df['Housing'].unique())
Credits = st.slider('Credits', float(df['Credits'].min()), float(df['Credits'].max()))
Post = st.selectbox('Post', df['Post'].unique())
Authorities = st.slider('Authorities', float(df['Authorities'].min()), float(df['Authorities'].max()))
Phone = st.selectbox('Phone', df['Phone'].unique())
Expatriate = st.selectbox('Expatriate', [True, False])
InstallmentCredit = st.slider('InstallmentCredit', float(df['InstallmentCredit'].min()), float(df['InstallmentCredit'].max()))
Yearly_Period = st.slider('Yearly_Period', float(df['Yearly_Period'].min()), float(df['Yearly_Period'].max()))

if st.button('Predict Cluster'):
    cluster = cluster_pred(Account1, Monthly_Period, History, Motive, Credit1, Account2, Employment_Period, InstallmentRate, Gender_Type, Sponsors, Tenancy_Period,
                            Plotsize, Age, Plan, Housing, Credits, Post, Authorities, Phone, Expatriate, InstallmentCredit, Yearly_Period)
    st.write(f'The predicted cluster is: {cluster}')
