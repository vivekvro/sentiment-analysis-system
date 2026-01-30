import streamlit as st, pandas as pd, requests


main_url="http://127.0.0.1:8000"
st.title("Support Dashboard")
st.subheader("Filter")
col1,col2 = st.columns(2)

with col1:
    status = st.selectbox("Status",["All","Pending"]) # add Resolved later
with col2:
    category = st.selectbox("Category",["All","payment-problems","positive-opinion","help-information-requests","Issues-Complaints"])

if st.button("Apply Filter"):
    with st.spinner("fetching records...."):
        try:
            # 🔹 Case 1: All status + All category
            if status == "All" and category == "All":
                res = requests.get(f"{main_url}/All", timeout=10)

            # 🔹 Case 2: Specific status + All category
            elif status != "All" and category == "All":
                res = requests.get(f"{main_url}/{status}", timeout=10)

            # 🔹 Case 3: Specific status + Specific category
            else:
                res = requests.get(
                    f"{main_url}/{status}/{category}",
                    timeout=10
                )

            # 🔹 Unified response handling
            if res.status_code == 200:
                data = res.json()
                st.success(f"{len(data)} records found")
                st.dataframe(data)

            elif res.status_code == 404:
                st.info("No records found")

            else:
                st.error(f"API error: {res.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to Backend server")
