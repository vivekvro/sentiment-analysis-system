import streamlit as st
import requests,re


main_url="http://127.0.0.1:8000"
api_sentiment = "/sentiments"

st.title("Customer Feedback & Questions")
st.markdown("Enter your thoughts below: ")

user_id = st.text_area("Enter User ID (e.g. A341)",max_chars=6)



text = st.text_area(
    "Enter here",
    max_chars=300,
    placeholder="Write your feedback here (max 300 characters)...")



clean_text = re.sub(r"\s+", " ", text).strip()


def remove_emojis(text):
    return re.sub(
        r"[\U00010000-\U0010ffff]",
        "",
        text
    )
clean_text = remove_emojis(clean_text)





if st.button("Submit",disabled=st.session_state.get("submitting", False)):
    if not user_id.strip():
        st.warning("User ID is required")
        st.stop()

    if not clean_text:
        st.warning("Feedback cannot be empty")
        st.stop()

    with st.spinner("Submitting..."):
        input_data = {
            "user_id": user_id,
            "text": clean_text
        }

        try:
            response = requests.post(
                url=main_url + api_sentiment,
                json=input_data,
                timeout=5
            )

            if response.status_code == 200:
                st.success("Submitted")
            else:
                st.error(f"API error: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to FastAPI. Make sure it's running.")

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Try again.")
