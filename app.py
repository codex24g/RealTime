import streamlit as st

def main():
    st.title("Staff Image Recognition")

    # Add a selectbox for page navigation
    page = st.sidebar.selectbox("Choose a page", ["Image Classification", "Real-Time Staff Classification"])

    if page == "Image Classification":
        import image_classification  # Import the image classification module
        image_classification.run()
    elif page == "Real-Time Staff Classification":
        import real_time_classification  # Import the real-time classification module
        real_time_classification.run()

if __name__ == "__main__":
    main()
