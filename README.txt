SENTIMENT DETECTION FROM ONLINE REVIEWS — STREAMLIT PROJECT

HOW TO RUN IN VS CODE (WINDOWS)
1. Extract this ZIP into a folder.
2. Open VS Code, then File > Open Folder and select Sentiment_Review_Analytics.
3. Select Terminal > New Terminal.
4. Run: python -m pip install -r requirements.txt
5. Run: python -m streamlit run app.py
6. Open the Local URL printed in the terminal, usually http://localhost:8501.

The app has built-in synthetic demo reviews and does not need an API key.
To use your own CSV, include a review_text, review, text, or content column.
Optional columns: item_name/product/item/title, category/product_category/type,
sentiment_label, suspicious_label.

VADER provides an estimated sentiment. The suspicious-review check is a simple heuristic,
not proof that a review is fake. Built-in examples are synthetic, not real customer reviews.
