# 📈 Ecommerce Click Intent Analysis: Logistic Regression Project

## 🗂️ Project Overview
This project analyzes consumer behavior and demographic data from an e-commerce platform to predict advertising engagement. The ultimate objective is to determine the likelihood of a user clicking on an advertisement based on metrics like digital activity, age, and area income. By predicting consumer click intent, a company can optimize its digital marketing budgets, minimize ad waste, and maximize Return on Ad Spend (ROAS).

---

## 🚀 Key Insights from Exploratory Data Analysis (EDA)
* **The Bi-Modal Divide**: A joint visualization of *Daily Time Spent on Site* versus *Daily Internet Usage* reveals two completely isolated consumer segments. Heavy internet users browse the site intensely but display high "ad-blindness," whereas casual internet users browse selectively and are significantly more likely to click on advertisements.
* **Demographic Core**: The audience profile forms a steady bell-shaped distribution peaking tightly between **ages 30 and 45**, representing a stable consumer segment with mature spending power.
* **Socioeconomic Stratification**: The active consumer core is concentrated in residential areas with stable, middle-to-upper-class baseline household incomes ranging from **\$45,000 to \$78,000** annually.

---

## 🛠️ Data Profile & Feature Engineering
The dataset contains consumer behavioral footprints. High-cardinality text columns (`City`, `Country`, `Ad Topic Line`, `Timestamp`) were intentionally excluded from this baseline deployment to prevent column inflation and model degradation. 

The machine learning model utilizes the following clean numerical features:
* `Daily Time Spent on Site`: Average minutes spent on the platform per day.
* `Age`: Consumer age in years.
* `Area Income`: Average annual household income of the consumer's geographic neighborhood.
* `Daily Internet Usage`: Total average daily minutes spent online by the consumer.
* `Male`: Binary flag indicating gender (0 = Female, 1 = Male).
* **`Clicked on Ad`**: The categorical target variable (**y**) to be predicted (0 = Did not click, 1 = Clicked).

---

## 📊 Model Evaluation Results

The data was segmented using a standard 67/33 train/test split. The trained Logistic Regression model was evaluated against completely unseen testing data, producing the following performance metrics:

### 1. Confusion Matrix
```text
[[149   8]
 [ 15 158]]
```

### 2. Classification Report
```text
              precision    recall  f1-score   support

           0       0.91      0.95      0.93       157
           1       0.94      0.91      0.93       173

    accuracy                           0.93       330
   macro avg       0.93      0.93      0.93       330
weighted avg       0.93      0.93      0.93       330
```

* **Overall Accuracy**: **93%** — The model successfully predicted the correct outcome for 307 out of 330 unseen consumers.
* **Precision (Class 1)**: **94%** — When the system flags an individual as a likely clicker, it is highly reliable.
* **Balanced F1-Score**: **0.93** — Symmetrical performance across both categories proves that the mathematical sigmoid boundary generalizes beautifully without bias toward a single outcome.

---

## 💡 Final Conclusions & Business Recommendations

1. **Algorithmic Audience Filtering**: The platform should integrate this trained Logistic Regression classifier directly into the ad-delivery backend network. By evaluating incoming user behavior data in real time, the infrastructure can selectively display promotions *only* to individuals flagged as high-probability clickers (`Class 1`).
2. **Budget Reallocation**: Ad spend should be aggressively redirected away from heavy power-browsers (who display high ad resistance) and funneled toward consumers aged **35–45** residing in mid-to-lower area income bands. These profiles demonstrate the lowest structural friction to ad engagement.
3. **Architecture Scalability**: Because the mathematical threshold of this model is exceptionally robust, the script is fully optimized to be containerized using Docker and deployed as an isolated Python FastAPI microservice connected to enterprise Angular/NestJS/SQL software stacks.
