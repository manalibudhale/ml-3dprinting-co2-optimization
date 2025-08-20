# Optimizing 3D Printing Parameters with Machine Learning for CO₂ Reduction  

This project demonstrates how **machine learning** can optimize **3D printing (Fused Deposition Modelling – FDM)** parameters to reduce **energy consumption** and **CO₂ emissions** while maintaining **print quality**.  

---

## 🚀 Project Highlights  
- **Goal:** Minimize the carbon footprint of 3D printing without sacrificing product integrity.  
- **Data:** ~1,950 samples & 182 features collected from **Prusa MK4** printer using *UltiMaker Cura* & *OctoPrint*.  
- **Target Variable:** Estimated Print Time (ET-mm), strongly correlated with **energy use** and **CO₂ emissions**.  

---

## 🧠 Machine Learning Approach  
- **Algorithms Implemented:**  
  - Linear Regression (baseline)  
  - Decision Trees & Random Forests  
  - **XGBoost (best-performing, with Bayesian optimization)**  
  - Artificial Neural Networks (various architectures)  

- **Techniques Used:**  
  - Feature engineering & one-hot encoding  
  - Scaling (MinMax, Robust, MaxAbs)  
  - Hyperparameter tuning (GridSearchCV, BayesSearchCV)  
  - Regularization, Batch Normalization & Early Stopping  

---

## 📊 Key Results  
- **XGBoost** achieved the strongest performance:  
  - Test **R² = 0.9690**  
  - Key influential parameters: *Inner Wall Speed, Outer Wall Speed, Print Speed, Travel Speed*  

- **ANNs** achieved **96% training accuracy**, but XGBoost generalized better on test data.  
- Clear evidence that **ML-driven optimization can reduce CO₂ emissions significantly** while retaining print quality.  

---

## 🛠️ Tech Stack  
- **Languages & Tools:** Python, Google Colab, Jupyter Notebook  
- **Libraries:** Scikit-learn, XGBoost, TensorFlow/Keras, Pandas, NumPy, Matplotlib, Seaborn  
- **3D Printing Tools:** UltiMaker Cura (v5.7.1), OctoPrint, Prusa MK4 Printer  

