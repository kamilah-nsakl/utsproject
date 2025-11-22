import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve
)

# Opsional: SMOTE (imbalance handling)
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

sns.set(style="whitegrid")

# --------------------------------------------------
# KONFIGURASI HALAMAN
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

st.sidebar.title("🩺 Diabetes ML Dashboard")
st.sidebar.write("Analisis, visualisasi, dan pemodelan prediksi diabetes.")

# --------------------------------------------------
# INPUT DATASET
# --------------------------------------------------
st.sidebar.subheader("Dataset")
uploaded = st.sidebar.file_uploader(
    "Upload dataset CSV (mis. Dataset9_Diabetes_Prediction.csv)",
    type=["csv"]
)

default_path = "Dataset9_Diabetes_Prediction.csv"
use_default = st.sidebar.checkbox(
    f"Gunakan file lokal: {default_path}",
    value=False
)

@st.cache_data
def load_data(uploaded_file, use_default_local, default_path):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif use_default_local:
        df = pd.read_csv(default_path)
    else:
        df = None
    return df

df = load_data(uploaded, use_default, default_path)

# --------------------------------------------------
# FUNGSI PRA-PROSES (sesuai notebook)
# --------------------------------------------------
@st.cache_data
def preprocess_data(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # Salin seperti di notebook: df_copy
    df_copy = df.copy(deep=True)
    cols_zero_to_nan = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_copy[cols_zero_to_nan] = df_copy[cols_zero_to_nan].replace(0, np.NaN)

    # Imputasi median
    for c in cols_zero_to_nan:
        df_copy[c].fillna(df_copy[c].median(), inplace=True)

    # df1 untuk EDA status diabetes dan BMI
    df1 = df_copy.copy()

    # Status diabetes
    df1.loc[df1['Outcome'] == 1, 'Status_Diabetes'] = 'Positif'
    df1.loc[df1['Outcome'] == 0, 'Status_Diabetes'] = 'Negatif'

    # Status BMI
    def get_bmi_status(bmi):
        if bmi <= 18.4:
            return "Low"
        elif bmi <= 25:
            return "Normal"
        else:
            return "High"

    df1["Status_BMI"] = df1["BMI"].apply(get_bmi_status)

    # Status Blood Pressure
    df1.loc[df1['BloodPressure'] < 59, 'Status_Bloodpressure'] = 'Low'
    df1.loc[(df1['BloodPressure'] >= 60) & (df1['BloodPressure'] <= 80), 'Status_Bloodpressure'] = 'Normal'
    df1.loc[df1['BloodPressure'] > 80, 'Status_Bloodpressure'] = 'High'

    return df_copy, df1

# --------------------------------------------------
# FUNGSI TRAINING MODEL (RF, KNN, LOGREG)
# --------------------------------------------------
@st.cache_resource
def train_models(df_copy: pd.DataFrame):
    X = df_copy.drop("Outcome", axis=1)
    y = df_copy["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=8
    )

    # Random Forest
    rf = RandomForestClassifier(random_state=8)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    # Logistic Regression
    log = LogisticRegression(max_iter=500)
    log.fit(X_train, y_train)
    y_pred_log = log.predict(X_test)

    # ROC & AUC (berdasarkan test set)
    def get_auc_scores(model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        return auc_score, fpr, tpr, y_pred

    auc_rf, fpr_rf, tpr_rf, y_pred_rf = get_auc_scores(rf, X_test, y_test)
    auc_knn, fpr_knn, tpr_knn, y_pred_knn = get_auc_scores(knn, X_test, y_test)
    auc_log, fpr_log, tpr_log, y_pred_log = get_auc_scores(log, X_test, y_test)

    results = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "rf": rf,
        "knn": knn,
        "log": log,
        "auc_rf": auc_rf,
        "auc_knn": auc_knn,
        "auc_log": auc_log,
        "fpr_rf": fpr_rf,
        "tpr_rf": tpr_rf,
        "fpr_knn": fpr_knn,
        "tpr_knn": tpr_knn,
        "fpr_log": fpr_log,
        "tpr_log": tpr_log,
        "y_pred_rf": y_pred_rf,
        "y_pred_knn": y_pred_knn,
        "y_pred_log": y_pred_log
    }

    # SMOTE (opsional, untuk ditampilkan di interface)
    if HAS_SMOTE:
        smote = SMOTE(random_state=8)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        model_tuned = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            max_features='sqrt',
            min_samples_leaf=5,
            random_state=8
        ).fit(X_smote, y_smote)
        tuned_pred = model_tuned.predict(X_smote)
        tuned_acc = accuracy_score(y_smote, tuned_pred)
        results["smote_info"] = {
            "X_smote": X_smote,
            "y_smote": y_smote,
            "model_tuned": model_tuned,
            "tuned_acc": tuned_acc
        }

    return results

# --------------------------------------------------
# LAYOUT MENU
# --------------------------------------------------
menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Beranda",
        "EDA & Preprocessing",
        "Visualisasi",
        "Pemodelan",
        "Prediksi Individu",
        "Tentang Data"
    ]
)

# --------------------------------------------------
# KONTEN HALAMAN
# --------------------------------------------------
if df is None:
    st.title("🩺 Diabetes Prediction Dashboard")
    st.info(
        "Silakan upload dataset atau aktifkan opsi file lokal di sidebar "
        "untuk mulai menggunakan dashboard."
    )
else:
    # Preprocess
    df_copy, df1 = preprocess_data(df)

    if menu == "Beranda":
        st.title("🩺 Diabetes Prediction Dashboard")
        st.markdown("""
        Dashboard ini menyajikan **analisis data, visualisasi, dan pemodelan machine learning**
        untuk prediksi diabetes berdasarkan dataset klinis.
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Observasi", df_copy.shape[0])
        with col2:
            st.metric("Jumlah Fitur", df_copy.shape[1] - 1)
        with col3:
            st.metric("Fitur Target", "Outcome (Diabetes)")

        st.subheader("Cuplikan Dataset")
        st.dataframe(df_copy.head())

    elif menu == "EDA & Preprocessing":
        st.title("🔍 EDA & Preprocessing")

        tab1, tab2, tab3 = st.tabs(
            ["Informasi Dataset", "Missing Value & Imputasi", "Distribusi Outcome"]
        )

        with tab1:
            st.subheader("Informasi Struktur Dataset")
            st.write("**Tipe data dan jumlah non-null:**")
            buf = []
            df_copy.info(buf=buf)
            info_str = "\n".join(buf)
            st.text(info_str)

            st.subheader("Statistik Deskriptif")
            st.write(df_copy.describe())

        with tab2:
            st.subheader("Missing Value (sebelum imputasi)")
            cols_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            zero_counts = (df[cols_zero] == 0).sum()
            st.write("Jumlah nilai 0 (dianggap missing) per kolom:")
            st.write(zero_counts.to_frame("count"))

            st.markdown("""
            Nilai 0 pada beberapa variabel klinis dianggap sebagai *missing value* dan
            diganti dengan **median** dari masing-masing kolom (seperti pada notebook).
            """)

        with tab3:
            st.subheader("Distribusi Status Diabetes")
            status_counts = df1["Status_Diabetes"].value_counts()

            col1, col2 = st.columns(2)
            with col1:
                st.write("Tabel frekuensi:")
                st.write(status_counts.to_frame("Jumlah"))
            with col2:
                fig, ax = plt.subplots()
                ax.pie(
                    status_counts.values,
                    labels=status_counts.index,
                    autopct="%1.1f%%",
                    startangle=90
                )
                ax.set_title("Proporsi Positif vs Negatif Diabetes")
                st.pyplot(fig)

    elif menu == "Visualisasi":
        st.title("📈 Visualisasi Fitur Klinis")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["BMI & Status Diabetes", "Blood Pressure", "Glucose", "Korelasi"]
        )

        with tab1:
            st.subheader("Status BMI & Diabetes")
            data_bmi = df1["Status_BMI"].value_counts()

            col1, col2 = st.columns(2)
            with col1:
                st.write("Distribusi kategori BMI:")
                st.write(data_bmi.to_frame("Jumlah"))
            with col2:
                fig, ax = plt.subplots()
                ax.pie(
                    data_bmi.values,
                    labels=data_bmi.index,
                    autopct="%1.1f%%",
                    startangle=90
                )
                ax.set_title("Proporsi Status BMI")
                st.pyplot(fig)

            st.subheader("BMI vs Status Diabetes")
            data2 = df1.groupby(['Status_BMI', 'Status_Diabetes'])['Outcome'].count().unstack()
            st.write(data2)

            fig2, ax2 = plt.subplots()
            data2.plot(kind='bar', ax=ax2)
            ax2.set_title("Status Diabetes per Kategori BMI")
            ax2.set_ylabel("Jumlah")
            st.pyplot(fig2)

        with tab2:
            st.subheader("Status Blood Pressure")
            data_bp = df1["Status_Bloodpressure"].value_counts()

            col1, col2 = st.columns(2)
            with col1:
                st.write(data_bp.to_frame("Jumlah"))
            with col2:
                fig3, ax3 = plt.subplots()
                ax3.bar(data_bp.index, data_bp.values, alpha=0.7)
                ax3.set_title("Distribusi Status Blood Pressure")
                st.pyplot(fig3)

            st.subheader("Blood Pressure vs Status Diabetes")
            data4 = df1.groupby(['Status_Bloodpressure', 'Status_Diabetes'])['Outcome'].count().unstack()
            st.write(data4)

            fig4, ax4 = plt.subplots()
            data4.plot(kind='bar', ax=ax4)
            ax4.set_title("Diabetes dengan Blood Pressure")
            st.pyplot(fig4)

        with tab3:
            st.subheader("Distribusi Glucose")
            fig5, ax5 = plt.subplots()
            sns.histplot(df1["Glucose"], bins=20, kde=True, ax=ax5)
            ax5.set_title("Distribusi Glucose")
            st.pyplot(fig5)

            st.subheader("Glucose: Positif vs Negatif Diabetes")
            data_pos = df1[df1.Status_Diabetes == "Positif"]["Glucose"]
            data_neg = df1[df1.Status_Diabetes == "Negatif"]["Glucose"]

            fig6, ax6 = plt.subplots()
            ax6.hist(data_pos, bins=10, alpha=0.5, label="Positif")
            ax6.hist(data_neg, bins=10, alpha=0.5, label="Negatif")
            ax6.set_title("Perbandingan Glucose (Positif vs Negatif)")
            ax6.legend()
            st.pyplot(fig6)

        with tab4:
            st.subheader("Heatmap Korelasi Fitur")
            fig7, ax7 = plt.subplots(figsize=(10, 8))
            sns.heatmap(df_copy.corr(), annot=True, cmap="crest", ax=ax7)
            st.pyplot(fig7)

    elif menu == "Pemodelan":
        st.title("🤖 Pemodelan Machine Learning")

        results = train_models(df_copy)
        X_test = results["X_test"]
        y_test = results["y_test"]

        rf = results["rf"]
        knn = results["knn"]
        log = results["log"]

        # Ringkasan akurasi
        acc_rf = accuracy_score(y_test, results["y_pred_rf"])
        acc_knn = accuracy_score(y_test, results["y_pred_knn"])
        acc_log = accuracy_score(y_test, results["y_pred_log"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Akurasi Random Forest", f"{acc_rf:.3f}")
        col2.metric("Akurasi KNN", f"{acc_knn:.3f}")
        col3.metric("Akurasi Logistic Regression", f"{acc_log:.3f}")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Laporan Klasifikasi", "Confusion Matrix", "ROC Curve", "Feature Importance"]
        )

        with tab1:
            st.subheader("Laporan Klasifikasi (Test Set)")
            st.write("**Random Forest**")
            st.text(classification_report(y_test, results["y_pred_rf"]))
            st.write("**KNN**")
            st.text(classification_report(y_test, results["y_pred_knn"]))
            st.write("**Logistic Regression**")
            st.text(classification_report(y_test, results["y_pred_log"]))

        with tab2:
            st.subheader("Confusion Matrix")
            models = {
                "Random Forest": results["y_pred_rf"],
                "KNN": results["y_pred_knn"],
                "Logistic Regression": results["y_pred_log"]
            }

            for name, y_pred in models.items():
                st.write(f"**{name}**")
                cm = confusion_matrix(y_test, y_pred)
                st.write(pd.DataFrame(
                    cm,
                    index=["Actual 0 (Non-diabetes)", "Actual 1 (Diabetes)"],
                    columns=["Pred 0", "Pred 1"]
                ))

        with tab3:
            st.subheader("ROC Curve & AUC")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(results["fpr_rf"], results["tpr_rf"], label=f"RF (AUC = {results['auc_rf']:.3f})")
            ax.plot(results["fpr_knn"], results["tpr_knn"], label=f"KNN (AUC = {results['auc_knn']:.3f})")
            ax.plot(results["fpr_log"], results["tpr_log"], label=f"LogReg (AUC = {results['auc_log']:.3f})")
            ax.plot([0, 1], [0, 1], "k--", label="Random (0.5)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            st.pyplot(fig)

        with tab4:
            st.subheader("Feature Importance (Random Forest)")
            importances = pd.Series(rf.feature_importances_, index=df_copy.drop("Outcome", axis=1).columns)
            importances = importances.sort_values()

            fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
            importances.plot(kind="barh", ax=ax_imp)
            ax_imp.set_title("Random Forest Feature Importances")
            st.pyplot(fig_imp)

            if HAS_SMOTE and "smote_info" in results:
                st.markdown(f"""
                **Informasi Tambahan (SMOTE + RF Tuned)**  
                Akurasi model RF setelah resampling SMOTE dan tuning manual:
                **{results['smote_info']['tuned_acc']:.3f}**
                """)

    elif menu == "Prediksi Individu":
        st.title("🧍 Prediksi Risiko Diabetes (Input Individu)")

        results = train_models(df_copy)
        rf = results["rf"]  # gunakan RF sebagai model utama

        st.markdown("Masukkan nilai-nilai fitur klinis untuk melakukan prediksi:")

        col1, col2, col3 = st.columns(3)

        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
            blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)

        with col2:
            skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
            insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)

        with col3:
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
            age = st.number_input("Age", min_value=1, max_value=120, value=30)

        if st.button("Prediksi"):
            input_array = np.array([[pregnancies, glucose, blood_pressure,
                                     skin_thickness, insulin, bmi, dpf, age]])
            pred = rf.predict(input_array)[0]
            proba = rf.predict_proba(input_array)[0][1]

            if pred == 1:
                st.error(f"Hasil Prediksi: **Diabetes (Positif)** — Probabilitas ≈ {proba:.2f}")
            else:
                st.success(f"Hasil Prediksi: **Non-Diabetes (Negatif)** — Probabilitas ≈ {proba:.2f}")

    elif menu == "Tentang Data":
        st.title("ℹ️ Tentang Dataset & Proyek")

        st.markdown("""
        - Dataset berisi data klinis terkait risiko diabetes (misalnya: **PIMA Indians Diabetes** atau sejenis).
        - Fitur mencakup: *Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
          BMI, DiabetesPedigreeFunction, Age,* serta **Outcome** sebagai label target.
        - Analisis pada dashboard ini mencakup:
          - Exploratory Data Analysis (EDA)
          - Penanganan missing value dengan imputasi median
          - Visualisasi distribusi fitur dan hubungan dengan status diabetes
          - Pemodelan dengan **Random Forest, KNN, dan Logistic Regression**
          - Evaluasi model menggunakan akurasi, confusion matrix, dan ROC–AUC
        """)

        st.caption("Dashboard disusun kembali dari skrip analisis Anda menggunakan Streamlit.")
