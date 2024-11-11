import pandas as pd

from experiments import LogRegExperiment, KNeighborsExperiment, DecisionTreeExperiment,\
    MultilayerPerceptronExperiment, SupportVectorMachineExperiment
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime


def run_experiments(categorical_columns, target, df, ds_name):
    df_train, df_test = train_test_split(df, random_state=42, stratify=df[target])
    metadata = {'categorical_cols': categorical_columns, 'target': target}
    experiments = [LogRegExperiment(df_train, df_test, metadata),
                   KNeighborsExperiment(df_train, df_test, metadata),
                   DecisionTreeExperiment(df_train, df_test, metadata),
                   MultilayerPerceptronExperiment(df_train, df_test, metadata),
                   SupportVectorMachineExperiment(df_train, df_test, metadata)]

    experiment_results = {}

    for experiment in experiments:
        print(f"{datetime.datetime.now()} Starting {experiment}")
        results = experiment.run()
        print(f"{datetime.datetime.now()} Finalized {experiment}")
        experiment_results = {**experiment_results, **results}

    x = [0, 1, 2, 3]
    for i, experiment in enumerate(experiment_results.keys()):
        # Print results
        print(f"{experiment}'s Performance:")
        print(experiment_results[experiment]['scoring'])

        # Plot results
        y = [experiment_results[experiment]['scoring']['accuracy'],
             experiment_results[experiment]['scoring']['weighted avg']['precision'],
             experiment_results[experiment]['scoring']['weighted avg']['f1-score'],
             experiment_results[experiment]['scoring']['weighted avg']['recall']]
        plt.plot(x, y, 'o', label=experiment)

        plt.ylabel('Performance in %')
        plt.title(f'Baseline Performance on {ds_name}')

    y = ['Accuracy', 'Precision_weighted', 'F1-Score_weighted', 'Recall_weighted']
    plt.xticks(x, y)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.savefig(f"BaselinePerformance{ds_name}Raw.png", bbox_inches="tight")
    # plt.savefig(f"BaselinePerformance{ds_name}Standardization.png", bbox_inches="tight")
    # plt.savefig(f"BaselinePerformance{ds_name}Normalization.png", bbox_inches="tight")
    # plt.savefig(f"BaselinePerformance{ds_name}OptimalPreprocessing.png", bbox_inches="tight")
    # plt.show() is not useful as it does not adjust the figure size to the plots AND the legend

    plt.clf()


def main() -> None:
    """
    This method is intended to execute defined experiments.
    """

    # Loading a clean dataset
    column_names = ["Wife's age", "Wife's education", "Husband's education", "Number of children", "Wife's religion",
                    "Wife's now working?", "Husband's occupation", "Standard-of-living index", "Media exposure",
                    "Contraceptive method used"]
    df_cmc = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data", names=column_names,
                         header=None)
    df_credit = pd.read_csv("../data/clean/SouthGermanCredit.csv")
    df_telco = pd.read_csv("../data/clean/TelcoCustomerChurn.csv")

    # Run baseline experiments
    categorical_columns_credit = ["status", "credit_history", "purpose", "savings", "employment_duration",
                                  "installment_rate", "personal_status_sex", "other_debtors", "present_residence",
                                  "property", "other_installment_plans", "housing", "number_credits", "job",
                                  "people_liable", "telephone", "foreign_worker"]
    categorical_columns_cmc = ["Wife's education", "Husband's education", "Wife's religion", "Wife's now working?",
                               "Husband's occupation", "Standard-of-living index", "Media exposure"]

    categorical_cols_telco = ["PaymentMethod", "PaperlessBilling", "Contract", "StreamingMovies", "StreamingTV",
                              "TechSupport", "DeviceProtection", "OnlineBackup", "OnlineSecurity", "InternetService",
                              "MultipleLines", "PhoneService", "Dependents", "Partner", "SeniorCitizen", "gender"]

    run_experiments(categorical_columns_cmc, 'Contraceptive method used', df_cmc, 'ContraceptiveMethodChoice')
    run_experiments(categorical_columns_credit, 'credit_risk', df_credit, 'SouthGermanCredit')
    run_experiments(categorical_cols_telco, 'Churn', df_telco, 'TelcoCustomerChurn')


if __name__ == "__main__":
    main()
