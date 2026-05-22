import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram(df, column):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[column].dropna(),
        kde=True
    )

    plt.title(f"Distribution of {column}")

    plt.show()


def plot_boxplot(df, column):

    plt.figure(figsize=(8, 5))

    sns.boxplot(x=df[column])

    plt.title(f"Boxplot of {column}")

    plt.show()


def plot_correlation(df, columns):

    corr = df[columns].corr()

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Matrix")

    plt.show()