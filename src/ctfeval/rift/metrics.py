from scipy.stats import spearmanr, pearsonr


def pearson(theory, real):
    return pearsonr(theory, real).statistic


def spearman(theory, real):
    return spearmanr(theory, real).statistic


METRICS = {
    "spearman": spearman,
    "pearson": pearson,
}
