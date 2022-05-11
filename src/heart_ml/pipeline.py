from sklearn.linear_model import LogisticRegression
from sklearn import tree


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(
    use_scaler: bool, max_iter: int, logreg_C: float, random_state: int, max_depth: int, min_samples_split:int or float
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
                              ),
            tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split
                                        )
        )
    )
    return Pipeline(steps=pipeline_steps)
