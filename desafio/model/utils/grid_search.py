from sklearn.model_selection import GridSearchCV


def grid_search_estimator(estimator, params, cross_validation_split, scoring, verbose, n_jobs, x_train, y_train):
    grid_search = GridSearchCV(estimator,
                               params,
                               cv=cross_validation_split,
                               scoring=scoring,
                               verbose=verbose,
                               n_jobs=n_jobs
                               )

    grid_search.fit(x_train, y_train)

    return grid_search.best_estimator_
