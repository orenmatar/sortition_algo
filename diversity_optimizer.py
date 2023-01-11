import pandas as pd
import numpy as np
import os
import itertools
from scipy.stats import gmean
import mip
from mip import MAXIMIZE, CBC, OptimizationStatus, MINIMIZE, BINARY, xsum, maximize, minimize
from sklearn.preprocessing import OneHotEncoder


class DiversityOptimizer(object):
    MAX_DIMS_INTERSECTION_N_TO_OPTIMIZE = 100

    def __init__(self, df):
        self.df = df
        self.data, self.all_dims_combs = self.prepare_all_data()
        self.all_ohe = self.create_all_one_hot_encodings()

    def prepare_dims_data(self, dims):
        all_cat_combs = ['__'.join(x) for x in itertools.product(*[self.df[d].unique() for d in dims])]
        peoples_ids = self.df.loc[:, dims].apply(lambda row: '__'.join([row[d] for d in dims]), axis=1).values
        return {'all_cat_combs': all_cat_combs, 'peoples_ids': peoples_ids}

    def prepare_all_data(self):
        all_dims_combs = [itertools.combinations(self.df.columns, r=i) for i in range(1, self.df.shape[1] + 1)]
        all_dims_combs = [x for y in all_dims_combs for x in y]
        return {dims: self.prepare_dims_data(dims) for dims in all_dims_combs}, all_dims_combs

    def peoples_idx_to_val(self, idx):
        """
        Calculates the geometric mean of the counts for each dimension and intersections
        """
        res = []
        for dims in self.all_dims_combs:
            peoples_ids = self.data[dims]['peoples_ids'][idx]  # the ids of the categories for all of the individuals
            all_cat_combs = self.data[dims][
                'all_cat_combs']  # all possible category combinations for this intersection. add to make > 0 for all
            counts = np.unique(np.concatenate([peoples_ids, all_cat_combs]), return_counts=True)[1]
            res.append(gmean(counts))
        return res

    def create_all_one_hot_encodings(self):
        """
        For every set of dimensions - one hot encode who is in which intersection.
        Rows are people. Columns are the different intersection
        """
        all_ohe = []
        for dims in self.all_dims_combs:
            data = self.data[dims]
            possible_profiles = data['all_cat_combs']
            # only up to a certain amount of intersections. Each intersection adds a variable,
            # makes the optimization more difficult.
            if len(possible_profiles) > DiversityOptimizer.MAX_DIMS_INTERSECTION_N_TO_OPTIMIZE:
                continue

            peoples_ids = data['peoples_ids']
            ohe = OneHotEncoder(categories=[possible_profiles], sparse=False)
            ohe = ohe.fit_transform(peoples_ids.reshape(-1, 1))
            all_ohe.append(ohe)
        return all_ohe

    def optimize(self, categories, panel_size):
        """
        Uses MIP to optimize based on the categories constraints

        For the optimization goal, fo every dims intersection:
        Take the one hot encoded of who is in which intersection for these dims
        Take the binary vector of who is selected and multiply and sum to get the sizes of each intersection of categories
        Figure out the "best" value - if all intersections were of equal size
        Take the abs for each intersection from that value
        Minimize that abs
        """
        df = self.df
        # for this to work you must have a gurubi license under env variable GRB_LICENSE_FILE.
        # Otherwise specify mip solver_name
        m = mip.Model()
        # binary variable for each person - if they are selected or not
        model_variables = pd.Series([m.add_var(var_type='B', name=str(x)) for x in df.index], index=df.index)

        # the sum of all people in each category must be between the min and max specified
        for dim, d in categories.items():
            for cat, const in d.items():
                relevant = model_variables[df[dim] == cat]
                rel_sum = xsum(relevant)
                m.add_constr(rel_sum >= const['min'])
                m.add_constr(rel_sum <= const['max'])
        m.add_constr(xsum(model_variables) == panel_size)  # cannot exceed panel size

        # define the optimization goal
        all_objectives = []
        for ohe in self.all_ohe:  # for every set of dims
            intersection_sizes = (model_variables.values.reshape(-1, 1) * ohe).sum(axis=0)
            best_val = panel_size / ohe.shape[1]  # if all intersections were equal size

            # set support variables that are the diffs from each intersection size to the most equal value
            diffs_from_best_val = [m.add_var(var_type='C') for x in intersection_sizes]
            # constrain these support variables to be the abs diff from the intersection size
            for abs_diff, intersection_size in zip(diffs_from_best_val, intersection_sizes):
                m.add_constr(abs_diff >= (intersection_size - best_val))
                m.add_constr(abs_diff >= (best_val - intersection_size))

            support_vars_sum = xsum(diffs_from_best_val)  # we will minimize the abs diffs
            all_objectives.append(support_vars_sum)

        obj = xsum(all_objectives)
        m.objective = minimize(obj)
        m.optimize()  # add assert that it was a success?
        selected = pd.Series([v.x for v in m.vars][:len(df)]) == 1
        return m, selected


if __name__ == '__main__':
    # generate data
    # generate any data
    n = 364
    panel_size = 40

    # cat names, probs in population, response rate for each cat
    dimensions = {
        'gender': (['male', 'female'], [0.5, 0.5], [0.1, 0.07]),
        'age_bucket': (
        ['18-25', '26-35', '36-50', '51-65', '66+'], [.23, .18, .26, .17, .16], [0.05, 0.08, 0.05, 0.1, 0.14]),
        'income': (['below median', 'around median', 'above median'], [0.3, 0.4, 0.3], [0.05, 0.07, 0.12]),
        'settlement type': (['urban', 'suburban', 'rural'], [0.25, 0.6, 0.15], [0.13, 0.1, 0.06]),
    }
    populations_dfs = {dim: pd.DataFrame(vals[1:], columns=vals[0], index=['%population', '%response']) for dim, vals in
                       dimensions.items()}

    df = pd.DataFrame()
    categories = {}
    for dim_name, (vals, probs, response_rates) in dimensions.items():
        sample_probs = np.array(probs) * response_rates
        sample_probs = sample_probs / sample_probs.sum()
        df[dim_name] = np.random.choice(vals, n, p=sample_probs)
        dim_d = {}
        total = panel_size
        for i, (cat, p) in enumerate(zip(vals, probs)):
            s = int(round(panel_size * p))
            total -= s
            if i == len(vals) - 1:  # if last category
                s += total  # add whatever is left, so sum of cats will be the panel size
            dim_d[cat] = {'min': s, 'max': s}
        categories[dim_name] = dim_d

    div_optimizer = DiversityOptimizer(df)
    m, selected = div_optimizer.optimize(categories, panel_size)
    print(m.status)
    print(selected)
