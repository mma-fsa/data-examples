import pandas as pd, numpy as np
import sklearn as sk, xgboost as xgb
import json

import scipy.sparse as sp

def convert_to_dataframe(matrix, column_names=None):
    # Check if the matrix is a SciPy sparse format
    if sp.issparse(matrix):
        return pd.DataFrame.sparse.from_spmatrix(matrix, columns=column_names)
    
    # Otherwise, treat it as a dense array or list
    return pd.DataFrame(matrix, columns=column_names)

class PoissonDecisionTree():
    
    def __init__(
        self,
        actual_col,
        offset_col,
        params=None,
        monotone_constraints=None,
        interaction_constraints=None,
    ):
        self.actual_col = actual_col
        self.offset_col = offset_col
        self._params = {
            'objective': 'count:poisson',
            # 'hist' (unlike 'exact') supports native categorical splits.
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'max_depth': 3,
            'min_child_weight': 10,
            'gamma': 1.0,
            'seed': 0,
        }
        if params is not None:
            self._params.update(params)
        self._monotone_constraints = monotone_constraints
        self._interaction_constraints = interaction_constraints
        self._preproc = None
        self._bst = None
        self._feature_columns = None
        self._str_cols = None
        self._categorical_categories = None
        self._feature_names = None
        self._feature_types = None

    def _require_feature_names(self):
        if self._feature_names is None:
            raise ValueError("Model features are not available before fit().")
        return self._feature_names

    def _resolve_feature_index(self, feature_name, feature_index_map):
        feature_names = self._require_feature_names()
        if isinstance(feature_name, (int, np.integer)):
            idx = int(feature_name)
            if idx < 0 or idx >= len(feature_names):
                raise ValueError(
                    f"Feature index out of bounds for constraints: {feature_name}"
                )
            return idx

        if feature_name not in feature_index_map:
            raise ValueError(
                f"Unknown feature in constraints: {feature_name}. "
                f"Available features: {feature_names}"
            )
        return feature_index_map[feature_name]

    def _build_monotone_constraints_param(self):
        if self._monotone_constraints is None:
            return None
        if isinstance(self._monotone_constraints, str):
            return self._monotone_constraints

        feature_names = self._require_feature_names()
        feature_index_map = {name: i for i, name in enumerate(feature_names)}
        n_features = len(feature_names)

        if isinstance(self._monotone_constraints, dict):
            values = [0] * n_features
            for feature_name, direction in self._monotone_constraints.items():
                idx = self._resolve_feature_index(feature_name, feature_index_map)
                direction_int = int(direction)
                if direction_int not in (-1, 0, 1):
                    raise ValueError(
                        f"Monotone constraint for {feature_name} must be -1, 0, or 1."
                    )
                values[idx] = direction_int
            return "(" + ",".join(str(v) for v in values) + ")"

        values = [int(v) for v in self._monotone_constraints]
        if len(values) != n_features:
            raise ValueError(
                "Monotone constraints length must equal number of model features. "
                f"Got {len(values)} constraints for {n_features} features."
            )
        bad_values = [v for v in values if v not in (-1, 0, 1)]
        if bad_values:
            raise ValueError("Monotone constraints must only contain -1, 0, or 1 values.")

        return "(" + ",".join(str(v) for v in values) + ")"

    def _build_interaction_constraints_param(self):
        if self._interaction_constraints is None:
            return None
        if isinstance(self._interaction_constraints, str):
            return self._interaction_constraints

        feature_names = self._require_feature_names()
        feature_index_map = {name: i for i, name in enumerate(feature_names)}

        groups = []
        raw_constraints = self._interaction_constraints

        if isinstance(raw_constraints, dict):
            for key, include_group in raw_constraints.items():
                if not include_group:
                    continue
                if isinstance(key, (tuple, list)):
                    groups.append(list(key))
                else:
                    groups.append([key])
        else:
            groups = list(raw_constraints)

        parsed_groups = []
        for group in groups:
            if isinstance(group, (str, int, np.integer)):
                group_values = [group]
            else:
                group_values = list(group)

            indices = [
                self._resolve_feature_index(feature_name, feature_index_map)
                for feature_name in group_values
            ]
            parsed_groups.append(indices)

        return json.dumps(parsed_groups)
        
    def _prepare_input_df(self, df : pd.DataFrame) -> pd.DataFrame:
        if self._preproc is None:
            raise ValueError("Model must be fit before preparing input data.")
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        if self._feature_columns is None:
            raise ValueError("No fitted feature columns are available.")

        missing = [c for c in self._feature_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns for prediction: {missing}")

        df_for_model = df.loc[:, self._feature_columns].copy()

        if self._str_cols:
            for col in self._str_cols:
                if col not in df_for_model.columns:
                    continue
                if self._categorical_categories is not None and col in self._categorical_categories:
                    categories = self._categorical_categories[col]
                    df_for_model[col] = pd.Categorical(
                        df_for_model[col].astype("string"),
                        categories=categories,
                    )
                else:
                    df_for_model[col] = df_for_model[col].astype("category")
                df_for_model[col] = df_for_model[col].cat.codes.astype("int8")

        return df_for_model
        
    def fit(self, df : pd.DataFrame):
        
        if not self.actual_col in df.columns:
            raise Exception(f"Missing column (for actuals): {self.actual_col}")
        if not self.offset_col in df.columns:
            raise Exception(f"Missing column (for offset): {self.offset_col}")
        
        response_cols = [
            self.actual_col,
            self.offset_col
        ]

        self._str_cols = [
            c for c in df.select_dtypes(
                include=['object', 'string', 'category']
            ).columns.tolist()
            if c not in response_cols
        ]
        self._feature_columns = [
            c for c in df.columns
            if c not in response_cols
        ]

        num_cols = [
            c for c in self._feature_columns
            if c not in self._str_cols
        ]

        # Encode categorical columns as integer category codes; combined with
        # feature_types="c" and enable_categorical=True below, XGBoost treats
        # these as true categorical splits (arbitrary category subsets) rather
        # than ordinal numeric thresholds over the codes.
        df_for_model = df.copy()
        self._categorical_categories = {}
        for col in self._str_cols:
            if col in df_for_model.columns:
                ser = df_for_model[col].astype("category")
                self._categorical_categories[col] = list(ser.cat.categories)
                df_for_model[col] = ser.cat.codes.astype("int8")

        xfrm = [
            (
                "num", "passthrough", num_cols + self._str_cols
            )
        ]
        
        self._preproc = sk.compose.ColumnTransformer(
            transformers=xfrm,
            remainder="drop",
            verbose_feature_names_out=False
        )
        
        X_mat = self._preproc.fit_transform(df_for_model)
        offset = df_for_model[self.offset_col].to_numpy("float64")
        y = df_for_model[self.actual_col].to_numpy("float64")
        self._feature_names = self._preproc.get_feature_names_out().tolist()
        self._feature_types = [
            "c" if c in self._str_cols else "q"
            for c in self._feature_names
        ]
        xgb_mat = xgb.DMatrix(
            data=X_mat,
            label=y,
            base_margin = np.log(offset),
            feature_names=self._feature_names,
            feature_types=self._feature_types,
            enable_categorical=True,
        )
        df_X_mat = convert_to_dataframe(
            X_mat,
            column_names=self._feature_names
        ) 
        
        train_params = dict(self._params)

        monotone_constraints = self._build_monotone_constraints_param()
        if monotone_constraints is not None:
            train_params["monotone_constraints"] = monotone_constraints

        interaction_constraints = self._build_interaction_constraints_param()
        if interaction_constraints is not None:
            train_params["interaction_constraints"] = interaction_constraints

        self._bst = xgb.train(
            params = train_params,
            dtrain=xgb_mat,
            num_boost_round=1
        )
        
        tree_json = json.loads(
            self._bst.get_dump(dump_format="json")[0])
        
        self._tree_data = collect_split_stats(
            tree_json, 
            df_X_mat,
            y, 
            offset,
            categorical_categories=self._categorical_categories)
        
        self._tree_df = pd.DataFrame(self._tree_data)

    def predict(self, df : pd.DataFrame):
        if self._bst is None or self._preproc is None:
            raise ValueError("Model must be fit before calling predict().")

        df_for_model = self._prepare_input_df(df)
        X_mat = self._preproc.transform(df_for_model)

        offset = np.ones(len(df_for_model), dtype="float64")
        if self.offset_col in df.columns:
            offset = df[self.offset_col].to_numpy("float64")

        xgb_mat = xgb.DMatrix(
            data=X_mat,
            feature_names=self._feature_names,
            feature_types=self._feature_types,
            enable_categorical=True,
            base_margin=np.log(offset),
        )
        return self._bst.predict(xgb_mat)
        
    def __str__(self):
        return pretty_print_compact_tree(self._tree_df)
        
    def print_tree(self):
        print(str(self))
        
    def print_nodes(self, digits=4) -> str:
        print(
            pretty_print_split_stats(
                self._tree_df,
                digits=digits
            )
        )

# -------------------------------------------------
# 3) Helpers
# -------------------------------------------------
def poisson_deviance(y, mu):
    """
    Exact Poisson deviance:
      2 * sum( y*log(y/mu) - (y - mu) )
    with y*log(y/mu) defined as 0 when y == 0.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if y.size == 0 or mu.size == 0:
        return 0.0
    if np.any(mu <= 0):
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * np.sum(term - (y - mu))


def node_mu_with_offset(y_node, exposure_node):
    """
    Poisson MLE in a node with log(exposure) as fixed offset:
      mu_i = exposure_i * rate_hat
      rate_hat = sum(y) / sum(exposure)
    """
    exposure_node = np.asarray(exposure_node, dtype=float)
    if exposure_node.size == 0:
        return np.array([], dtype=float)

    total_exp = np.sum(exposure_node)
    if total_exp <= 0:
        return np.zeros_like(exposure_node, dtype=float)

    rate_hat = np.sum(y_node) / total_exp
    # keep strictly positive to avoid numerical issues when all y == 0
    rate_hat = max(rate_hat, 1e-15)
    return exposure_node * rate_hat


def split_mask(x_col, split_value, missing_go_to, yes_child_id, no_child_id, missing_child_id):
    """
    XGBoost JSON tree nodes use:
      condition
      split
      yes / no / missing
    For numeric splits, 'yes' is the branch when feature < condition.
    Missing values go to 'missing'.
    """
    is_missing = pd.isna(x_col).to_numpy()
    go_yes = (~is_missing) & (x_col.to_numpy() < split_value)
    go_missing = is_missing

    left = np.zeros(len(x_col), dtype=bool)
    right = np.zeros(len(x_col), dtype=bool)

    # Map yes/no/missing onto left/right according to children ordering
    # We identify left/right by matching child nodeids.
    left_child_id = yes_child_id if yes_child_id in (yes_child_id, no_child_id, missing_child_id) else None
    # But more robustly we'll infer from actual child list outside this helper.

    return go_yes, go_missing


def _category_level_names(feature, codes, categorical_categories):
    """
    Translate categorical split codes (as stored in XGBoost's split_condition)
    into their original category level names, falling back to the raw code
    when no mapping is available.
    """
    levels = None
    if categorical_categories is not None:
        levels = categorical_categories.get(feature)

    names = []
    for code in codes:
        code_int = int(code)
        if levels is not None and 0 <= code_int < len(levels):
            names.append(str(levels[code_int]))
        else:
            names.append(str(code_int))
    return names


def collect_split_stats(
    node,
    X_sub,
    y_sub,
    exposure_sub,
    path_conditions=None,
    show_missing=False,
    categorical_categories=None,
):
    """
    Recursively compute exact Poisson deviance reduction for each split,
    while storing a human-readable chain of split conditions.

    Parameters
    ----------
    node : dict
        Parsed XGBoost JSON node.
    X_sub : pd.DataFrame
        Data reaching this node.
    y_sub : np.ndarray
        Counts reaching this node.
    exposure_sub : np.ndarray
        Exposure reaching this node.
    path_conditions : list[str] | None
        List of human-readable conditions defining the node.
    categorical_categories : dict[str, list] | None
        Maps a categorical feature name to its ordered list of category
        level names (index == category code), used to render categorical
        splits with their actual level names instead of raw codes.

    Returns
    -------
    list[dict]
    """
    if path_conditions is None:
        path_conditions = []

    results = []

    if "children" not in node:
        return results

    feature = node["split"]
    threshold = node["split_condition"]
    yes_id = node["yes"]
    no_id = node["no"]
    missing_id = node["missing"]

    children = node["children"]
    left_child = children[0]
    right_child = children[1]
    left_id = left_child["nodeid"]
    right_id = right_child["nodeid"]

    # A categorical split's condition is a list of category codes that take
    # the "yes" branch; a numeric split's condition is a single threshold.
    is_categorical_split = isinstance(threshold, list)

    xcol = X_sub[feature]
    is_missing = pd.isna(xcol).to_numpy()
    missing_text = f"{feature} missing"

    if is_categorical_split:
        go_yes = (~is_missing) & np.isin(xcol.to_numpy(), threshold)
        level_names = _category_level_names(feature, threshold, categorical_categories)
        if len(level_names) == 1:
            yes_text = f"{feature} = {level_names[0]}"
            no_text = f"{feature} != {level_names[0]}"
        else:
            levels_joined = ", ".join(level_names)
            yes_text = f"{feature} in {{{levels_joined}}}"
            no_text = f"{feature} not in {{{levels_joined}}}"
        split_label = yes_text
    else:
        go_yes = (~is_missing) & (xcol.to_numpy() < threshold)
        yes_text = f"{feature} < {threshold}"
        no_text = f"{feature} >= {threshold}"
        split_label = yes_text

    go_no = (~is_missing) & (~go_yes)
    go_missing = is_missing

    left_mask = np.zeros(len(X_sub), dtype=bool)
    right_mask = np.zeros(len(X_sub), dtype=bool)

    if yes_id == left_id:
        left_mask |= go_yes
    elif yes_id == right_id:
        right_mask |= go_yes
    else:
        raise ValueError("yes child id not found among node children")

    if no_id == left_id:
        left_mask |= go_no
    elif no_id == right_id:
        right_mask |= go_no
    else:
        raise ValueError("no child id not found among node children")

    if missing_id == left_id:
        left_mask |= go_missing
    elif missing_id == right_id:
        right_mask |= go_missing
    else:
        raise ValueError("missing child id not found among node children")

    # If missing follows the same side as yes or no, merge text for readability
    left_conditions = list(path_conditions)
    right_conditions = list(path_conditions)

    left_parts = []
    right_parts = []

    if yes_id == left_id:
        left_parts.append(yes_text)
    if no_id == left_id:
        left_parts.append(no_text)
    if missing_id == left_id and show_missing:
        left_parts.append(missing_text)

    if yes_id == right_id:
        right_parts.append(yes_text)
    if no_id == right_id:
        right_parts.append(no_text)
    if missing_id == right_id and show_missing:
        right_parts.append(missing_text)

    if left_parts:
        left_item = "(" + " OR ".join(left_parts) + ")" if len(left_parts) > 1 else left_parts[0]
        left_conditions.append(left_item)
    if right_parts:
        right_item = "(" + " OR ".join(right_parts) + ")" if len(right_parts) > 1 else right_parts[0]
        right_conditions.append(right_item)

    mu_parent = node_mu_with_offset(y_sub, exposure_sub)
    dev_parent = poisson_deviance(y_sub, mu_parent)

    y_left = y_sub[left_mask]
    y_right = y_sub[right_mask]
    exp_left = exposure_sub[left_mask]
    exp_right = exposure_sub[right_mask]

    mu_left = node_mu_with_offset(y_left, exp_left)
    mu_right = node_mu_with_offset(y_right, exp_right)

    dev_left = poisson_deviance(y_left, mu_left)
    dev_right = poisson_deviance(y_right, mu_right)
    dev_children = dev_left + dev_right
    dev_reduction = dev_parent - dev_children

    sum_exp_left = float(np.sum(exp_left))
    sum_exp_right = float(np.sum(exp_right))

    results.append({
        "nodeid": node["nodeid"],
        "split_label": split_label,
        "path_pretty": " AND ".join(path_conditions) if path_conditions else "ROOT",
        "left_path_pretty": " AND ".join(left_conditions),
        "right_path_pretty": " AND ".join(right_conditions),
        "feature": feature,
        "threshold": threshold,
        "n_parent": len(y_sub),
        "sum_y_parent": float(np.sum(y_sub)),
        "sum_exp_parent": float(np.sum(exposure_sub)),
        "ae_parent": float(np.sum(y_sub)) / float(np.sum(exposure_sub)) if np.sum(exposure_sub) > 0 else np.nan,
        "dev_parent": float(dev_parent),
        "n_left": int(left_mask.sum()),
        "sum_y_left": float(np.sum(y_left)),
        "sum_exp_left": sum_exp_left,
        "ae_left": float(np.sum(y_left)) / sum_exp_left if sum_exp_left > 0 else np.nan,
        "dev_left": float(dev_left),
        "n_right": int(right_mask.sum()),
        "sum_y_right": float(np.sum(y_right)),
        "sum_exp_right": sum_exp_right,
        "ae_right": float(np.sum(y_right)) / sum_exp_right if sum_exp_right > 0 else np.nan,
        "dev_right": float(dev_right),
        "dev_children": float(dev_children),
        "dev_reduction": float(dev_reduction),
        "xgb_gain": float(node.get("gain", np.nan)),
        "xgb_cover": float(node.get("cover", np.nan)),
    })

    results.extend(
        collect_split_stats(
            left_child,
            X_sub.iloc[left_mask].copy(),
            y_left,
            exp_left,
            left_conditions,
            categorical_categories=categorical_categories,
        )
    )

    results.extend(
        collect_split_stats(
            right_child,
            X_sub.iloc[right_mask].copy(),
            y_right,
            exp_right,
            right_conditions,
            categorical_categories=categorical_categories,
        )
    )

    return results

def pretty_print_split_stats(split_df, digits=4, sort_by_node=True):
    import pandas as pd

    if split_df is None or len(split_df) == 0:
        return "No splits found."

    df = split_df.copy()

    if sort_by_node and "nodeid" in df.columns:
        df = df.sort_values(["nodeid"]).reset_index(drop=True)

    def fmt(x):
        if pd.isna(x):
            return "NA"
        if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer()):
            return f"{int(x)}"
        if isinstance(x, float):
            return f"{x:,.{digits}f}"
        return str(x)

    lines = []

    line = "=" * 120
    lines.append(line)
    lines.append("POISSON SPLIT SUMMARY")
    lines.append(line)

    for _, row in df.iterrows():
        lines.append(f"Node {row['nodeid']}")
        lines.append(f"  Path: {row['path_pretty']}")
        lines.append(f"  Split: {row['split_label']}")

        lines.append("  Parent")
        lines.append(
            f"    n={fmt(row.get('n_parent'))}  "
            f"actual={fmt(row.get('sum_y_parent'))}  "
            f"exp={fmt(row.get('sum_exp_parent'))}  "
            f"ae={fmt(row.get('ae_parent'))}  "
            f"dev={fmt(row.get('dev_parent'))}"
        )

        lines.append("  Left child")
        lines.append(f"    path: {row['left_path_pretty']}")
        lines.append(
            f"    n={fmt(row.get('n_left'))}  "
            f"actual={fmt(row.get('sum_y_left'))}  "
            f"exp={fmt(row.get('sum_exp_left'))}  "
            f"ae={fmt(row.get('ae_left'))}  "
            f"dev={fmt(row.get('dev_left'))}"
        )

        lines.append("  Right child")
        lines.append(f"    path: {row['right_path_pretty']}")
        lines.append(
            f"    n={fmt(row.get('n_right'))}  "
            f"actual={fmt(row.get('sum_y_right'))}  "
            f"exp={fmt(row.get('sum_exp_right'))}  "
            f"ae={fmt(row.get('ae_right'))}  "
            f"dev={fmt(row.get('dev_right'))}"
        )

        lines.append("  Improvement")
        lines.append(
            f"    children_dev={fmt(row.get('dev_children'))}  "
            f"dev_reduction={fmt(row.get('dev_reduction'))}  "
            f"xgb_gain={fmt(row.get('xgb_gain'))}"
        )
        lines.append("-" * 120)

    return "\n".join(lines)

def pretty_print_compact_tree(split_stats, digits=4, sort_children=True):
    """
    Render a compact tree string from the output of collect_split_stats(...).

    Accepts either:
      - list[dict] returned by collect_split_stats(...)
      - pd.DataFrame built from that list

    Example output:

    root: ae=1.0000   y=10000
      - foo < 12: ae=1.2000   y=4234
        - thud < 3: ae=0.9000   y=100
        - thud >= 3: ae=1.3000   y=402
      - foo >= 12: ae=0.8000   y=5766
    """
    import pandas as pd

    if split_stats is None:
        return "No splits found."

    if isinstance(split_stats, pd.DataFrame):
        df = split_stats.copy()
    else:
        df = pd.DataFrame(split_stats)

    if len(df) == 0:
        return "No splits found."

    def fmt(x):
        if pd.isna(x):
            return "NA"
        if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer()):
            return str(int(x))
        if isinstance(x, float):
            return f"{x:,.{digits}f}"
        return str(x)

    def path_depth(path):
        if path == "ROOT":
            return 0
        return len(str(path).split(" AND "))

    def last_condition(path):
        if path == "ROOT":
            return "root"
        parts = str(path).split(" AND ")
        return parts[-1]

    # One row per internal node, indexed by its parent node path
    rows_by_path = {row["path_pretty"]: row for _, row in df.iterrows()}

    def build_line(label, ae, y, depth):
        prefix = "  " * depth
        if depth == 0:
            return f"{prefix}{label}: ae={fmt(ae)}   y={fmt(y)}"
        return f"{prefix}- {label}: ae={fmt(ae)}   y={fmt(y)}"

    lines = []

    # Root summary comes from the row whose path_pretty == ROOT
    root_row = rows_by_path.get("ROOT")
    if root_row is None:
        # Fallback: shallowest row
        root_row = df.sort_values("path_pretty", key=lambda s: s.map(path_depth)).iloc[0]

    lines.append(build_line("root", root_row.get("ae_parent"), root_row.get("sum_y_parent"), 0))

    visited = set()

    def walk(parent_path, depth):
        if parent_path in visited:
            return
        visited.add(parent_path)

        row = rows_by_path.get(parent_path)
        if row is None:
            return

        children = [
            {
                "path": row["left_path_pretty"],
                "label": last_condition(row["left_path_pretty"]),
                "ae": row.get("ae_left"),
                "y": row.get("sum_y_left"),
            },
            {
                "path": row["right_path_pretty"],
                "label": last_condition(row["right_path_pretty"]),
                "ae": row.get("ae_right"),
                "y": row.get("sum_y_right"),
            },
        ]

        if sort_children:
            children = sorted(children, key=lambda x: (path_depth(x["path"]), x["label"]))

        for child in children:
            child_y = child.get("y")
            if child_y is None:
                continue
            try:
                if float(child_y) <= 0:
                    continue
            except (TypeError, ValueError):
                pass
            lines.append(build_line(child["label"], child["ae"], child_y, depth))
            # Recurse only if this child path is itself a parent node somewhere
            if child["path"] in rows_by_path:
                walk(child["path"], depth + 1)

    walk("ROOT", 1)
    return "\n".join(lines)
