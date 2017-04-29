'''
Count-combination features in Pandas. 
Intended for use on Amazon access data.
'''
import pandas as pd
import itertools as it


def get_counts(df, fields):
    '''
    Create value-count features for combinations of fields in a data frame.
    Counts are taken over combinations of the values in fields.

    Args:
        df: input data frame
        fields: list of column names in df

    Returns:
        1-column data frame of counts of combinations of values in fields
    '''
    gp = df.groupby(fields)
    field_name = '_'.join(fields + ['ct'])
    return pd.DataFrame({field_name: gp.size()})


def apply_counts(df, cts):
    '''
    Applies counts of combinations of values to a data frame.
    If combinations occur in df that were not in cts, these are 0-filled.

    Args:
        df: input data frame
        cts: 1-column data frame of counts of occurrances of combinations

    Returns:
        the input data frame, df, with the counts joined on
    '''
    fields = cts.index.names
    out = df.merge(cts, how='left', left_on=fields, right_index=True)
    ct_field = cts.columns[0]
    return out[ct_field].fillna(0).astype(int)


def combo_features(df_in, df_out, n):
    '''
    Produces count features for all n-combinations of columns in df_in.
    Applies them to df_out (without changing df_out). The frames df_in and df_out 
    can be the same, in which case this is like fit_transform from the sklearn API.
    NB: df_in and df_out should have the same column names.

    Args:
        df_in: input data frame for collecting counts
        df_out: data frame to which counts are applied
        n_val: int. Produces all n-combinations of the columns.

    Returns:
        Pandas DataFrame with the counts applied to df_out.
    '''
    out = pd.DataFrame(index=df_out.index)
    field_combos = it.combinations(df_in.columns, n)
    for fields in field_combos:
        cts = get_counts(df_in, list(fields))
        ct_field = cts.columns[0]
        out[ct_field] = apply_counts(df_out, cts)
    return out
    

def range_combo(df_in, df_out, n_vals):
    '''
    Computes value-count features for k-combinations of columns in df_in/df_out
    for all k in n_vals.

    Args:
        df_in: input data frame for collecting counts
        df_out: data frame to which counts are applied
        n_vals: list of int. Produces counts for k-combinations 
                of the columns for all k in n_vals

    Returns:
        Pandas DataFrame with the counts applied to df_out.
    '''
    chunks = [combo_features(df_in, df_out, n) for n in n_vals]
    return pd.concat(chunks, axis=1)
