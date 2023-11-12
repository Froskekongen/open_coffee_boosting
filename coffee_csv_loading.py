import typing as tp
import functools
import re
import numpy as np
import pandas as pd


def hasstring(ss: tp.Iterator) -> bool:
    for cc in ss:
        if isinstance(cc, str):
            return True
    return False


def allnumeric(ss: tp.Iterator) -> bool:
    for cc in ss:
        if not isinstance(cc, np.number):
            return False
    return True


def string_to_set(thestr: str) -> tuple[str, ...]:
    if isinstance(thestr, str):
        thestr = thestr.lower()
        thestr = re.sub("\(.*?\)", "", thestr)
        tems = list(set(thestr.split(",")))
        tems = [t.strip() for t in tems]
        return tuple(set(tems))
    return np.nan


def nonempty(x: tp.Iterator, a_set=None) -> bool:
    if isinstance(x, (type(pd.NA), type(np.nan))):
        return np.nan
    inter = a_set.intersection(x)

    if len(inter) == 1:
        return True
    elif len(inter) == 0:
        return False
    print(x, a_set)
    print(len(inter), inter)
    print(a_set)
    raise ValueError()


def process_data(input_df, target_col="Lastly, what was your favorite overall coffee?"):
    booltype = np.float32

    df = input_df.iloc[:, 3:].copy()
    df = df.drop(
        labels=[
            "What kind of dairy do you add? (Other)",
            "What is your ZIP code?",
            "Where else do you purchase coffee?",
            "Coffee A - Notes",  # Notes can leak info.
            "Coffee B - Notes",
            "Coffee C - Notes",
            "Coffee D - Notes",
            "Between Coffee A, Coffee B, and Coffee C which did you prefer?",
            "Between Coffee A and Coffee D, which did you prefer?",
            "Coffee D - Personal Preference",
            "Coffee A - Personal Preference",
            "Coffee B - Personal Preference",
            "Coffee C - Personal Preference",
            "Coffee D - Bitterness",  # possibly drop characteristics that are evaluated
            "Coffee D - Acidity",
            "Coffee B - Acidity",
            "Coffee B - Bitterness",
            "Coffee A - Acidity",
            "Coffee B - Acidity",
            "Coffee B - Bitterness",
            "Coffee A - Acidity",
            "Coffee C - Bitterness",
            "Coffee A - Bitterness",
            "Coffee C - Acidity",
            # "Before today's tasting, which of the following best described what kind of coffee you like?"
        ],
        axis=1,
    )
    boolean_cols = []
    cat_cols = []
    drop_cols = []
    numeric_cols = []
    set_cols = []

    string_cols = [
        "Other reason for drinking coffee",
        "Ethnicity/Race (please specify)",
        "What else do you add to your coffee?",
        "How else do you brew coffee at home?",
        "Please specify what your favorite coffee drink is",
    ]

    df[string_cols] = df[string_cols].astype(pd.StringDtype())

    for iii, c in enumerate(df.columns.copy()):
        unq = set(df[c].unique())
        if len(unq) <= 1:
            df.drop(labels=[c], axis=1, inplace=True)
            drop_cols.append(c)
        elif True in unq and len(unq) < 4:
            boolean_cols.append(c)
            df[c] = df[c].astype(booltype)
        elif len(unq) < 4 and "Yes" in unq and "No" in unq:
            df[c] = df[c].replace({"Yes": True, "No": False}).astype(booltype)
            boolean_cols.append(c)
        elif len(unq) < 20 and hasstring(unq):
            df[c] = (
                df[c]
                .apply(lambda x: x.lower() if isinstance(x, str) else "")
                .astype(pd.CategoricalDtype())
            )
            cat_cols.append(c)

        elif len(unq) <= 1:
            drop_cols.append(c)
        elif len(unq) <= 50 and allnumeric(unq):
            df[c] = df[c].astype(np.float32)
            numeric_cols.append(c)
        elif df[c].dtype == pd.StringDtype():
            df[c] = (
                df[c]
                .apply(lambda x: x.lower() if isinstance(x, str) else pd.NA)
                .astype(pd.StringDtype())
            )
        elif len(unq) > 20 and hasstring(unq) and not df[c].dtype == pd.StringDtype():
            df[c] = df[c].apply(string_to_set)
            sr = df[c].copy()
            set_cols.append(c)
            theset = set()
            for subs in sr.unique():
                if isinstance(subs, (type(pd.NA), type(np.nan))):
                    continue
                theset.update(subs)

            for newcol in theset:
                fset = set([newcol])
                pfunc = functools.partial(nonempty, a_set=fset)
                df[(c, newcol)] = sr.apply(pfunc).astype(booltype)
        else:
            print(c)
            print(len(unq), unq)
            print()
    df = df.drop(set_cols, axis=1)
    df = df.dropna(axis=1, thresh=3000)
    target = df[target_col]
    target = target[target != ""].copy().astype(pd.CategoricalDtype())
    print(target)
    df = df.loc[target.index].copy()
    df = df.drop([target_col], axis=1)
    df = df.reset_index(drop=True)
    target = target.reset_index(drop=True)
    cat_cols = []
    string_cols = []
    for iii, dt in enumerate(df.dtypes):
        # print(dt.name)
        if dt.name == "category":
            cat_cols.append(iii)
        if dt.name == "string":
            string_cols.append(string_cols)

    # nandf = df.isna().sum()
    # print(nandf[nandf>200])
    return df, target, cat_cols, string_cols


if __name__ == "__main__":
    import requests
    import io

    csv = io.BytesIO(requests.get("https://bit.ly/gacttCSV").content)
    _df = pd.read_csv(csv)
    data, target, ccols, scols = process_data(_df)
    print(data)
    for col in data.columns:
        print(col, data[col].dtype)
