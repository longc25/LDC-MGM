import pandas as pd

# https://stackoverflow.com/questions/16490261/python-pandas-write-dataframe-to-fixed-width-file-to-fwf

from tabulate import tabulate
import numpy as np


def to_fwf(df, fname):
    content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain")
    open(fname, "w").write(content)


def save_outcat(outcat_name, outcat):
    outcat_colums = outcat.shape[1]
    pd.DataFrame.to_fwf = to_fwf
    if outcat_colums == 10:
        detected_outcat_name = outcat_name
        table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum',
                       'Volume']
        dataframe = pd.DataFrame(outcat, columns=table_title)
        dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Cen1': 3, 'Cen2': 3,
                                     'Size1': 3, 'Size2': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(detected_outcat_name, sep='\t', index=False)
        # dataframe.to_fwf(detected_outcat_name)


    if outcat_colums == 13:
        print(13)
        detected_outcat_name = outcat_name
        table_title = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak', 'Sum',
                       'Volume']
        dataframe = pd.DataFrame(outcat, columns=table_title)
        dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
                                     'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(detected_outcat_name, sep='\t', index=False)
        # dataframe.to_fwf(detected_outcat_name)

    if outcat_colums == 11:
        fit_outcat_name = outcat_name
        fit_outcat = outcat
        table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'theta', 'Peak',
                       'Sum', 'Volume']
        dataframe = pd.DataFrame(fit_outcat, columns=table_title)
        dataframe = dataframe.round(
            {'ID': 0, 'Peak1': 3, 'Peak2': 3, 'Cen1': 3, 'Cen2': 3, 'Size1': 3, 'Size2': 3, 'theta': 3, 'Peak': 3,
             'Sum': 3, 'Volume': 3})
        dataframe.to_csv(fit_outcat_name, sep='\t', index=False)

        # dataframe.to_fwf(fit_outcat_name)


if __name__ == '__main__':
    pd.DataFrame.to_fwf = to_fwf

    # %%
    # df = pd.DataFrame({"ID": [1, 2, 3, 4, 50000], "a": np.random.rand(5), "b": [1.0, 2.01, 3.02, 4.03, 25.006]})
    df = pd.read_csv(r'F:\LDC_python\detection\R2_data\data_9\0175+000\0175+000_L\LDC_outcat_wcs.txt', sep='\t')
    # pd.DataFrame.to_fwf = to_fwf

    # %%
    # df = pd.DataFrame({"ID": [1, 2, 3, 4, 50000], "a": np.random.rand(5), "b": [1.0, 2.01, 3.02, 4.03, 25.006]})
    df.to_fwf("fwf1.txt")
