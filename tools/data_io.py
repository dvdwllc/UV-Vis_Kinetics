import pandas as pd
import numpy as np

def clean_cary_uvvis_data(filename):
    """
    Import and clean raw UV-Vis kinetics data from Cary BioVis software.

    :param filename: string.  path to raw UV-Vis kinetics datafile as csv.
    (Raw datafile exported from Cary BioVis spectrophotometer software.)

    :return: Pandas DataFrame
    """
    df = pd.read_csv(filename, skiprows=1)

    time_rows = df['Wavelength (nm)'].values == '[Time] '
    time_indices = np.argwhere(time_rows).flatten()
    time_vals = df['Abs'][time_indices].values.astype(float)

    non_data_cols = df['Wavelength (nm)'].values == '[Wavelength] '
    n_cols_to_delete = len(np.argwhere(non_data_cols).flatten())

    last_wavelength_index = np.argwhere(pd.isnull(df['Abs'].values))[0][0]

    df = df.drop(df.index[last_wavelength_index:])
    df = df.drop(df.columns[2::2], axis = 1)
    df = df.set_index('Wavelength (nm)')
    if n_cols_to_delete > 0:
        df = df.drop(df.columns[-n_cols_to_delete:], axis = 1)

    n_rows = df.shape[1]
    df.columns = range(n_rows)

    df = df.sort_index()

    return df, time_vals

def clean_agilent_uvvis_data(filename):
    """
    Import and clean raw UV-Vis kinetics data from Agilent UV-Vis software.

    :param filename: string.  path to raw UV-Vis kinetics datafile as .TXT.
    (Raw datafile exported from Agilent UV-Vis spectrophotometer software.)

    :return:Pandas DataFrame
    """
    df = pd.read_csv(filename, encoding='utf-16', skiprows=5, sep='\t')

    df = df.transpose()

    df_t = df.iloc[[0]].values[0]/60.0

    return df, df_t

def write_concs(run_concs, name):
    """
    write_concs(run_concs, name)

    Write time, [Br2], [NaBr3] as space-separated values to a file ('name.txt')

    Parameters
    ----------
    run_concs: a tuple containing three NumPy arrays: time, [Br2], [NaBr3]

    name: a string in quotes. This is the prefix of the file to be written.
        '.txt' will be appended to the end of the prefix.
    """
    with open(name + '.txt', 'w') as outfile:
        outfile.write('%s,%s,%s\n' % (TIME_LABEL,
                                      HALOGEN_LABEL,
                                      TRIHALIDE_LABEL))
        for i in range(len(run_concs[0])):
            outfile.write(str(run_concs[0][i]) + ',' +
                          str(run_concs[1][i]) + ',' +
                          str(run_concs[2][i]) + '\n')