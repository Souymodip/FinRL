import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import matplotlib.pyplot as plt
import numpy as np

def read_data(path):
    df = pd.read_csv(path)
    high = df['High'].str.replace('$', '', regex=False).astype(float).to_numpy()
    return high[::-1]


def simple_es(_x:np.ndarray, last=30):
    _x_ts = np.arange(len(_x))
    x_tx = _x_ts[:-last]

    x = _x[:-last]
    es = SimpleExpSmoothing(x)
    fitted_es = es.fit()

    # manual
    alpha = 0.9999999850988388
    l0 = x[0]
    ls =[l0]
    for i in range(1, len(x)):
        ls.append(alpha*x[i] + (1-alpha)*ls[-1])
    ls = np.array(ls)

    fitted_values = fitted_es.fittedvalues
    forcast = fitted_es.forecast(last)
    import pdb; pdb.set_trace()

    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(12, 6))
    ax.plot(x_tx[-last*4:], x[-last*4:], 'r-')
    ax.plot(_x_ts[-last*5:], _x[-last*5:], 'b--')
    ax.plot(x_tx[-last*4:], fitted_values[-last*4:], 'g')
    ax.plot(_x_ts[-last:], forcast, 'o')
    plt.show()


def main():
    csv_path = 'nasdaq/amazon_5y.csv'
    high = read_data(csv_path)
    simple_es(high)


if __name__ == '__main__':
    main()

