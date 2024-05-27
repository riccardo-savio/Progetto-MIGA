def plot_rev_timedist(saveFig=False, figName='imgs/rev_time_dist.png'):
    """
    Plot the distribution of reviews over time.
    """
    from data_gathering import get_processed_reviews
    import pandas as pd
    import matplotlib.pyplot as plt
    # Load the data
    data = get_processed_reviews()

    data = data[['timestamp', 'rating']].groupby('timestamp').count().reset_index()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['rating'])
    plt.title('Distribuzione di Recensioni nel Tempo')
    plt.xlabel('Data')
    plt.ylabel('Numero di Recensioni')
    plt.grid(True)
    if saveFig:
        plt.savefig(figName)
    plt.show()

def hist_rev_ratings(saveFig=False, figName='imgs/rev_rating_dist.png'):
    """
    Plot the distribution of review ratings.
    """
    from data_gathering import get_processed_reviews
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    # Load the data
    data = get_processed_reviews()
    def millions(x, pos):
        """The two args are the value and tick position."""
        return '{:1.1f}M'.format(x*1e-6)
    # Create the plot
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(millions)
    ax.hist(data['rating'], rwidth=.8, bins=np.arange(1, 5+2) - 0.5)
    ax.set_xticks(np.arange(1, 5+1))
    ax.set_title('Distribuzione delle Recensioni')
    ax.set_xlabel('Valutazione')
    ax.set_ylabel('Numero di Recensioni')
    if saveFig:
        plt.savefig(figName)
    plt.show()

def main():
    hist_rev_ratings(True)

if __name__ == "__main__":
    main()
    