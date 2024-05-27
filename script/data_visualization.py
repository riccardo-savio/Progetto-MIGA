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
    # Create the plot
    plt.hist(data['rating'], rwidth=.8, bins=np.arange(1, 5+2) - 0.5)
    plt.xticks([1, 2, 3, 4, 5])
    if saveFig:
        plt.savefig(figName)
    plt.show()

def main():
    hist_rev_ratings()

if __name__ == "__main__":
    main()
    