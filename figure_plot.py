import numpy as np
import matplotlib.pyplot as plt

topics = ['climate change', 'EU', 'royal family', 'UK politics', 'trade tariffs', 'multiculturalism','vaccines', 'Donald Trump', 'Joe Biden', 'privacy','coronavirus','immigration', 'healthcare', 'education', 'BLM']

def plotBarChart(topic, pos_counts, neg_counts):
    """
    Plot Bar Chart to show the analysis result for different topics
    """
    outlets = ("BBC", "DailyMail", "Guardian", "Metro", "Mirror", "Reuters", "Independent", "Sun")

    fig, ax = plt.subplots()
    y_pos = np.arange(len(outlets))
    bar_width = 0.20
    opacity = 0.8

    rects1 = plt.barh(y_pos, neg_counts, bar_width,
    alpha=opacity,
    color='#ff4542',
    label='Negative')

    rects3 = plt.barh(y_pos + bar_width, pos_counts, bar_width,
    alpha=opacity,
    color='#5eff7c',
    label='Positive')

    plt.yticks(y_pos, outlets)
    plt.xlabel('News Sentiment Percentage')
    plt.title('News Sentiment Analysis: '+str(topic))
    plt.legend()

    plt.tight_layout()
    plt.show()

