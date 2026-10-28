import matplotlib.pyplot as plt
from textblob import TextBlob


reviews = [
    "This product is absolutely amazing, I love it!",
    "Terrible quality, broke after one use.",    
    "Worst purchase ever. Do not buy this.",
    "It is okay, nothing special but works.",
    "Fast shipping and great packaging. Thanks!",
    "The battery life is terrible, dies in 2 hours.",
    "Highly recommended! Best value for money.",
    "I am very disappointed with the quality.",
    "Average performance. You get what you pay for.",
    "Five stars! Will buy again.",
    "Total waste of money. Broken on arrival.",
    "Fantastic design and easy to use.",
    "Customer support was rude and unhelpful.",
    "It arrived late but the product is fine.",
    "I honestly don't know how I feel about this.",
    "Superb experience, exceeding my expectations.",
    "The screen resolution is crystal clear, very impressive.",
    "Buttons feel cheap and plastic, not durable.",
    "Instruction manual is missing pages, very confusing.",
    "Battery lasts about 12 hours, which is decent.",
    "I was worried at first, but now I'm happy with it.",
    "Overpriced for what you get, look for alternatives.",
    "Simple design, does the job well.",
    "Delivery took 3 weeks, very slow service.",
    "Customer service solved my issue immediately, great job.",
    "Not compatible with Mac, which is annoying.",
    "Best investment I made this year!",
    "Absolutely terrible support, they never reply.",
    "The color is slightly different than the picture.",
    "Works perfectly with my setup, no issues.",
    "Box was damaged upon arrival but item is safe.",
    "Sent it back immediately, did not like it.",
    "A bit expensive, but the build quality is high.",
    "Nothing to complain about, solid product.",
    "The app crashes constantly, needs an update.",
    "Connectivity issues happen every day.",
    "Smooth interface and very responsive touch.",
    "Material feels premium to the touch.",
    "Too heavy to carry around in a pocket.",
    "Battery drains fast in cold weather.",
    "Got it on sale, absolute steal at this price.",
    "I regret buying this, honestly.",
    "Five stars for the product, one star for delivery.",
    "It is just okay, I might return it.",
    "Microphone quality is poor, people can't hear me.",
    "Sound is crisp and bass is deep.",
    "Charges very slowly, takes forever.",
    "User friendly interface, my grandma uses it easily.",
    "Documentation is confusing and poorly translated.",
    "Firmware update fixed all the bugs.",
    "Does exactly what it says on the box.",
    "Ugly design but very functional.",
    "My wife loves it, great gift idea.",
    "Refund process was a nightmare.",
    "Would recommend to a friend without hesitation.",
    "Not worth the hype, just average.",
    "Decent for the price, but not pro quality.",
    "The worst UI I have ever seen.",
    "Innovative features that I haven't seen before."
]

x_polarity = []
y_subjectivity = []
colors = []

print(f"--- STARTING BATCH SENTIMENT ANALYSIS ({len(reviews)} items) ---\n")

for i, text in enumerate(reviews):
    analysis = TextBlob(text)
    
    pol = analysis.sentiment.polarity
    sub = analysis.sentiment.subjectivity
    
    x_polarity.append(pol)
    y_subjectivity.append(sub)
    
    if pol > 0.1:
        color_code = 'green'
        category = "POSITIVE"
        term_color = "\033[92m"
    elif pol < -0.1:
        color_code = 'red'
        category = "NEGATIVE"
        term_color = "\033[91m"
    else:
        color_code = 'blue'
        category = "NEUTRAL "
        term_color = "\033[94m" 

    colors.append(color_code)
    
    reset = "\033[0m"
    print(f"#{i+1:02d} | Polarity: {pol: .2f} | Subjectivity: {sub: .2f} | {term_color}{category}{reset}")

print("\nGenerating Scatter Plot visualization...")

plt.figure(figsize=(10, 6))

plt.scatter(x_polarity, y_subjectivity, c=colors, s=100, alpha=0.7)

plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.5) 
plt.axvline(0, color='black', linestyle='-', linewidth=1)     

plt.title(f'Sentiment Analysis Results ({len(reviews)} reviews)', fontsize=14)
plt.xlabel('Polarity (Negative <-> Positive)', fontsize=12)
plt.ylabel('Subjectivity (Objective <-> Subjective)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)

plt.xlim(-1.1, 1.1)
plt.ylim(-0.1, 1.1)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Positive'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Negative'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Neutral')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.show()