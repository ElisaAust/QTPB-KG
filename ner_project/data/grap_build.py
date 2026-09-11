import matplotlib.pyplot as plt

labels = ['Feature', 'Food', 'Distribution', 'Habitat', 'Chinese name', 'Others (10 types)']
sizes = [8410, 6141, 4754, 3854, 1745, 8154]
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#BEBEBE']

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct='%1.2f%%',
    startangle=140,
    pctdistance=0.78
)

for text in texts:
    text.set_fontsize(11)
for autotext in autotexts:
    autotext.set_fontsize(9)

ax.set_title('Distribution of Entity Types in QTPB-NER Dataset', fontsize=13, pad=20)
plt.tight_layout()
plt.savefig('entity_pie.png', dpi=300, bbox_inches='tight')
plt.show()