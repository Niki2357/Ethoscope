import os
import matplotlib.pyplot as plt


save_dir = "plots"  
os.makedirs(save_dir, exist_ok=True)

x_labels = ['Dec', 'Jan', "Feb", "Mar", 'Apr']
x = list(range(len(x_labels)))
y = [0, 0, 0, 0, 1]  # 1 = Introvert, 0 = Extrovert

plt.figure(figsize=(6, 4))

plt.plot(x, y, linestyle='-', color='#E7880B', marker='o', label='Extroversion')

plt.xticks(ticks=x, labels=x_labels) 
plt.yticks(ticks=[0, 1], labels=["Introvert", "Extrover"])

plt.xlabel('Month')
plt.ylabel('Personality Type')
plt.title('Introvert vs. Extrovert Plot')
plt.legend()

file_path = os.path.join(save_dir, "IE_labeled.png")
plt.savefig(file_path, bbox_inches='tight', dpi=300)

