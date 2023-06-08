import ast
import matplotlib.pyplot as plt

with open('/home/miran045/reine097/projects/AlexNet_Abrol2021/results/model03/distributions.txt') as f:
    data = f.read()

d = ast.literal_eval(data)

plt.hist(d[1],
         alpha=0.5,
         label='score of 1')
plt.hist(d[2],
         alpha=0.5,
         label='score of 2')
plt.hist(d[3],
         alpha=0.5,
         label='score of 3')
plt.hist(d[4],
         alpha=0.5,
         label='score of 4')

plt.legend(loc='upper right')
plt.title('Actual QC motion score vs. frequency of prediction')

plt.savefig('/home/miran045/reine097/projects/AlexNet_Abrol2021/doc/qc_motion_score_prediction.png')
plt.show()
