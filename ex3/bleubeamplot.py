import matplotlib.pyplot as plt

#fibonacci beam_vals
beam_val = [0,1,2,3,5,8,13,21,34,55]

#bleu values
bleu_val = [3.4,3.4,3.6,3.4,3.2,3.0,2.6,2.4,2.3,2.1]

plt.plot(beam_val,bleu_val)
plt.title('Impact of beam size on translation quality')
plt.xlabel('Beam Values')
plt.ylabel('Bleu')
plt.savefig('bleubeamplot.png')
plt.show()


