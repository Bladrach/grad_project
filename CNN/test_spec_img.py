from learn_spec_img import *
import matplotlib.pyplot as plt


directory = "./saved_models"
files = list(filter(os.path.isfile, glob.glob(directory + "/*.pth")))
files.sort(key=lambda x: os.path.getmtime(x))
acc_list = []
epoch_list = []
for filename in files:
    net.load_state_dict(torch.load(filename))
    correct = 0
    total = 0
    print(filename)
    with torch.no_grad():
        for data in testloader:
            spec = data[0].unsqueeze(1).to(device)  # unsqueeze is used to add channel to make our input as (batch_size x 1 x imgHeight x imgWidth)
            label = data[1].to(device)
            output = net(spec)

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    acc_list.append(100 * (correct/total))
    epoch_list.append(int(filename.split("_")[-1].split(".")[0]))

elapsedTimeFile = open('elapsed_time.txt')  # open the text file
elapsedTime = float(elapsedTimeFile.read()) # elapsed time is taken from the text file
elapsedTimeFile.close()  # close the text file

fig = plt.figure()
plt.plot(epoch_list, acc_list)
plt.title("Elapsed Time ({} minutes), Test Accuracy ({:.3f} %)".format(float(elapsedTime), float(acc_list[-1])))
plt.grid()
plt.xticks(range(0, 26, 1))
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
fig.savefig("TrainingHistory.jpg")
plt.show()
