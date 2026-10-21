import random

batPositions = []
chosenBatPositions = []
nonBatPositions = []
chosenNonBatPositions = []

name_input_file = "NamesWithClass.txt"
name_output_file = "NamesTraining.txt"
data_input_file = "wpc50.dat"
data_output_file = "wpc50Training.dat"


with open(name_input_file, "r", encoding="utf-8") as nameinfile, \
        open(name_output_file, "w", encoding="utf-8", newline="") as nameoutfile, \
        open(data_input_file, "r", encoding="utf-8") as datainfile, \
        open(data_output_file, "w", encoding="utf-8", newline="") as dataoutfile:

    namelines = nameinfile.readlines()
    datalines = datainfile.readlines()

    for index, line in enumerate(namelines):
        # pega linha sem endline
        record = line.rstrip()

        # index 81 tem o numero de classe
        if record[81] == '1':
            batPositions.append(index)
        else:
            nonBatPositions.append(index)

    # 38 bats 1001 non bats
    for x in range(37):
        pos = random.choice(batPositions)
        batPositions.remove(pos)
        chosenBatPositions.append(pos)
    for x in range(1000):
        pos = random.choice(nonBatPositions)
        nonBatPositions.remove(pos)
        chosenNonBatPositions.append(pos)

    for i in chosenBatPositions:
        nameoutfile.write(namelines[i])
        dataoutfile.write(datalines[i])

    for i in chosenNonBatPositions:
        nameoutfile.write(namelines[i])
        dataoutfile.write(datalines[i])
