# Read file in python
filename = "input"
in_file = open(filename, 'r')
line = in_file.readline()

# lines in file
# Python readline() yield extra "\n" in between the lines when reading from a text file
while line != "":
    words = line.split()

    num_words = len(words)
    num_letters = len(line) - num_words

    print(line, "\n --->", "word: ", num_words, "letters:", num_letters, "\n")
    line = in_file.readline()