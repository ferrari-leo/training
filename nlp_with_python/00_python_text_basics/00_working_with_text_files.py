# region Part 1: String Formatting

library = [
    ("Author", "Topic", "Pages"),
    ("Twain", "Rafting", 601),
    ("Feynman", "Physics", 95),
    ("Hamilton", "Mythology", 144),
]

# tuple unpacking, padding, aligning
for author, topic, pages in library:
    print(f"{author:{10}} {topic:{30}} {pages:->{10}}")

# datetime strings (check https://strftime.org/ for aliases)
from datetime import datetime

today = datetime(year=2019, month=2, day=28)
print(f"{today:%B %d, %Y}")

# endregion
# region Part 2: Text files

myfile = open("test.txt")
myfile.read()
# you can't read a text file twice as the cursor is at the end of the file
# to counter this, reset the cursor:
myfile.seek(0)
myfile.read()
# set this as a variable to avoid needing to reset the cursor every time
myfile.seek(0)
content = myfile.read()
# remember to close the file
myfile.close()

# use newline tag to split lines
myfile = open("test.txt")
mylines = myfile.readlines()

for line in mylines:
    print(line)  # print whole lines
for line in mylines:
    print(line[0])  # print first character in a line
for line in mylines:
    print(line.split()[0])  # print first word in a line

myfile.close()

# allow read and write access (WARNING - w+ does a truncation on the original)
myfile = open("test.txt", "w+")
myfile.read()  # returns an empty string
myfile.write("brand new text")
myfile.seek(0)
myfile.read()

ls = [
    "hhhhhhhhhh",
    "hhhhhhhhhh",
    "hhhhhhhhhh",
    "hhhhhhhhhh",
    "hhhhhhhhhh", 'hhh',

    "hhhhhhhhhh",
    "hhhhhhhhhh",
    "hhhhhhhhhh",
]
# endregion
