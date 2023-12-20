import re

text = "The phone number of the agent is 408-555-1234. Call soon!"

print("408-555-1234" in text)

pattern = "phone"
mymatch = re.search(pattern, text)
print(mymatch.span())
print(mymatch.start())
print(mymatch.end())

# find multiple matches
text = "my phone is a new phone"
match = re.search(pattern, text)
print(match.span())
all_matches = re.findall(pattern, text)
print(all_matches)

for match in re.finditer(pattern, text):
    print(match.span())

# if you don't know the exact characters, use designations
# \d - digit
# \w - alphanumeric
# \s - whitespace
# \D - non-digit
# \W - non-alphanumeric
# \S - non-whitespace

text = "my phone number is 777-555-1234"
pattern = r"\d\d\d-\d\d\d-\d\d\d\d"
phone_number = re.search(pattern, text)
print(phone_number.group())

# quantifiers
# + one or more times
# {3} exactly 3 times
# {2,4} 2 to 4 times
# {3, } 3 or more times
# * 0 or more times
# ? once or none

phone_number = re.search(r"\d{3}-\d{3}-\d{4}", text)
print(phone_number.group())

# grabbing separate groups
phone_number = re.search(r"(\d{3})-(\d{3})-(\d{4})", text)
print(phone_number.group(1))

# pipe operator
re.search(r"man|woman", "This man was here").group()

# wildcard
print(re.findall(r".at", "the cat in the hat sat splat"))
print(re.findall(r"..at", "the cat in the hat sat splat"))

# starts/ends with
print(re.findall(r"\d$", "This ends with a number 2"))
print(re.findall(r"^\d", "1 is the loneliest number"))

# character exclusions
phrase = "there are 3 numbers 34 inside 5 this sentence"
print(re.findall(r"[^\d]+", phrase))

test_phrase = "this is a string! but it has punctuation. how to remove it?"
print(re.findall(r"[^!.? ]+", test_phrase))

text = "only find the hyphen-words. where are the long-ish dash words?"
print(re.findall(r"[\w]+-[\w]+", text))
