freq = {}
with open('allowed_words.txt', 'r') as file:
    for word in map(str.strip, file.readlines()):
        freq[word] = 0

with open('5-letter-words.txt', 'r') as file:
    for line in file:
        items = line.strip().split()
        word = items[0].lower()

        if word not in freq:
            continue

        total = 0
        for info in items[1:]:
            year, count, works = info.split(',')
            total += int(count)

        freq[word] += total

for word, cnt in sorted(freq.items()):
    print(word, cnt)