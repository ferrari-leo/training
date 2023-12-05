count = 0
old_estimate = 0

while True:
    count += 1
    val = float(input('enter a number: '))
    cur_estimate = old_estimate + (1 / count) * (val - old_estimate)
    old_estimate = cur_estimate
    print(f'running average: {cur_estimate}')