my_list = ['orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange', 'banana','banana','banana','banana','banana','banana','banana','banana','banana','banana','banana','banana','banana','banana','banana','banana','banana',]

my_dict = {}
for fruit in my_list:
    if fruit in my_dict:
        my_dict[fruit] += 1
    else:
        my_dict[fruit] = 1
