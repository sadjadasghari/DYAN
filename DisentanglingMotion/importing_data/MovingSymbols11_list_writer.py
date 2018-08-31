dir_file_train = '/netscratch/arenas/dataset/moving_symbols/MovingSymbols11_trainlist.txt'
dir_file_test = '/netscratch/arenas/dataset/moving_symbols/MovingSymbols11_testlist.txt'
movements = ['Backward', 'Circle', 'Down', 'Eight', 'Forward',
             'Infinite', 'Left', 'Right', 'Square', 'Triangle', 'Up']

trainfile = open(dir_file_train, 'w')
for i, mov in enumerate(movements):
    for j in range(1000):
        content = trainfile.write('%s/%s_video_%s.avi\n' % (mov, mov, j+1))
trainfile.close()

testfile = open(dir_file_test, 'w')
for k, mov in enumerate(movements):
    for j in range(100):
        content = testfile.write('%s/%s_video_%s.avi\n' % (mov, mov, j+1))
testfile.close()
