dir_file_train = './moving_symbols/MovingSymbols2_trainlist.txt'
dir_file_test = './moving_symbols/MovingSymbols2_testlist.txt'

trainfile = open(dir_file_train, 'w')
for i in range(5000):
    trainfile.write('Horizontal/Horizontal_video_{}.avi\n'.format(i+1))
for j in range(5000):
    trainfile.write('Vertical/Vertical_video_{}.avi\n'.format(j+1))
trainfile.close()

testfile = open(dir_file_test, 'w')
for i in range(500):
    testfile.write('Horizontal/Horizontal_video_{}.avi\n'.format(i+1))
for j in range(500):
    testfile.write('Vertical/Vertical_video_{}.avi\n'.format(j+1))
testfile.close()
