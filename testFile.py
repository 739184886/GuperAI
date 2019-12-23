import os
# testFile = open("test.txt","w+",encoding='utf-8')
# testFile.write("test")
# #将指针移动到文件首，防止读不到数据；
# # 因为write完成后，指针在最后录入信息的位置，读的时候指针会向后读取数据，这样造成不到数据
# testFile.seek(0,0)
# str = testFile.read()
# print(str)
# testFile.close();
#
# #此方式重新打开了文件，可以不用seek()
# with open('test.txt','r') as f:
#     print(f.read())
##################
# print(os.getcwd())
filePath = os.getcwd()
if not os.path.exists(filePath):
      os.makedirs(filePath)
      print(filePath)

file =  open(filePath + "/file.txt")