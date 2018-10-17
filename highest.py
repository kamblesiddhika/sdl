import sys
def write_message():

   f = open("D:\Java aptech\Jeopardy\src\jeopardy\player.txt","a+")
   f.write(sys.argv[0])
   print"asds"
   f.close()

write_message()