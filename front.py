from tkinter import *
from final import *
import numpy as np
from numpy import array
from numpy import reshape



class Window(Frame):


    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()


    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget
        self.master.title("Diabetes Prediction System")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)



fields = 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'

global text
global a
global b
b=[]
a=[]
text=[]
def fetch(entries):
    for entry in entries:
        field = entry[0]
       # text  = entry[1].get()
        text = entry.get().split(",")
        a.append(text)
        print('%s: "%s"' % (field, text))
        
    b = np.reshape(a, (1,-2))
    b = (",".join(map(str, b)))
    print(b)
        
        
         

def makeform(root, fields):
    entries = []
    for i in fields:
        row = Frame(root)
        lab = Label(row, width=25, text=i, anchor='w')
        ent = Entry(row)    #get entry

        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries.append((i, ent))

    return entries


if __name__ == '__main__':
    root = Tk()
    ents = makeform(root, fields)

    root.bind('<Return>',(lambda event, e=ents: fetch(e)))  #search this
    #print(ents)
    b1 = Button(root, text='Show', command= (lambda  e= ents: fetch(e)))
    #(lambda e=ents: fetch(e)))
    b1.pack(side=LEFT, padx=5, pady=5)
    b2 = Button(root, text='Exit', command=(root.destroy))
    b2.pack(side=LEFT, padx=5, pady=5)
   # b3 = Button(root, text='Quit', command=root.destroy)
    #b3.pack(side=LEFT, padx=5,pady=5)


#size of the window
root.geometry("800x600")
app = Window(root)
#backend()
obj = backend.knn(backend)
obj1 = backend.predictor(b)
root.mainloop()