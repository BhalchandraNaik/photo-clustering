from tkinter import *
from sorting import *

def clustering():
    results = clusterMain(textentry.get()+'*')
    output.delete(0.0, END)
    output.insert(END, results)
    

window = Tk()
window.title("Photo Sorter project - CSCI-5722 - (Bhalchandra Naik)")

# FILE PATH
label_path = Label(window, text="Enter the directory where the files have been located : ")
label_path.grid(row=0, column=0, sticky=W)
textentry = Entry(window, width=20, bg="white")
textentry.grid(row=1, column=0, sticky=W)
Button(window, text="SUBMIT", width=6, command=clustering).grid(row=3, column=0, sticky=W)

# OUTPUT
output = Text(window, width=75, height=6, wrap=WORD, background="white")
output.grid(row=5, column=0, sticky=W)

# CALL MAIN LOOP
window.mainloop()