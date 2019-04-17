from tkinter import *
from PIL import Image, ImageTk
from PyDictionary import PyDictionary


#CLICK FUNCTION
def click():
    entered_text = textentry.get()
    output.delete(0.0, END)
    try:
        definition = dictionary.meaning(entered_text)
    except:
        definition = "Shabd Astitvat Nahi"
    output.insert(END, definition)

# WINDOW
window = Tk()
window.title("Photo Sorter project - CSCI-5722 - (Bhalchandra Naik)")
window.configure(background="black")

# PICTURE
picture = Image.open('sample.jpg').resize((250, 250), Image.ANTIALIAS)
pictureTk = ImageTk.PhotoImage(picture)
Label(window,image=pictureTk, bg='black').grid(row=0, column=0, sticky=W)

# LABEL
Label(window, text="Enter the word you would like the definition for!", bg = "black", fg = "white", font = "none 12 bold").grid(row=1, column=0, sticky=W)

# TEXTBOXES
textentry = Entry(window, width=20, bg="white")
textentry.grid(row=2, column=0, sticky=W)

# SUBMIT BUTTON
Button(window, text="SUBMIT", width=6, command=click).grid(row=3, column=0, sticky=W)


# CREATE ANOTHER LABEL
Label(window, text="The definition of the given word is :", bg = "black", fg = "white", font = "none 12 bold").grid(row=4, column=0, sticky=W)

# OUTPUT TEXT BOX
output = Text(window, width=75, height=6, wrap=WORD, background="white")
output.grid(row=5, column=0, sticky=W)

# DICTIONARY
dictionary=PyDictionary()

# MAIL LOOP
window.mainloop()
