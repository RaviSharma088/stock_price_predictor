import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import visualize
HEIGHT=500
WIDTH=600


root=tk.Tk()

canvas=tk.Canvas(root,height=HEIGHT,width=WIDTH)
canvas.pack()

def read_dataset(event):
    if compname.get() == 'Amazon':
        data=pd.read_csv("Amazon.csv")
    elif compname.get() == 'Google':
        data=pd.read_csv("Google.csv")
    elif compname.get() == 'Microsoft':
        data=pd.read_csv("Microsoft.csv")
    else:
        data=pd.read_csv("Apple.csv")
    
    label=tk.Label(lower_frame)
    label.place(relwidth=1,relheight=1)
    label["text"]= data.head()

background_image=tk.PhotoImage(file='see3.PNG')
background_label=tk.Label(root,image=background_image)
background_label.place(relwidth=1,relheight=1)

frame=tk.Frame(root,bg='#80c1ff',bd=5)
frame.place(relx=0.5,rely=0.1,relwidth=0.75,relheight=0.1,anchor='n')

#entry=tk.Entry(frame,font=40)
#entry.place(relwidth=0.65,relheight=1)
compname=ttk.Combobox(frame)
compname['values']=('Amazon','Google','Microsoft','Apple')
compname.place(relwidth=0.65,relheight=1)
compname.bind("<<ComboboxSelected>>",read_dataset)
compname.current()
    

button=tk.Button(frame,text="Start",font=40)
button.place(relx=0.7,relheight=1,relwidth=0.3)

lower_frame=tk.Frame(root,bg='#80c1ff',bd=10)
lower_frame.place(relx=0.5,rely=0.25,relwidth=0.75,relheight=0.6,anchor='n')

#label=tk.Label(lower_frame)
#label.place(relwidth=1,relheight=1)

root.mainloop()

