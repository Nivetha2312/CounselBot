import streamlit as st
import random
from PIL import Image
def showit():
	img=Image.open("img/"+"coun"+".png")
	st.image(img,use_column_width=True)



