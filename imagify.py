import streamlit as st
import random
from PIL import Image
def imageify(n):
	img=Image.open("img/"+str(n)+".png")
	st.image(img,width=350)
	n=n+1
	return n



