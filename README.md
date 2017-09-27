# Saito
Internship on image and music classification according to emotions.  
All code should be self-documented.

## Image
Edit `config.py` accordingly and launch the main process by running `main.py`  
`segmentation.py` is not bug-free.

## Music
First edit `config.py` as you want.  
Use `getdata.py` to retrieve only the relevant data from the NicoNico dataset (read documentation inside code to tune what criterion should be used to select relevant data).  
Use `getaudio.py` to retrieve video from data and convert it to audio. (Warning : NicoNico is very slow and often interrupts downloads).  
Use `melody.py` functions to extract melody from audio and manipulate it.  
Use `main.py` as a base for parsing downloaded data.

For details see the documentation inside each source file.
