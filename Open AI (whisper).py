#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install git+https://github.com/openai/whisper.git')


# In[2]:


import whisper
import os
import numpy as np
import torch


# In[3]:


torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# In[4]:


model = whisper.load_model("base", device=DEVICE)
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)


# In[5]:


audio = whisper.load_audio("ich_trinke_esse_gerne-mf.mp3")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(model.device)


# In[6]:


_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")


# In[7]:


options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16 = False)
result = whisper.decode(model, mel, options)
print(result.text)


# In[9]:


result = model.transcribe("ich_trinke_esse_gerne-mf.mp3")
print(result["text"])


# In[ ]:




