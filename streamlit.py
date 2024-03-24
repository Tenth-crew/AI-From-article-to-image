import os

import PIL
import torch
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, BartForConditionalGeneration, BartTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, PNDMScheduler, StableDiffusionPipeline, DPMSolverMultistepScheduler
from bigdl.llm.optimize import low_memory_init, load_low_bit
import torch.nn.functional as F
import streamlit as st
import time 
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# build model
with low_memory_init():
    summarizer = BartForConditionalGeneration.from_pretrained('./checkpoints/bart/', torch_dtype="auto", trust_remote_code=True)

summarizer = load_low_bit(summarizer, './checkpoints/bart/')

bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

diffusion_pipeline_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(diffusion_pipeline_id)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


# build data
ARTICLE = """ Trump’s lawyers acknowledged Monday that he was struggling to find an insurance company willing to underwrite his $454 million bond. Privately, Trump had been counting on Chubb, which underwrote his $91.6 million bond to cover the E. Jean Carroll judgment, to come through, but the insurance giant informed his attorneys in the last several days that that option was off the table.
Trump’s team has sought out wealthy supporters and weighed what assets could be sold – and fast. The presumptive GOP presidential nominee himself has become increasingly concerned about the optics the March 25 deadline could present – especially the prospect that someone whose identity has long been tied to his wealth would confront financial crisis. Trump has continued to privately lash out at the New York Attorney General Letitia James and Judge Arthur Engoron over the matter, these sources told CNN.
Shortly before 6:30 a.m. Tuesday, Trump took those grievances public, posting on his social media platform eight times within two hours about the deadline, arguing that he shouldn’t have to put up the money and worrying that he “would be forced to mortgage or sell Great Assets, perhaps at Fire Sale prices, and if and when I win the Appeal, they would be gone.”
“Does that make sense? WITCH HUNT. ELECTION INTERFERENCE!” the former president wrote.
“These baseless innuendos are pure bullsh*t,” Trump campaign spokesman Steven Cheung said in a statement Tuesday. “President Trump has filed a motion to stay the unjust, unconstitutional, un-American judgment from New York Judge Arthur Engoron in a political Witch Hunt brought by a corrupt Attorney General. A bond of this size would be an abuse of the law, contradict bedrock principals of our Republic, and fundamentally undermine the rule of law in New York.”
"""

# user_input =st.text_input('请输入文章：')
# ARTICLE = user_input

btime1 = time.time()
# summarize
inputs = bart_tokenizer(ARTICLE, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = summarizer.generate(inputs['input_ids'])
summary = bart_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
# print(summary)
st.write(f'输出的摘要为：{summary}')
st.write(f'输出摘要耗时：{time.time()-btime1}')
# draw image
btime2 = time.time()
image = pipe(summary[0]).images[0]

save_directory = "./pictures/trump.png"
image.save("./pictures/trump.png")
st.write(f'输出的摘要为：')
st.image(save_directory)
st.write(f'输出图片耗时：{time.time()-btime2}')