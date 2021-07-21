from django.shortcuts import render
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from model_.src.args import get_parser
import pickle
from model_.src.model import get_model
from torchvision import transforms
from model_.src.utils.output_utils import prepare_output
from PIL import Image
import time
import requests
from io import BytesIO
import random
from collections import Counter
import glob
from shutil import move
from django.core.files.storage import FileSystemStorage
from gensim.models import FastText
from gensim.test.utils import datapath
from gensim.utils import tokenize
from gensim import utils
from simple_image_download import simple_image_download as simp

def homepage(request):
    return render(request, 'recommender/index.html')

def members(request):
    return render(request, 'recommender/members.html')

def remove_transparency(im, bg_colour=(255, 255, 255)):

    if im.mode in ('RGBA', 'LA'):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im

def create_embeddings(sentence,model_fasttext):
  embeddingList = []
  countFound = 0
  for wordx in sentence:
    try:
      vector1 = model_fasttext.wv[wordx]
      embeddingList.append(vector1)
      countFound+=1
    except:
      continue
  sumEmbeddings = sum(embeddingList)
  #print("count found {}".format(countFound))
  return np.true_divide(sumEmbeddings, countFound)

def recommend(input,recipe_embeddings,recipe_name,model_fasttext):
    #ingre = predict()
    ingre = input
    embedding = create_embeddings(ingre,model_fasttext)

    a = embedding
    dishtances = {}

    for i in range(len(recipe_embeddings)):
        try:
            dn = recipe_name[i]
            b = recipe_embeddings[i]
            cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            if cos_sim not in dishtances.values():
                dishtances[i] = cos_sim
        except:
            continue

    dishtances_2 = {k: v for k, v in sorted(dishtances.items(), key=lambda item: item[1], reverse = True)}
    mostSimilarDishes = []
    countSim = 0

    for el in dishtances_2.keys():
        mostSimilarDishes.append(el)
        countSim+=1
        if countSim==3:
            break

    allSuggestedDishNames = []

    for simIndex in mostSimilarDishes:
        dName = recipe_name[simIndex]
        dishName = dName #" ".join([w for w in dName.split() if w.lower()!='recipe'])
        dishNameShort = dName#" ".join(dishName.split()[-2:])
        allSuggestedDishNames.append(dishNameShort)

    return allSuggestedDishNames


def click(request):
    if request.method == 'POST' and request.FILES['img']:
        myfile = request.FILES['img']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        path = fs.path(filename)

        use_gpu = False
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'

        with open('static/ingr_vocab.pkl', 'rb') as f:
            ingrs_vocab = pickle.load(f)
        with open('static/instr_vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        with open('static/recipe-embeddings.pkl', 'rb') as f:
            recipe_embeddings = pickle.load(f)
        with open('static/recipe-names.pkl', 'rb') as f:
            recipe_name = pickle.load(f)

        greedy = [True, False, False, False]
        beam = [-1, -1, -1, -1]
        temperature = 1.0
        numgens = len(greedy)

        ingr_vocab_size = len(ingrs_vocab)
        instrs_vocab_size = len(vocab)
        output_dim = instrs_vocab_size
        import sys; sys.argv=['']; del sys
        args = get_parser()
        args.maxseqlen = 15
        args.ingrs_only=False
        model = get_model(args, ingr_vocab_size, instrs_vocab_size)
        # Load the trained model parameters
        model_path = 'static/modelbest.ckpt'
        model.load_state_dict(torch.load(model_path, map_location=map_loc))
        model.to(device)
        model.eval()
        model.ingrs_only = False
        model.recipe_only = False

        transf_list_batch = []
        transf_list_batch.append(transforms.ToTensor())
        transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225)))
        to_input_transf = transforms.Compose(transf_list_batch)

        image = Image.open(path)

        if image.mode != 'RGB':
            #If you just convert to RGB, sometimes the background ends up black, though I'm not sure if it effects the model
            image = remove_transparency(image)
            image = image.convert('RGB')

        transf_list = []
        transf_list.append(transforms.Resize(256))
        transf_list.append(transforms.CenterCrop(224))
        transform = transforms.Compose(transf_list)

        image_transf = transform(image)
        image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)

        num_valid = 1
        inpt = []

        for i in range(numgens):
                with torch.no_grad():
                    outputs = model.sample(image_tensor, greedy=greedy[i],
                                        temperature=temperature, beam=beam[i], true_ingrs=None)

                ingr_ids = outputs['ingr_ids'].cpu().numpy()
                recipe_ids = outputs['recipe_ids'].cpu().numpy()

                outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)
                print(valid)

                if valid['is_valid']:
                    inpt = outs['ingrs']
                    instruction = outs['recipe']
                    title = outs['title']
                else:
                    print ("Not a valid recipe!")
                    print ("Reason: ", valid['reason'])

                break
        if len(inpt) <= 4:
            return render(request, 'recommender/index_.html', {'uploaded_file_url':uploaded_file_url, 'ingredients':  ['*************'],'suggestions':['************'], 'instruction':['**********'], 'title':'RECIPE - INVALID-RECIPE', 'links':[]})


        model_fasttext = FastText.load('static/model_fasttext.model')
        suggestions = recommend(inpt,recipe_embeddings,recipe_name,model_fasttext)
        response = simp.simple_image_download
        for i in range(3):
            suggestions[i] = suggestions[i].replace(',',' and')
            suggestions[i] = suggestions[i].replace(' ','_')

        urls = []
        for rep in suggestions:
            urls.append(response().urls(rep,1)[0])

        links = []

        for i in range(0,3):
            links.append((urls[i],suggestions[i]))

        return render(request, 'recommender/index_.html', {'links': links , 'uploaded_file_url':uploaded_file_url, 'ingredients':  inpt,'suggestions':suggestions, 'instruction':instruction, 'title': title})
